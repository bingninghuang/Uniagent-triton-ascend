import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_vec_cores():
    try:
        import torch_npu
        return torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
    except Exception:
        return 48


@triton.jit
def _grouped_gemm_dequant_kernel(x_ptr, w_ptr, xs_ptr, ws_ptr, out_ptr,
                                 starts_ptr, counts_ptr,
                                 K: tl.constexpr, N: tl.constexpr,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                 BLOCK_K: tl.constexpr):
    """Int8 grouped GEMM with per-token x-scale and per-channel weight-scale,
    writing float32 hidden [m, n] (rows are placed at global expert offsets)."""
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    start = tl.load(starts_ptr + pid_e)
    t = tl.load(counts_ptr + pid_e)

    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offs < t
    rows = start + row_offs
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = x_ptr + rows[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + pid_e * (K * N) + offs_k[:, None] * N + cols[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0)
        b = tl.load(b_ptrs, mask=col_mask[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    xs = tl.load(xs_ptr + rows, mask=row_mask, other=0.0)
    ws = tl.load(ws_ptr + pid_e * N + cols, mask=col_mask, other=0.0)
    val = (acc.to(tl.float32) * xs[:, None]) * ws[None, :]
    out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
    tl.store(out_ptrs, val, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _swiglu_rowmax_kernel(hid_ptr, max_ptr,
                          M, N: tl.constexpr, HALF: tl.constexpr,
                          BLOCK_HALF: tl.constexpr):
    """Per-row absmax of swiglu(hidden). Used only for per-group quant mode."""
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    per_prog = (M + nprogs - 1) // nprogs
    r0 = pid * per_prog
    r1 = tl.minimum(r0 + per_prog, M)
    offs = tl.arange(0, BLOCK_HALF)
    cmask = offs < HALF
    for r in range(r0, r1):
        l = tl.load(hid_ptr + r * N + offs, mask=cmask, other=0.0)
        rr = tl.load(hid_ptr + r * N + HALF + offs, mask=cmask, other=0.0)
        v = (l / (1.0 + tl.exp(-l))) * rr
        mrow = tl.max(tl.abs(v))
        tl.store(max_ptr + r, mrow)


@triton.jit
def _group_scale_kernel(max_ptr, scale_ptr, starts_ptr, counts_ptr,
                        BLOCK: tl.constexpr):
    """Per-group absmax -> common group scale written into out_scale rows."""
    pid_e = tl.program_id(0)
    start = tl.load(starts_ptr + pid_e)
    t = tl.load(counts_ptr + pid_e)
    gmax = 0.0
    for c0 in range(0, t, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        cmask = offs < t
        v = tl.load(max_ptr + start + offs, mask=cmask, other=0.0)
        gmax = tl.maximum(gmax, tl.max(v))
    gscale = gmax / 127.0
    for c0 in range(0, t, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        cmask = offs < t
        tl.store(scale_ptr + start + offs, gscale, mask=cmask)


@triton.jit
def _swiglu_quant_kernel(hid_ptr, scale_ptr, out_ptr,
                         M, N: tl.constexpr, HALF: tl.constexpr,
                         QMODE: tl.constexpr, BLOCK_HALF: tl.constexpr):
    """SwiGLU activation + quantize to int8. QMODE 0 = per-token dynamic scale,
    QMODE 1 = per-group scale (precomputed into scale_ptr rows)."""
    pid = tl.program_id(0)
    nprogs = tl.num_programs(0)
    per_prog = (M + nprogs - 1) // nprogs
    r0 = pid * per_prog
    r1 = tl.minimum(r0 + per_prog, M)
    offs = tl.arange(0, BLOCK_HALF)
    cmask = offs < HALF
    if QMODE == 0:
        for r in range(r0, r1):
            l = tl.load(hid_ptr + r * N + offs, mask=cmask, other=0.0)
            rr = tl.load(hid_ptr + r * N + HALF + offs, mask=cmask, other=0.0)
            v = (l / (1.0 + tl.exp(-l))) * rr
            mrow = tl.max(tl.abs(v))
            scv = tl.maximum(mrow, 1e-10) / 127.0
            tl.store(scale_ptr + r, scv)
            scv2 = tl.load(scale_ptr + r + tl.arange(0, 1))
            l2 = tl.load(hid_ptr + r * N + offs, mask=cmask, other=0.0)
            rr2 = tl.load(hid_ptr + r * N + HALF + offs, mask=cmask, other=0.0)
            v2 = (l2 / (1.0 + tl.exp(-l2))) * rr2
            qf = v2 / scv2
            q = tl.where(qf >= 0.0, qf + 0.5, qf - 0.5).to(tl.int32)
            q = tl.minimum(tl.maximum(q, -128), 127)
            tl.store(out_ptr + r * HALF + offs, q.to(tl.int8), mask=cmask)
    else:
        for r in range(r0, r1):
            scv = tl.load(scale_ptr + r + tl.arange(0, 1))
            l = tl.load(hid_ptr + r * N + offs, mask=cmask, other=0.0)
            rr = tl.load(hid_ptr + r * N + HALF + offs, mask=cmask, other=0.0)
            v = (l / (1.0 + tl.exp(-l))) * rr
            qf = v / scv
            q = tl.where(qf >= 0.0, qf + 0.5, qf - 0.5).to(tl.int32)
            q = tl.minimum(tl.maximum(q, -128), 127)
            tl.store(out_ptr + r * HALF + offs, q.to(tl.int8), mask=cmask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.vec_cores = _get_vec_cores()

    def forward(self, x: torch.Tensor, weight: list, weight_scale: list,
                x_scale: torch.Tensor, group_list: torch.Tensor,
                smooth_scale=None, weight_assist_matrix=None, bias=None,
                dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0,
                group_list_type=0, tuning_config=None):
        device = x.device
        x = x.contiguous()
        w3 = weight[0]
        if w3.dim() != 3:
            w3 = torch.stack(list(weight))
        w3 = w3.contiguous()
        ws3 = weight_scale[0]
        if ws3.dim() != 2:
            ws3 = torch.stack(list(weight_scale))
        ws3 = ws3.contiguous()
        x_scale = x_scale.contiguous()

        m, k = x.shape
        e, _kw, n = w3.shape
        half = n // 2

        gl = group_list.tolist()
        starts = [0] * e
        counts = [0] * e
        prev = 0
        if group_list_type == 0:
            for i in range(e):
                starts[i] = prev
                counts[i] = gl[i] - prev
                prev = gl[i]
        else:
            for i in range(e):
                starts[i] = prev
                counts[i] = gl[i]
                prev += gl[i]
        starts_t = torch.tensor(starts, dtype=torch.int32, device=device)
        counts_t = torch.tensor(counts, dtype=torch.int32, device=device)

        hidden = torch.empty((m, n), dtype=torch.float32, device=device)
        out = torch.empty((m, half), dtype=torch.int8, device=device)
        out_scale = torch.empty((m,), dtype=torch.float32, device=device)

        # ---- grouped int8 GEMM + dequant scales ----
        BLOCK_K = 64
        BLOCK_N = 128
        max_t = max(counts) if counts else 0
        if max_t > 64:
            BLOCK_M = 64
        elif max_t > 32:
            BLOCK_M = 32
        else:
            BLOCK_M = 16
        grid = (e, triton.cdiv(max_t, BLOCK_M), triton.cdiv(n, BLOCK_N))
        _grouped_gemm_dequant_kernel[grid](
            x, w3, x_scale, ws3, hidden, starts_t, counts_t,
            K=k, N=n, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

        # ---- SwiGLU + quantize ----
        BH = triton.next_power_of_2(half)
        g = (min(m, self.vec_cores),)
        if quant_mode == 1:
            rowmax = torch.empty((m,), dtype=torch.float32, device=device)
            _swiglu_rowmax_kernel[g](hidden, rowmax, m,
                                     N=n, HALF=half, BLOCK_HALF=BH)
            _group_scale_kernel[(e,)](rowmax, out_scale, starts_t, counts_t,
                                      BLOCK=256)
        _swiglu_quant_kernel[g](hidden, out_scale, out, m,
                                N=n, HALF=half,
                                QMODE=0 if quant_mode == 0 else 1,
                                BLOCK_HALF=BH)
        return out, out_scale