import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm_kernel(x_ptr, w_ptr, xs_ptr, ws_ptr, hid_ptr, gl_ptr,
                         GLT: tl.constexpr, E: tl.constexpr, EBLK: tl.constexpr,
                         NBLK: tl.constexpr,
                         K: tl.constexpr, N: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Int8 grouped GEMM (per expert) with per-token x-scale and per-channel
    weight-scale, writing float32 hidden[m, n] at global row positions.
    Grid: (E * NBLK,). Group offsets are derived in-kernel from group_list."""
    pid = tl.program_id(0)
    e = pid // NBLK
    nblk = pid % NBLK

    offs_e = tl.arange(0, EBLK)
    glv = tl.load(gl_ptr + offs_e, mask=offs_e < E, other=0)
    cur = tl.sum(tl.where(offs_e == e, glv, 0), axis=0)
    if GLT == 0:
        start = tl.sum(tl.where(offs_e == e - 1, glv, 0), axis=0)
        cnt = cur - start
    else:
        start = tl.sum(tl.where(offs_e < e, glv, 0), axis=0)
        cnt = cur
    start = start.to(tl.int32)
    cnt = cnt.to(tl.int32)

    cols = nblk * BN + tl.arange(0, BN)
    col_mask = cols < N
    offs_k = tl.arange(0, BK)
    ws_row = tl.load(ws_ptr + e * N + cols, mask=col_mask, other=0.0)

    nmb = (cnt + BM - 1) // BM
    for mb in range(0, nmb):
        row_offs = mb * BM + tl.arange(0, BM)
        row_mask = row_offs < cnt
        rows = start + row_offs
        a_ptrs = x_ptr + rows[:, None] * K + offs_k[None, :]
        b_ptrs = w_ptr + e * (K * N) + offs_k[:, None] * N + cols[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for _k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=row_mask[:, None], other=0)
            b = tl.load(b_ptrs, mask=col_mask[None, :], other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_ptrs += BK
            b_ptrs += BK * N
        xs_row = tl.load(xs_ptr + rows, mask=row_mask, other=0.0)
        val = (acc.to(tl.float32) * xs_row[:, None]) * ws_row[None, :]
        tl.store(hid_ptr + rows[:, None] * N + cols[None, :], val,
                 mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _swiglu_rowmax_kernel(hid_ptr, maxp_ptr,
                          N: tl.constexpr, HALF: tl.constexpr,
                          BH: tl.constexpr):
    """Per-row absmax of swiglu(hidden), one program per row."""
    r = tl.program_id(0)
    offs = tl.arange(0, BH)
    cmask = offs < HALF
    l = tl.load(hid_ptr + r * N + offs, mask=cmask, other=0.0)
    rt = tl.load(hid_ptr + r * N + HALF + offs, mask=cmask, other=0.0)
    v = (l / (1.0 + tl.exp(-l))) * rt
    tl.store(maxp_ptr + r, tl.max(tl.abs(v)))


@triton.jit
def _group_scale_kernel(maxp_ptr, scale_ptr, gl_ptr,
                        GLT: tl.constexpr, E: tl.constexpr, EBLK: tl.constexpr,
                        MBLK: tl.constexpr):
    """Per-group absmax -> common group scale filled into out_scale rows."""
    e = tl.program_id(0)
    offs_e = tl.arange(0, EBLK)
    glv = tl.load(gl_ptr + offs_e, mask=offs_e < E, other=0)
    cur = tl.sum(tl.where(offs_e == e, glv, 0), axis=0)
    if GLT == 0:
        start = tl.sum(tl.where(offs_e == e - 1, glv, 0), axis=0)
        cnt = cur - start
    else:
        start = tl.sum(tl.where(offs_e < e, glv, 0), axis=0)
        cnt = cur
    start = start.to(tl.int32)
    cnt = cnt.to(tl.int32)
    offs = tl.arange(0, MBLK)
    cmask = offs < cnt
    gmax = tl.max(tl.load(maxp_ptr + start + offs, mask=cmask, other=0.0))
    gscale = gmax / 127.0
    fill = gscale * tl.full((MBLK,), 1.0, tl.float32)
    tl.store(scale_ptr + start + offs, fill, mask=cmask)


@triton.jit
def _swiglu_quant_kernel(hid_ptr, scale_ptr, out_ptr,
                         N: tl.constexpr, HALF: tl.constexpr,
                         BH: tl.constexpr, QMODE: tl.constexpr):
    """SwiGLU activation + quantize, one program per row.
    QMODE 0 = per-token dynamic scale, QMODE 1 = per-group scale (row of scale_ptr)."""
    r = tl.program_id(0)
    offs = tl.arange(0, BH)
    cmask = offs < HALF
    l = tl.load(hid_ptr + r * N + offs, mask=cmask, other=0.0)
    rt = tl.load(hid_ptr + r * N + HALF + offs, mask=cmask, other=0.0)
    v = (l / (1.0 + tl.exp(-l))) * rt
    if QMODE == 0:
        mrow = tl.max(tl.abs(v))
        scv = (tl.maximum(mrow, 1e-10) / 127.0) + tl.arange(0, 1) * 0.0
        tl.store(scale_ptr + r, tl.max(scv))
    else:
        scv = tl.load(scale_ptr + r + tl.arange(0, 1))
    qf = v / scv
    q = tl.where(qf >= 0.0, qf + 0.5, qf - 0.5).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -128), 127)
    tl.store(out_ptr + r * HALF + offs, q.to(tl.int8), mask=cmask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, weight: list, weight_scale: list,
                x_scale: torch.Tensor, group_list: torch.Tensor,
                smooth_scale=None, weight_assist_matrix=None, bias=None,
                dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0,
                group_list_type=0, tuning_config=None):
        device = x.device
        x = x.to(device).contiguous()
        w3 = weight[0].to(device).contiguous()
        ws3 = weight_scale[0].to(device).contiguous()
        x_scale = x_scale.to(device).contiguous()
        gl = group_list.to(device)

        m, k = x.shape
        e, _kw, n = w3.shape
        half = n // 2

        hidden = torch.empty((m, n), dtype=torch.float32, device=device)
        rowmax = torch.empty((m,), dtype=torch.float32, device=device)
        out = torch.empty((m, half), dtype=torch.int8, device=device)
        out_scale = torch.empty((m,), dtype=torch.float32, device=device)

        if n >= 1024:
            bn = 256
        else:
            bn = 128
        nblk = (n + bn - 1) // bn
        t_est = (m + e - 1) // e
        if t_est <= 16:
            bm = 16
        elif t_est <= 32:
            bm = 32
        else:
            bm = 64
        eblk = triton.next_power_of_2(e)
        _grouped_gemm_kernel[(e * nblk,)](
            x, w3, x_scale, ws3, hidden, gl,
            GLT=group_list_type, E=e, EBLK=eblk, NBLK=nblk,
            K=k, N=n, BM=bm, BN=bn, BK=64)

        bh = triton.next_power_of_2(half)
        mblk = triton.next_power_of_2(m)
        if quant_mode == 1:
            _swiglu_rowmax_kernel[(m,)](hidden, rowmax,
                                        N=n, HALF=half, BH=bh)
            _group_scale_kernel[(e,)](rowmax, out_scale, gl,
                                      GLT=group_list_type, E=e, EBLK=eblk,
                                      MBLK=mblk)
        _swiglu_quant_kernel[(m,)](hidden, out_scale, out,
                                   N=n, HALF=half, BH=bh,
                                   QMODE=0 if quant_mode == 0 else 1)
        return out, out_scale