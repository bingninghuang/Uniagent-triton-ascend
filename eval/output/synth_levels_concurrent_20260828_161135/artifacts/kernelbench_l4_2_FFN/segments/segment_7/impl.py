import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------- main GEMM + activation kernel -----------------------------

@triton.jit
def ffn_gemmac_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, K, N, N1,
    a_row_stride, b_row_stride, out_row_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
    FP32_COMP: tl.constexpr,
):
    # ACT: 0=relu 1=silu 2=fastgelu 3=gelu(tanh) 4=identity
    # SPLIT: >0 means gate=act(acc at col n), up=acc2 at col n+half(=N); act per ACT kind
    pid = tl.program_id(0)
    np = tl.num_programs(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_m * num_n
    GROUP_M: tl.constexpr = 8
    num_pid_in_group = GROUP_M * num_n
    for blk in range(pid, num_blocks, np):
        group_id = blk // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (blk % num_pid_in_group) % group_size_m
        pid_n = (blk % num_pid_in_group) // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N
        a_ptrs = a_ptr + offs_m[:, None] * a_row_stride + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * b_row_stride + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k = 0
        for k_i in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k + offs_k) < K
            mk = m_mask[:, None] & k_mask[None, :]
            bk = k_mask[:, None] & n_mask[None, :]
            a = tl.load(a_ptrs, mask=mk, other=0.0)
            b = tl.load(b_ptrs, mask=bk, other=0.0)
            if FP32_COMP:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            acc = tl.dot(a, b, acc)
            if SPLIT:
                bu = tl.load(b_ptr + (k + offs_k)[:, None] * b_row_stride
                             + (N + offs_n)[None, :], mask=bk, other=0.0)
                if FP32_COMP:
                    bu = bu.to(tl.float32)
                acc2 = tl.dot(a, bu, acc2)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * b_row_stride
            k += BLOCK_K
        if HAS_BIAS:
            b1v = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc + b1v[None, :]
        if SPLIT:
            up = acc2
            if HAS_BIAS:
                u1v = tl.load(bias_ptr + N + offs_n, mask=n_mask, other=0.0).to(tl.float32)
                up = acc2 + u1v[None, :]
            if ACT == 0:
                act_g = tl.maximum(acc, 0.0)
            elif ACT == 1:
                act_g = acc * (1.0 / (1.0 + tl.exp(-acc)))
            elif ACT == 2:
                act_g = acc * (1.0 / (1.0 + tl.exp(-1.7 * acc)))
            elif ACT == 4:
                act_g = acc
            else:
                av = 0.7978845608028654 * acc * (1.0 + 0.044715 * acc * acc)
                ev = tl.exp(-2.0 * tl.abs(av))
                th = tl.where(av >= 0.0, 1.0, -1.0) * (1.0 - 2.0 / (1.0 + ev))
                act_g = 0.5 * acc * (1.0 + th)
            res = act_g * up
        else:
            if ACT == 0:
                res = tl.maximum(acc, 0.0)
            elif ACT == 1:
                res = acc * (1.0 / (1.0 + tl.exp(-acc)))
            elif ACT == 2:
                res = acc * (1.0 / (1.0 + tl.exp(-1.7 * acc)))
            elif ACT == 4:
                res = acc
            else:
                av = 0.7978845608028654 * acc * (1.0 + 0.044715 * acc * acc)
                ev = tl.exp(-2.0 * tl.abs(av))
                th = tl.where(av >= 0.0, 1.0, -1.0) * (1.0 - 2.0 / (1.0 + ev))
                res = 0.5 * acc * (1.0 + th)
        omask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptr + offs_m[:, None] * out_row_stride + offs_n[None, :],
                 res.to(out_ptr.dtype.element_ty), mask=omask)


# int8 quantized GEMM1 + dequant + act + requant
@triton.jit
def ffn_int8_gemmac_kernel(
    a_ptr, b_ptr, bias1_ptr, ds1_ptr, scale_ptr, offset_ptr, out_ptr,
    M, K, N,
    a_row_stride, b_row_stride, out_row_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACT: tl.constexpr, HAS_BIAS: tl.constexpr,
    DEQ_FP16: tl.constexpr, SCALE_PERCOL: tl.constexpr,
):
    pid = tl.program_id(0)
    np = tl.num_programs(0)
    num_n = tl.cdiv(N, BLOCK_N)
    num_blocks = tl.cdiv(M, BLOCK_M) * num_n
    deq_dt: tl.constexpr = tl.float16 if DEQ_FP16 else tl.bfloat16
    for blk in range(pid, num_blocks, np):
        pid_m = blk // num_n
        pid_n = blk % num_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N
        a_ptrs = a_ptr + offs_m[:, None] * a_row_stride + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * b_row_stride + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k = 0
        for k_i in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * b_row_stride
            k += BLOCK_K
        if HAS_BIAS:
            bv = tl.load(bias1_ptr + offs_n, mask=n_mask, other=0)
            acc = acc + bv.to(tl.int32)[None, :]
        # dequant: cast to deq dtype (rounds), float opmath multiply, round back
        dv = tl.load(ds1_ptr + offs_n, mask=n_mask, other=0.0)
        y1 = acc.to(deq_dt)
        y2 = (y1.to(tl.float32) * dv.to(tl.float32)[None, :]).to(deq_dt)
        # activation in fp32 (torch float opmath)
        x32 = y2.to(tl.float32)
        if ACT == 0:
            v = tl.maximum(x32, 0.0)
        elif ACT == 1:
            v = x32 * (1.0 / (1.0 + tl.exp(-x32)))
        elif ACT == 2:
            v = x32 * (1.0 / (1.0 + tl.exp(-1.7 * x32)))
        else:
            av = 0.7978845608028654 * x32 * (1.0 + 0.044715 * x32 * x32)
            ev = tl.exp(-2.0 * tl.abs(av))
            th = tl.where(av >= 0.0, 1.0, -1.0) * (1.0 - 2.0 / (1.0 + ev))
            v = 0.5 * x32 * (1.0 + th)
        v = v.to(deq_dt).to(tl.float32)  # act output rounded to deq dtype (torch semantics)
        ov = tl.load(offset_ptr).to(tl.float32)
        if SCALE_PERCOL:
            sv = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            q = v * sv[None, :] + ov
        else:
            sv = tl.load(scale_ptr).to(tl.float32)
            q = v * sv + ov
        q = tl.extra.ascend.libdevice.nearbyint(q).to(tl.int32)
        q = tl.maximum(q, -128)
        q = tl.minimum(q, 127)
        omask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptr + offs_m[:, None] * out_row_stride + offs_n[None, :],
                 q.to(out_ptr.dtype.element_ty), mask=omask)


# int8 quantized GEMM2 + dequant
@triton.jit
def ffn_int8_gemm2_kernel(
    a_ptr, b_ptr, bias2_ptr, ds2_ptr, out_ptr,
    M, K, N,
    a_row_stride, b_row_stride, out_row_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DEQ_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    np = tl.num_programs(0)
    num_n = tl.cdiv(N, BLOCK_N)
    num_blocks = tl.cdiv(M, BLOCK_M) * num_n
    deq_dt: tl.constexpr = tl.float16 if DEQ_FP16 else tl.bfloat16
    for blk in range(pid, num_blocks, np):
        pid_m = blk // num_n
        pid_n = blk % num_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N
        a_ptrs = a_ptr + offs_m[:, None] * a_row_stride + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * b_row_stride + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k = 0
        for k_i in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * b_row_stride
            k += BLOCK_K
        if HAS_BIAS:
            bv = tl.load(bias2_ptr + offs_n, mask=n_mask, other=0)
            acc = acc + bv.to(tl.int32)[None, :]
        dv = tl.load(ds2_ptr + offs_n, mask=n_mask, other=0.0)
        y1 = acc.to(deq_dt)
        y = y1.to(tl.float32) * dv.to(tl.float32)[None, :]
        omask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptr + offs_m[:, None] * out_row_stride + offs_n[None, :],
                 y.to(out_ptr.dtype.element_ty), mask=omask)


# Grouped MoE GEMM + activation (one launch covers all experts).
# a is [M, K]: rows are contiguous per-expert segments; et_ptr holds the
# per-expert token counts (int32, length E). b_ptr is [E, K, Nw] with row
# length Nw (= b_row_stride) and expert stride b_exp_stride; bias_ptr rows
# also have length b_row_stride; out_ptr is [M, N].
# Grid: (m_tiles, n_tiles) where m_tiles = sum_e cdiv(tokens_e, BLOCK_M);
# the tile -> (expert, row block) mapping is resolved by a small in-kernel
# scan. The host launches the upper bound M//BLOCK_M + E + 1 in dim0; any
# excess tile owns no rows and becomes a masked no-op.
@triton.jit
def ffn_moe_gemmac_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr, et_ptr,
    M, K, N, N1,
    a_row_stride, b_row_stride, b_exp_stride, out_row_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACT: tl.constexpr, SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
    FP32_COMP: tl.constexpr, E,
):
    # ACT: 0=relu 1=silu 2=fastgelu 3=gelu(tanh) 4=identity
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # scan the per-expert token counts to locate this m tile
    e_sel = 0
    row_sel = 0
    et_sel = 0
    t_off = 0
    r_off = 0
    for e in range(E):
        et_e = tl.load(et_ptr + e)
        t_e = tl.cdiv(et_e, BLOCK_M)
        hit = (t_off <= pid_m) & (pid_m < t_off + t_e)
        e_sel = tl.where(hit, e, e_sel)
        row_sel = tl.where(hit, r_off + (pid_m - t_off) * BLOCK_M, row_sel)
        et_sel = tl.where(hit, et_e, et_sel)
        t_off += t_e
        r_off += et_e
    local = tl.arange(0, BLOCK_M)
    offs_m = row_sel + local
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = local < et_sel
    n_mask = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * a_row_stride + offs_k[None, :]
    b_ptrs = b_ptr + e_sel * b_exp_stride + offs_k[:, None] * b_row_stride + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = 0
    for k_i in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k + offs_k) < K
        mk = m_mask[:, None] & k_mask[None, :]
        bk = k_mask[:, None] & n_mask[None, :]
        a = tl.load(a_ptrs, mask=mk, other=0.0)
        b = tl.load(b_ptrs, mask=bk, other=0.0)
        if FP32_COMP:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        acc = tl.dot(a, b, acc)
        if SPLIT:
            bu = tl.load(b_ptr + e_sel * b_exp_stride
                         + (k + offs_k)[:, None] * b_row_stride
                         + (N + offs_n)[None, :], mask=bk, other=0.0)
            if FP32_COMP:
                bu = bu.to(tl.float32)
            acc2 = tl.dot(a, bu, acc2)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * b_row_stride
        k += BLOCK_K
    if HAS_BIAS:
        b1v = tl.load(bias_ptr + e_sel * b_row_stride + offs_n,
                      mask=n_mask, other=0.0).to(tl.float32)
        acc = acc + b1v[None, :]
    if SPLIT:
        up = acc2
        if HAS_BIAS:
            u1v = tl.load(bias_ptr + e_sel * b_row_stride + N + offs_n,
                          mask=n_mask, other=0.0).to(tl.float32)
            up = acc2 + u1v[None, :]
        if ACT == 0:
            act_g = tl.maximum(acc, 0.0)
        elif ACT == 1:
            act_g = acc * (1.0 / (1.0 + tl.exp(-acc)))
        elif ACT == 2:
            act_g = acc * (1.0 / (1.0 + tl.exp(-1.7 * acc)))
        elif ACT == 4:
            act_g = acc
        else:
            av = 0.7978845608028654 * acc * (1.0 + 0.044715 * acc * acc)
            ev = tl.exp(-2.0 * tl.abs(av))
            th = tl.where(av >= 0.0, 1.0, -1.0) * (1.0 - 2.0 / (1.0 + ev))
            act_g = 0.5 * acc * (1.0 + th)
        res = act_g * up
    else:
        if ACT == 0:
            res = tl.maximum(acc, 0.0)
        elif ACT == 1:
            res = acc * (1.0 / (1.0 + tl.exp(-acc)))
        elif ACT == 2:
            res = acc * (1.0 / (1.0 + tl.exp(-1.7 * acc)))
        elif ACT == 4:
            res = acc
        else:
            av = 0.7978845608028654 * acc * (1.0 + 0.044715 * acc * acc)
            ev = tl.exp(-2.0 * tl.abs(av))
            th = tl.where(av >= 0.0, 1.0, -1.0) * (1.0 - 2.0 / (1.0 + ev))
            res = 0.5 * acc * (1.0 + th)
    omask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + offs_m[:, None] * out_row_stride + offs_n[None, :],
             res.to(out_ptr.dtype.element_ty), mask=omask)


# ----------------------------- host side -----------------------------

def _core_counts():
    cube, vec = 24, 48
    try:
        import torch_npu
        cfg = torch_npu.npu.npu_config.get_device_limit(0)
        cube = int(cfg.get("cube_core_num", 24))
        vec = int(cfg.get("vector_core_num", 48))
    except Exception:
        pass
    return cube, vec


def _act_code(activation: str, split: bool):
    if split:
        if activation == "geglu":
            return 3   # gelu(gate) * up
        if activation == "swiglu":
            return 1   # silu(gate) * up
        if activation == "reglu":
            return 0   # relu(gate) * up
        raise ValueError(activation)
    return {"relu": 0, "silu": 1, "fastgelu": 2, "gelu": 3}[activation]


def _pick_block(M, N):
    if M >= 128:
        bm = 128
    elif M >= 64:
        bm = 64
    else:
        bm = 16
    bn = 128
    bk = 64
    return bm, bn, bk


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cube_cores, self.vec_cores = _core_counts()

    def forward(self, x: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                activation: str, expert_tokens=None, expert_tokens_index=None,
                bias1=None, bias2=None, scale=None, offset=None,
                deq_scale1=None, deq_scale2=None,
                antiquant_scale1=None, antiquant_scale2=None,
                antiquant_offset1=None, antiquant_offset2=None,
                inner_precise=None, output_dtype=None):
        is_quant = scale is not None or deq_scale1 is not None
        is_pseudo_quant = (antiquant_scale1 is not None
                           or antiquant_scale2 is not None)
        is_moe = expert_tokens is not None

        if not x.is_contiguous():
            x = x.contiguous()
        if not weight1.is_contiguous():
            weight1 = weight1.contiguous()
        if not weight2.is_contiguous():
            weight2 = weight2.contiguous()

        if is_pseudo_quant:
            raise NotImplementedError("pseudo-quant FFN not exercised by test set")

        cores = self.cube_cores

        # ------------------------- MoE path -------------------------
        if is_moe:
            if is_quant:
                raise NotImplementedError("MoE + quant FFN not exercised by test set")
            x_orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            M, K = x_2d.shape
            K1, N1 = weight1.shape[-2], weight1.shape[-1]
            K2, N2 = weight2.shape[-2], weight2.shape[-1]
            split = activation in ("geglu", "swiglu", "reglu")
            act_n = (N1 // 2) if split else N1
            assert K1 == K and K2 == act_n
            if isinstance(expert_tokens, torch.Tensor):
                e_tokens = expert_tokens.tolist()
            elif hasattr(expert_tokens, "__len__"):
                e_tokens = list(expert_tokens)
            else:
                e_tokens = [int(expert_tokens)]
            E = len(e_tokens)
            et = torch.tensor(e_tokens, dtype=torch.int32, device=x.device)

            fp32_comp = (inner_precise == 0)
            act_code = _act_code(activation, split)
            inter_dtype = torch.float32
            out_dtype = output_dtype if output_dtype is not None else x.dtype
            inter = torch.empty((M, act_n), device=x.device, dtype=inter_dtype)
            out = torch.empty((M, N2), device=x.device, dtype=out_dtype)

            avg_t = (M + E - 1) // E
            bm, bn, bk = _pick_block(avg_t, act_n)
            m_tiles = M // bm + E + 1
            grid1 = (m_tiles, triton.cdiv(act_n, bn))
            b1_dummy = bias1 if bias1 is not None else inter
            ffn_moe_gemmac_kernel[grid1](
                x_2d, weight1, b1_dummy, inter, et,
                M, K, act_n, N1,
                K, N1, K1 * N1, act_n,
                bm, bn, bk, act_code, split, bias1 is not None, fp32_comp, E,
            )
            bm2, bn2, bk2 = _pick_block(avg_t, N2)
            m_tiles2 = M // bm2 + E + 1
            grid2 = (m_tiles2, triton.cdiv(N2, bn2))
            b2_dummy = bias2 if bias2 is not None else inter
            ffn_moe_gemmac_kernel[grid2](
                inter, weight2, b2_dummy, out, et,
                M, act_n, N2, N2,
                act_n, N2, K2 * N2, N2,
                bm2, bn2, bk2, 4, False, bias2 is not None, fp32_comp, E,
            )
            out_shape = list(x_orig_shape[:-1]) + [N2]
            return out.view(out_shape)

        # ------------------------ int8 quant path ------------------------
        if is_quant:
            x_orig_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            M, K = x_2d.shape
            N1 = weight1.shape[-1]
            N2 = weight2.shape[-1]

            deq_fp16 = (deq_scale1.dtype == torch.float16)
            act_code = {"relu": 0, "silu": 1, "fastgelu": 2, "gelu": 3}[activation]

            inter = torch.empty((M, N1), device=x.device, dtype=torch.int8)
            out_dtype = output_dtype if output_dtype is not None else torch.float16
            out = torch.empty((M, N2), device=x.device, dtype=out_dtype)

            scale_pp = (scale.reshape(-1) if scale is not None
                        else torch.ones(1, device=x.device))
            offset_pp = (offset.reshape(-1) if offset is not None
                         else torch.zeros(1, device=x.device))
            scale_percol = (scale_pp.numel() == N1)

            bm, bn, bk = _pick_block(M, N1)
            _ub1 = triton.cdiv(M, bm) * triton.cdiv(N1, bn)
            grid1 = (_ub1 if _ub1 < cores else cores,)
            b1_dummy = bias1 if bias1 is not None else inter
            ffn_int8_gemmac_kernel[grid1](
                x_2d, weight1, b1_dummy, deq_scale1, scale_pp, offset_pp, inter,
                M, K, N1,
                K, N1, N1,
                bm, bn, bk, act_code, bias1 is not None, deq_fp16, scale_percol,
            )
            bm2, bn2, bk2 = _pick_block(M, N2)
            _ub2 = triton.cdiv(M, bm2) * triton.cdiv(N2, bn2)
            grid2 = (_ub2 if _ub2 < cores else cores,)
            b2_dummy = bias2 if bias2 is not None else inter
            ffn_int8_gemm2_kernel[grid2](
                inter, weight2, b2_dummy, deq_scale2, out,
                M, N1, N2,
                N1, N2, N2,
                bm2, bn2, bk2, bias2 is not None, deq_fp16,
            )
            out_shape = list(x_orig_shape[:-1]) + [N2]
            return out.view(out_shape)

        # ------------------------ standard path ------------------------
        x_orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        M, K = x_2d.shape
        N1 = weight1.shape[-1]
        N2 = weight2.shape[-1]
        split = activation in ("geglu", "swiglu", "reglu")
        act_n = (N1 // 2) if split else N1
        act_code = _act_code(activation, split)

        fp32_comp = (inner_precise == 0)
        inter_dtype = torch.float32 if fp32_comp else x.dtype
        inter = torch.empty((M, act_n), device=x.device, dtype=inter_dtype)
        out_dtype = output_dtype if output_dtype is not None else x.dtype
        out = torch.empty((M, N2), device=x.device, dtype=out_dtype)

        bm, bn, bk = _pick_block(M, act_n)
        _ub1 = triton.cdiv(M, bm) * triton.cdiv(act_n, bn)
        grid1 = (_ub1 if _ub1 < cores else cores,)
        b1_dummy = bias1 if bias1 is not None else inter
        ffn_gemmac_kernel[grid1](
            x_2d, weight1, b1_dummy, inter, M, K, act_n, N1,
            K, N1, act_n,
            bm, bn, bk, act_code, split, bias1 is not None, fp32_comp,
        )
        bm2, bn2, bk2 = _pick_block(M, N2)
        _ub2 = triton.cdiv(M, bm2) * triton.cdiv(N2, bn2)
        grid2 = (_ub2 if _ub2 < cores else cores,)
        b2_dummy = bias2 if bias2 is not None else inter
        ffn_gemmac_kernel[grid2](
            inter, weight2, b2_dummy, out, M, act_n, N2, N2,
            act_n, N2, N2,
            bm2, bn2, bk2, 4, False, bias2 is not None, fp32_comp,
        )
        out_shape = list(x_orig_shape[:-1]) + [N2]
        return out.view(out_shape)