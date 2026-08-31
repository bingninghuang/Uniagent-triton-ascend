import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_vec_core_num():
    try:
        import torch_npu

        return int(
            torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
        )
    except Exception:
        return 48


_VEC_CORE_NUM = _get_vec_core_num()


@triton.jit
def _flex_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    H,
    S_q,
    S_k,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_ob,
    stride_oh,
    stride_os,
    qk_scale,
    num_pids,
    GQA_REP: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NEED_Q_MASK: tl.constexpr,
    NEED_K_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    offs_m = tl.arange(0, BLOCK_M).to(tl.int32)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int32)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int32)

    num_m_blocks = tl.cdiv(S_q, BLOCK_M)
    n_blocks = B * H * num_m_blocks
    causal_shift = S_k - S_q

    for bid in range(pid, n_blocks, num_pids):
        m_blk = bid % num_m_blocks
        tmp = bid // num_m_blocks
        h = tmp % H
        b = tmp // H
        h_kv = h // GQA_REP

        m0 = m_blk * BLOCK_M
        q_rows = m0 + offs_m

        # ---- Load Q tile [BLOCK_M, BLOCK_D] (once per m block) ----
        q_ptrs = (
            q_ptr
            + b.to(tl.int32) * stride_qb
            + h.to(tl.int32) * stride_qh
            + q_rows[:, None] * stride_qs
            + offs_d[None, :]
        )
        if NEED_Q_MASK:
            q_mask = q_rows[:, None] < S_q
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        else:
            q = tl.load(q_ptrs)

        # K/V base for this (b, h_kv)
        k_base = k_ptr + b.to(tl.int32) * stride_kb + h_kv.to(tl.int32) * stride_kh
        v_base = v_ptr + b.to(tl.int32) * stride_vb + h_kv.to(tl.int32) * stride_vh

        # ---- Online softmax state ----
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        if IS_CAUSAL:
            hi = tl.minimum(S_k, m0 + BLOCK_M + causal_shift)
        else:
            hi = S_k
        n_iters = tl.cdiv(tl.maximum(hi, 1), BLOCK_N)

        for n_blk in range(n_iters):
            n0 = n_blk * BLOCK_N
            k_cols = n0 + offs_n

            # K^T tile [BLOCK_D, BLOCK_N]: element (d, j) at k_base + j*stride_ks + d
            kT_ptrs = k_base + k_cols[None, :] * stride_ks + offs_d[:, None]
            if NEED_K_MASK:
                k_mask = k_cols[None, :] < S_k
                kT = tl.load(kT_ptrs, mask=k_mask, other=0.0)
            else:
                kT = tl.load(kT_ptrs)

            qk = tl.dot(q, kT)  # [BLOCK_M, BLOCK_N] fp32
            qk = qk * qk_scale

            if NEED_K_MASK:
                qk = tl.where(k_mask, qk, -float("inf"))
            if IS_CAUSAL:
                causal_ok = k_cols[None, :] <= q_rows[:, None] + causal_shift
                qk = tl.where(causal_ok, qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(qk - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)

            v_ptrs = v_base + k_cols[:, None] * stride_vs + offs_d[None, :]
            if NEED_K_MASK:
                v = tl.load(v_ptrs, mask=(k_cols[:, None] < S_k), other=0.0)
            else:
                v = tl.load(v_ptrs)

            acc = acc * alpha[:, None] + tl.dot(
                p, v.to(tl.float32), out_dtype=tl.float32
            )
            m_i = m_ij

        acc = acc / l_i[:, None]
        o_ptrs = (
            o_ptr
            + b.to(tl.int32) * stride_ob
            + h.to(tl.int32) * stride_oh
            + q_rows[:, None] * stride_os
            + offs_d[None, :]
        )
        if NEED_Q_MASK:
            tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=q_mask)
        else:
            tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty))


class ModelNew(nn.Module):
    """FlexAttention forward (Flash-Attention style, no score materialization).

    O_h = softmax(scale * Q_h @ K_{h/g}^T  [causal masked]) @ V_{h/g}
    with GQA head mapping h_kv = h // (H // H_kv) when enable_gqa.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._cache = {}
        self._vec_core_num = _VEC_CORE_NUM

    def forward(self, query, key, value, is_causal=False, enable_gqa=False):
        B, H, S_q, D = query.shape
        H_kv = key.shape[1]
        S_k = key.shape[2]

        q_c = query.contiguous()
        k_c = key.contiguous()
        v_c = value.contiguous()
        out = torch.empty_like(q_c)

        if enable_gqa and H != H_kv:
            rep = H // H_kv
        else:
            rep = 1
        qk_scale = 1.0 / D ** 0.5

        if query.dtype == torch.float32:
            BLOCK_M = 32
            BLOCK_N = 32
        elif D <= 64:
            BLOCK_M = 64
            BLOCK_N = 64
        else:
            BLOCK_M = 32
            BLOCK_N = 64
        BLOCK_D = D

        num_m_blocks = (S_q + BLOCK_M - 1) // BLOCK_M
        total_blocks = B * H * num_m_blocks
        grid_size = (
            total_blocks if total_blocks < self._vec_core_num else self._vec_core_num
        )

        _flex_attn_fwd_kernel[(grid_size,)](
            q_c,
            k_c,
            v_c,
            out,
            B,
            H,
            S_q,
            S_k,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            k_c.stride(0),
            k_c.stride(1),
            k_c.stride(2),
            v_c.stride(0),
            v_c.stride(1),
            v_c.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            qk_scale,
            grid_size,
            GQA_REP=rep,
            IS_CAUSAL=is_causal,
            NEED_Q_MASK=(S_q % BLOCK_M != 0),
            NEED_K_MASK=(S_k % BLOCK_N != 0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        return out
