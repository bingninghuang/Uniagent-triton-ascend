import torch
import torch.nn as nn
import triton
import triton.language as tl

# ascend910b1: 24 AI Cores (24 CUBE + 48 VEC)
CUBE_CORE_NUM = 24
LOG2E: tl.constexpr = tl.constexpr(1.4426950408889634)


@triton.jit
def sdpa_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    scale,
    B, num_heads, sq_len, sk_len, dk, dv,
    stride_qb, stride_qh,
    stride_kb, stride_kh,
    stride_vb, stride_vh,
    stride_ob, stride_oh,
    num_blocks_total,
    num_m_tiles,
    num_cores,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_P: tl.constexpr,
):
    pid = tl.program_id(0)
    in_dtype = q_ptr.dtype.element_ty
    off_m_idx = tl.arange(0, BLOCK_M)

    for block_idx in range(pid, num_blocks_total, num_cores):
        m_tile = block_idx % num_m_tiles
        tmp = block_idx // num_m_tiles
        h = tmp % num_heads
        b = tmp // num_heads

        off_m = m_tile * BLOCK_M + off_m_idx
        mask_m = off_m < sq_len

        q_base = q_ptr + b * stride_qb + h * stride_qh
        k_base = k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        o_base = o_ptr + b * stride_ob + h * stride_oh

        offs_dv = tl.arange(0, BLOCK_DV)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

        n_tiles = tl.cdiv(sk_len, BLOCK_N)
        for n_tile in range(0, n_tiles):
            off_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = off_n < sk_len

            scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for d0 in range(0, dk, BLOCK_DK):
                offs_d = d0 + tl.arange(0, BLOCK_DK)
                mask_d = offs_d < dk
                q_tile = tl.load(
                    q_base + off_m[:, None] * dk + offs_d[None, :],
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0,
                )
                k_tile = tl.load(
                    k_base + off_n[:, None] * dk + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0,
                )
                scores = tl.dot(q_tile, tl.trans(k_tile), scores)

            scores = scores * scale
            scores = tl.where(mask_n[None, :], scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp2((scores - m_new[:, None]) * LOG2E)
            p = tl.where(mask_n[None, :], p, 0.0)
            alpha = tl.exp2((m_i - m_new) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            p_hi = p.to(in_dtype)
            v_tile = tl.load(
                v_base + off_n[:, None] * dv + offs_dv[None, :],
                mask=mask_n[:, None] & (offs_dv < dv)[None, :], other=0.0,
            )
            acc = tl.dot(p_hi, v_tile, acc * alpha[:, None])
            if SPLIT_P:
                p_lo = (p - p_hi.to(tl.float32)).to(in_dtype)
                acc = tl.dot(p_lo, v_tile, acc)

        out = acc * (1.0 / l_i)[:, None]
        mask_o = mask_m[:, None] & (offs_dv < dv)[None, :]
        tl.store(
            o_base + off_m[:, None] * dv + offs_dv[None, :],
            out.to(in_dtype), mask=mask_o,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, scale):
        B, H, Sq, Dk = Q.shape
        Sk = K.shape[2]
        Dv = V.shape[3]
        dtype = Q.dtype

        Q = Q.contiguous() if not Q.is_contiguous() else Q
        K = K.contiguous() if not K.is_contiguous() else K
        V = V.contiguous() if not V.is_contiguous() else V
        out = torch.empty((B, H, Sq, Dv), device=Q.device, dtype=dtype)

        is_f32 = (dtype == torch.float32)
        BLOCK_M = 32 if is_f32 else 64
        BLOCK_N = 64
        BLOCK_DK = 64
        BLOCK_DV = triton.next_power_of_2(Dv)

        num_m_tiles = triton.cdiv(Sq, BLOCK_M)
        total_blocks = B * H * num_m_tiles

        grid = (CUBE_CORE_NUM,)
        sdpa_fwd_kernel[grid](
            Q, K, V, out,
            float(scale),
            B, H, Sq, Sk, Dk, Dv,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            V.stride(0), V.stride(1),
            out.stride(0), out.stride(1),
            total_blocks,
            num_m_tiles,
            CUBE_CORE_NUM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DK=BLOCK_DK,
            BLOCK_DV=BLOCK_DV,
            SPLIT_P=(not is_f32),
        )
        return out
