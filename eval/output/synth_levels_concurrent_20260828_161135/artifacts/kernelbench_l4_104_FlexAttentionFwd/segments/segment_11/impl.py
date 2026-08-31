import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def flex_attention_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    S_q, S_k,
    IS_CAUSAL: tl.constexpr,
    G: tl.constexpr,
    IS_F32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    if G == 1:
        h_kv = pid_h
    else:
        h_kv = pid_h // G

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S_q), other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    sm_scale = 1.0 / tl.sqrt(tl.full([1], BLOCK_D, dtype=tl.float32))

    if IS_CAUSAL:
        hi = tl.minimum(S_k, m_start + BLOCK_M + (S_k - S_q))
    else:
        hi = S_k

    for n_start in range(0, hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n[None, :] < S_k

        k_ptrs = (
            K
            + pid_b * stride_kb
            + h_kv * stride_kh
            + offs_d[:, None]
            + offs_n[None, :] * stride_kn
        )
        k = tl.load(k_ptrs, mask=n_mask, other=0.0)
        s = tl.dot(q, k)
        s = s * sm_scale

        if IS_CAUSAL:
            causal_ok = (offs_m[:, None] + (S_k - S_q)) >= offs_n[None, :]
            s = tl.where(causal_ok, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptrs = (
            V
            + pid_b * stride_vb
            + h_kv * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < S_k), other=0.0)

        if IS_F32:
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32))
        else:
            # hi/lo split of P keeps ~2x mantissa precision on the fast
            # 16-bit CUBE matmul path (fp32 dot falls back to AIV scalar)
            p_hi = p.to(V.dtype.element_ty)
            p_lo = (p - p_hi.to(tl.float32)).to(V.dtype.element_ty)
            t = tl.dot(p_hi, v)
            t = tl.dot(p_lo, v, t)
            acc = acc * alpha[:, None] + t
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = (
        O
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :]
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=(offs_m[:, None] < S_q))


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, query, key, value, is_causal=False, enable_gqa=False):
        B, H, S_q, D = query.shape
        H_kv = key.shape[1]
        S_k = key.shape[2]

        out = torch.empty_like(query)
        G = H // H_kv if (enable_gqa and H != H_kv) else 1

        num_m_blocks = (S_q + 63) // 64

        def grid(args):
            return (B, H, num_m_blocks)

        flex_attention_fwd_kernel[grid](
            query, key, value, out,
            query.stride(0), query.stride(1), query.stride(2),
            key.stride(0), key.stride(1), key.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            S_q, S_k,
            IS_CAUSAL=bool(is_causal),
            G=G,
            IS_F32=(query.dtype == torch.float32),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_D=D,
        )
        return out
