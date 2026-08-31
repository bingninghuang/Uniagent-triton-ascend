import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O,
    Q_LEN, K_LEN, N_HEADS, REPEATS,
    SQB, SQM, SQH,
    SKB, SKN, SKH,
    SVB, SVN, SVH,
    SOB, SOM, SOH,
    scale,
    softcap,
    window_left,
    window_right,
    D,
    USE_SOFTCAP: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_WL: tl.constexpr,
    HAS_WR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // N_HEADS
    pid_h = pid_bh % N_HEADS
    kv_h = pid_h // REPEATS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_limit = offs_m < Q_LEN
    d_limit = offs_d < D

    q = tl.load(
        Q + pid_b * SQB + offs_m[:, None] * SQM + pid_h * SQH + offs_d[None, :],
        mask=m_limit[:, None] & d_limit[None, :],
        other=0.0,
    )

    kv_base = pid_b * SKB + kv_h * SKH
    v_base = pid_b * SVB + kv_h * SVH
    offset_m = K_LEN - Q_LEN + offs_m

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    n_tiles = tl.cdiv(K_LEN, BLOCK_N)
    for i in range(0, n_tiles):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        n_limit = offs_n < K_LEN

        kT = tl.load(
            K + kv_base + offs_n[None, :] * SKN + offs_d[:, None],
            mask=n_limit[None, :] & d_limit[:, None],
            other=0.0,
        )
        qk = tl.trans(kT) * scale

        if USE_SOFTCAP:
            e = tl.exp((2.0 * qk) / softcap)
            qk = softcap * (e - 1.0) / (e + 1.0)

        if HAS_CAUSAL:
            qk = tl.where(offs_n[None, :] <= offset_m[:, None], qk, float("-inf"))
        if HAS_WL:
            qk = tl.where(offs_n[None, :] >= offset_m[:, None] - window_left, qk, float("-inf"))
        if HAS_WR:
            qk = tl.where(offs_n[None, :] <= offset_m[:, None] + window_right, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2((qk - safe_m[:, None]) * 2.0)
        alpha = tl.exp2((m_i - safe_m) * 2.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            V + v_base + offs_n[:, None] * SVN + offs_d[None, :],
            mask=n_limit[:, None] & d_limit[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_i[:, None]
    tl.store(
        O + pid_b * SOB + offs_m[:, None] * SOM + pid_h * SOH + offs_d[None, :],
        out.to(O.dtype.element_ty),
        mask=m_limit[:, None] & d_limit[None, :],
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, causal, window_left, window_right, softcap):
        B, Q_LEN, H, D = q.shape
        K_LEN = k.shape[1]
        repeats = H // k.shape[2]
        out = torch.empty_like(q)

        block_m = 32
        block_n = 32
        block_d = int(triton.next_power_of_2(D))
        scale = 1.0 / math.sqrt(D)

        grid = (triton.cdiv(Q_LEN, block_m), B * H)
        _flash_attn_fwd_kernel[grid](
            q, k, v, out,
            Q_LEN, K_LEN, H, repeats,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            scale,
            float(softcap),
            int(window_left),
            int(window_right),
            D,
            float(softcap) > 0.0,
            bool(causal),
            int(window_left) >= 0,
            int(window_right) >= 0,
            block_m,
            block_n,
            block_d,
        )
        return out