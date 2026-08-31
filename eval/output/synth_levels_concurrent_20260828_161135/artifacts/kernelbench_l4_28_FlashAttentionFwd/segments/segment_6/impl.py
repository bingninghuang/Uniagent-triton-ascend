import torch
import triton
import triton.language as tl


@triton.jit
def _fa_quantize_score(
    qk32,
    offs_n,
    offset_m,
    n_limit,
    sqrtD,
    softcap,
    window_left,
    window_right,
    HAS_CAUSAL: tl.constexpr,
    HAS_WL: tl.constexpr,
    HAS_WR: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    IN: tl.constexpr,
):
    # Emulate the PyTorch-on-NPU reference: every op materializes its
    # output in the input dtype, so round to IN after each sub-op.
    s1 = qk32.to(IN)  # matmul output rounding
    s2 = (s1.to(tl.float32) / sqrtD).to(IN)  # scores / sqrt(D)
    if USE_SOFTCAP:
        t1 = (s2.to(tl.float32) / softcap).to(IN)
        t2 = tl.tanh(t1.to(tl.float32)).to(IN)
        s3 = (t2.to(tl.float32) * softcap).to(IN)
    else:
        s3 = s2
    sf = s3.to(tl.float32)
    if HAS_CAUSAL:
        sf = tl.where(offs_n[None, :] <= offset_m[:, None], sf, float("-inf"))
    if HAS_WL:
        sf = tl.where(
            offs_n[None, :] >= offset_m[:, None] - window_left, sf, float("-inf")
        )
    if HAS_WR:
        sf = tl.where(
            offs_n[None, :] <= offset_m[:, None] + window_right, sf, float("-inf")
        )
    # Padded key columns (>= K_LEN) must be -inf: otherwise qk==0 there and
    # they leak into the softmax denominator whenever no right-bound mask
    # (causal / window_right) covers them.
    sf = tl.where(n_limit[None, :], sf, float("-inf"))
    return sf


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O,
    Q_LEN, K_LEN, N_HEADS, REPEATS,
    SQB, SQM, SQH,
    SKB, SKN, SKH,
    SVB, SVN, SVH,
    SOB, SOM, SOH,
    D,
    softcap,
    window_left,
    window_right,
    USE_SOFTCAP: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_WL: tl.constexpr,
    HAS_WR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    IN: tl.constexpr = Q.dtype.element_ty
    sqrtD = tl.sqrt(D.to(tl.float32))

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

    if HAS_CAUSAL:
        hi = tl.minimum(K_LEN, pid_m * BLOCK_M + BLOCK_M - 1 + K_LEN - Q_LEN)
        n_tiles = tl.cdiv(hi, BLOCK_N)
    else:
        n_tiles = tl.cdiv(K_LEN, BLOCK_N)

    # ---- pass A: exact row max M and softmax normalizer L ----
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    for i in range(0, n_tiles):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        n_limit = offs_n < K_LEN
        kT = tl.load(
            K + kv_base + offs_n[None, :] * SKN + offs_d[:, None],
            mask=n_limit[None, :] & d_limit[:, None],
            other=0.0,
        )
        sf = _fa_quantize_score(
            tl.dot(q, kT), offs_n, offset_m, n_limit,
            sqrtD, softcap, window_left, window_right,
            HAS_CAUSAL, HAS_WL, HAS_WR, USE_SOFTCAP, IN,
        )
        m_new = tl.maximum(m_i, tl.max(sf, axis=1))
        safe_m = tl.where(m_new == float("-inf"), 0.0, m_new)
        e = tl.exp(sf - safe_m[:, None])
        alpha = tl.exp(m_i - safe_m)
        l_i = l_i * alpha + tl.sum(e, axis=1)
        m_i = m_new

    safe_M = tl.where(m_i == float("-inf"), 0.0, m_i)
    L_safe = tl.where(l_i == 0.0, 1.0, l_i)

    # ---- pass B: weights rounded to IN (like reference softmax output),
    #              then PV matmul with fp32 accumulation ----
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for i in range(0, n_tiles):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        n_limit = offs_n < K_LEN
        kT = tl.load(
            K + kv_base + offs_n[None, :] * SKN + offs_d[:, None],
            mask=n_limit[None, :] & d_limit[:, None],
            other=0.0,
        )
        sf = _fa_quantize_score(
            tl.dot(q, kT), offs_n, offset_m, n_limit,
            sqrtD, softcap, window_left, window_right,
            HAS_CAUSAL, HAS_WL, HAS_WR, USE_SOFTCAP, IN,
        )
        p = tl.exp(sf - safe_M[:, None]) / L_safe[:, None]
        v = tl.load(
            V + v_base + offs_n[:, None] * SVN + offs_d[None, :],
            mask=n_limit[:, None] & d_limit[None, :],
            other=0.0,
        )
        acc = tl.dot(p.to(IN), v, acc)

    tl.store(
        O + pid_b * SOB + offs_m[:, None] * SOM + pid_h * SOH + offs_d[None, :],
        acc.to(IN),
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

        block_d = 1 << (D - 1).bit_length()
        block_m = 32 if block_d <= 128 else 16
        block_n = 32

        grid = (triton.cdiv(Q_LEN, block_m), B * H)
        _flash_attn_fwd_kernel[grid](
            q, k, v, out,
            Q_LEN, K_LEN, H, repeats,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            D,
            float(softcap),
            int(window_left),
            int(window_right),
            float(softcap) > 0.0,
            bool(causal),
            int(window_left) >= 0,
            int(window_right) >= 0,
            block_m,
            block_n,
            block_d,
        )
        return out