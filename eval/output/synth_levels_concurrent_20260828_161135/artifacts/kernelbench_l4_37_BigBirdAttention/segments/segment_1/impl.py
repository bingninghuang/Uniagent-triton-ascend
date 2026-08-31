import torch
import torch.nn as nn
import triton
import triton.language as tl


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pow2(x: int) -> int:
    r = 1
    while r < x:
        r <<= 1
    return r


@triton.jit
def _bigbird_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, mask_ptr, pse_ptr, sink_ptr, out_ptr,
    H, BH, SQ, SKV, NUM_DV_TILES,
    s_qb, s_qh, s_qm,
    s_kb, s_kh, s_kn,
    s_vb, s_vh, s_vn,
    s_mb, s_mn,
    s_p0, s_p1, s_p2, s_p3,
    scale,
    PSE_MODE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BDA: tl.constexpr,
    BDV: tl.constexpr,
):
    # pid layout: (dv_tile fastest, then bh, then m_tile)
    pid = tl.program_id(0)
    dv_t = pid % NUM_DV_TILES
    r = pid // NUM_DV_TILES
    bh = r % BH
    mt = r // BH
    b = bh // H
    h = bh % H

    offs_m = mt * BM + tl.arange(0, BM)
    offs_dv = dv_t * BDV + tl.arange(0, BDV)
    m_ok = offs_m < SQ
    dv_ok = offs_dv < DV

    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_kb + h * s_kh
    v_base = v_ptr + b * s_vb + h * s_vh

    neg_inf = float("-inf")
    m_i = tl.full((BM,), neg_inf, dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    acc = tl.zeros((BM, BDV), dtype=tl.float32)

    for n0 in range(0, SKV, BN):
        offs_n = n0 + tl.arange(0, BN)
        n_ok = offs_n < SKV

        # ---- scores block = Q[m] @ K[n]^T (fp32, exact in fp16 inputs) ----
        s = tl.zeros((BM, BN), dtype=tl.float32)
        for d0 in range(0, DK, BDA):
            offs_d = d0 + tl.arange(0, BDA)
            d_ok = offs_d < DK
            q_t = tl.load(
                q_base + offs_m[:, None] * s_qm + offs_d[None, :],
                mask=m_ok[:, None] & d_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            k_t = tl.load(
                k_base + offs_n[:, None] * s_kn + offs_d[None, :],
                mask=n_ok[:, None] & d_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.dot(q_t, tl.trans(k_t), s)

        s = s * scale

        # ---- sparse BigBird mask (1 = keep) ----
        mask_t = tl.load(
            mask_ptr + offs_m[:, None] * s_mb + offs_n[None, :] * s_mn,
            mask=m_ok[:, None] & n_ok[None, :],
            other=0.0,
        )
        s = tl.where(mask_t > 0.0, s, neg_inf)

        # ---- optional positional encoding bias ----
        if PSE_MODE == 4:
            pse_t = tl.load(
                pse_ptr + b * s_p0 + h * s_p1
                + offs_m[:, None] * s_p2 + offs_n[None, :] * s_p3,
                mask=m_ok[:, None] & n_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            s = s + pse_t
        elif PSE_MODE == 1:
            pse_t = tl.load(
                pse_ptr + offs_n, mask=n_ok, other=0.0
            ).to(tl.float32)
            s = s + pse_t[None, :]

        # ---- optional per-head sink bias ----
        if HAS_SINK:
            s = s + tl.load(sink_ptr + h)

        # ---- online softmax (guarded against all-masked blocks) ----
        row_max = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, row_max)
        mn_safe = tl.where(m_new == neg_inf, 0.0, m_new)
        mi_safe = tl.where(m_i == neg_inf, 0.0, m_i)
        p = tl.exp(tl.where(s == neg_inf, neg_inf, s) - mn_safe[:, None])
        alpha = tl.where(m_i == neg_inf, 0.0, tl.exp(mi_safe - mn_safe))
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # ---- weighted sum with V slice (fp32) ----
        v_t = tl.load(
            v_base + offs_n[:, None] * s_vn + offs_dv[None, :],
            mask=n_ok[:, None] & dv_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v_t)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]

    tl.store(
        out_ptr + b * s_qb + h * s_qh + offs_m[:, None] * s_qm + offs_dv[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=m_ok[:, None] & dv_ok[None, :],
    )


class ModelNew(nn.Module):
    """
    Triton Ascend implementation of BigBird hybrid sparse attention.

    The deterministic sparse mask (local window + global tokens + seeded
    random positions) depends only on (sq, skv, window_size,
    num_random_blocks), so it is materialized once per parameter set using
    the exact reference algorithm and cached.  The attention itself
    (scale, mask, pse, sink, softmax, PV) runs in a single fused
    flash-attention style Triton kernel with fp32 accumulation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._mask_cache = {}

    def _create_bigbird_mask(self, seq_len_q: int, seq_len_k: int,
                             window_size: int, num_random_blocks: int,
                             device) -> torch.Tensor:
        key = (seq_len_q, seq_len_k, window_size, num_random_blocks,
               str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        # Identical to the reference construction so the mask matches bit
        # for bit (same seeded torch.randperm sequence, same op order).
        mask = torch.zeros(seq_len_q, seq_len_k, device=device)

        # 1. Local window
        for i in range(seq_len_q):
            start = max(0, i - window_size // 2)
            end = min(seq_len_k, i + window_size // 2 + 1)
            mask[i, start:end] = 1

        # 2. Global tokens
        if seq_len_q > 0:
            mask[0, :] = 1
            mask[-1, :] = 1
        if seq_len_k > 0:
            mask[:, 0] = 1
            mask[:, -1] = 1

        # 3. Random attention for each non-global Q row
        if num_random_blocks > 0:
            seed = (seq_len_q + seq_len_k * 10007 + window_size * 131
                    + num_random_blocks * 17)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            for i in range(1, seq_len_q - 1):
                if seq_len_k <= num_random_blocks:
                    mask[i, :] = 1
                else:
                    rand_indices = torch.randperm(
                        seq_len_k, device=device,
                        generator=generator)[:num_random_blocks]
                    mask[i, rand_indices] = 1

        mask = mask.contiguous()
        self._mask_cache[key] = mask
        return mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                window_size: int, num_random_blocks: int,
                scale: float = None, pse: torch.Tensor = None,
                sink: torch.Tensor = None) -> torch.Tensor:
        b, n, sq, dk = query.shape
        skv = key.shape[2]
        dv = value.shape[3]

        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()

        # Same default logic as the reference (falsy scale -> 1/sqrt(dk)).
        scale = scale or (1.0 / (dk ** 0.5))

        mask = self._create_bigbird_mask(
            sq, skv, window_size, num_random_blocks, q.device)

        # Pos-encoding bias: support 4-d broadcast and flat-over-K forms.
        if pse is None:
            pse_mode = 0
            pse_t = q
            ps0 = ps1 = ps2 = ps3 = 0
        elif pse.dim() == 4:
            pse_mode = 4
            shp = pse.shape
            ps0 = 0 if shp[0] == 1 else pse.stride(0)
            ps1 = 0 if shp[1] == 1 else pse.stride(1)
            ps2 = 0 if shp[2] == 1 else pse.stride(2)
            ps3 = 0 if shp[3] == 1 else pse.stride(3)
            pse_t = pse
        else:
            pse_mode = 1
            pse_t = pse.reshape(-1)
            ps0 = ps1 = ps2 = ps3 = 0

        if sink is not None:
            has_sink = True
            sink_t = sink.contiguous()
        else:
            has_sink = False
            sink_t = mask

        out = torch.empty_like(q)

        DK = _pow2(dk)
        DV = _pow2(dv)
        BM = 16
        BN = 32
        BDA = min(128, DK)
        BDV = min(128, DV)
        num_m = _cdiv(sq, BM)
        num_dv = _cdiv(DV, BDV)
        grid = (b * n * num_m * num_dv,)

        _bigbird_attn_fwd_kernel[grid](
            q, k, v, mask, pse_t, sink_t, out,
            n, b * n, sq, skv, num_dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            mask.stride(0), mask.stride(1),
            ps0, ps1, ps2, ps3,
            float(scale),
            PSE_MODE=pse_mode,
            HAS_SINK=has_sink,
            DK=DK,
            DV=DV,
            BM=BM,
            BN=BN,
            BDA=BDA,
            BDV=BDV,
        )
        return out
