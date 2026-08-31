import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_pow2(x, minimum=16):
    n = minimum
    while n < x:
        n *= 2
    return n


def _tl_dtype(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


@triton.jit
def _paged_kv_gather_kernel(
    kv_flat_ptr,
    block_table_ptr,
    dense_ptr,
    capacity,
    num_pages,
    page_size,
    D,
    n_kv_tiles,
    BLOCK_N: tl.constexpr,
    D_CHUNK: tl.constexpr,
):
    """Materialize paged KV rows into a dense [batch, capacity, D] buffer."""
    pid = tl.program_id(0)
    b = pid // n_kv_tiles
    tile = pid % n_kv_tiles

    j = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = j < capacity
    tl.assume(D > 0)
    tl.assume(page_size > 0)

    page_idx = j // page_size
    in_page = j - page_idx * page_size
    block_id = tl.load(
        block_table_ptr + b * num_pages + tl.where(row_valid, page_idx, 0),
        mask=row_valid,
        other=0,
    )
    flat_row = block_id * page_size + in_page

    d_ar = tl.arange(0, D_CHUNK)
    src_row = flat_row[:, None] * D + d_ar[None, :]
    for d0 in range(0, D, D_CHUNK):
        d_ok = (d0 + d_ar) < D
        mask = row_valid[:, None] & d_ok[None, :]
        val = tl.load(kv_flat_ptr + d0 + src_row, mask=mask, other=0.0)
        dst = (b * capacity + j)[:, None] * D + d0 + d_ar[None, :]
        tl.store(dense_ptr + dst, val, mask=mask)


@triton.jit
def _paged_mqa_attention_kernel(
    q_ptr,
    kv_dense_ptr,
    cache_seqlens_ptr,
    out_ptr,
    H,
    qlen,
    capacity,
    D,
    DV,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DK: tl.constexpr,
    DV_PAD: tl.constexpr,
    CAUSAL: tl.constexpr,
    DOT_DT: tl.constexpr,
):
    """Flash-attention (MQA: single KV head) over dense gathered K/V.

    Q:   [B, qlen, H, D]  (contiguous)
    KV:  [B, capacity, D] rows 0:DV are V, full row is K
    OUT: [B, qlen, H, DV] (contiguous)
    """
    pid = tl.program_id(0)
    nqt = (qlen + BLOCK_M - 1) // BLOCK_M
    hb = pid // nqt
    qt = pid % nqt
    h = hb % H
    b = hb // H

    m0 = qt * BLOCK_M
    m_offs = m0 + tl.arange(0, BLOCK_M)
    row_valid = m_offs < qlen

    seqlen = tl.load(cache_seqlens_ptr + b)
    seqlen_min = tl.minimum(seqlen, capacity)

    q_base = b * qlen * H * D + h * D
    kv_base = b * capacity * D

    m_i = tl.full((BLOCK_M,), -1.0e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, DV_PAD), dtype=tl.float32)

    j_ar = tl.arange(0, BLOCK_N)
    d_ar = tl.arange(0, DK)
    dv_ar = tl.arange(0, DV_PAD)
    LOG2E_C: tl.constexpr = 1.4426950408889634
    NEG_INF: tl.constexpr = -float("inf")

    for n0 in range(0, seqlen_min, BLOCK_N):
        j = n0 + j_ar
        in_seqlen = j < seqlen_min

        if CAUSAL:
            last = (seqlen - qlen) + m_offs
            key_ok = in_seqlen[None, :] & (j[None, :] <= last[:, None])
        else:
            key_ok = in_seqlen[None, :]

        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in range(0, D, DK):
            d_ok = (d0 + d_ar) < D
            q_mask = row_valid[:, None] & d_ok[None, :]
            q_c = tl.load(
                q_ptr + q_base + m_offs[:, None] * (H * D) + d0 + d_ar[None, :],
                mask=q_mask,
                other=0.0,
            ).to(DOT_DT)
            k_mask = in_seqlen[:, None] & d_ok[None, :]
            k_c = tl.load(
                kv_dense_ptr + kv_base + j[:, None] * D + d0 + d_ar[None, :],
                mask=k_mask,
                other=0.0,
            ).to(DOT_DT)
            scores = tl.dot(q_c, tl.trans(k_c), scores)

        scores = scores * qk_scale
        scores = tl.where(key_ok, scores, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2((m_i - m_new) * LOG2E_C)
        p = tl.math.exp2((scores - m_new[:, None]) * LOG2E_C)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_mask = in_seqlen[:, None] & (dv_ar[None, :] < DV)
        v_c = tl.load(
            kv_dense_ptr + kv_base + j[:, None] * D + dv_ar[None, :],
            mask=v_mask,
            other=0.0,
        ).to(DOT_DT)
        acc = tl.dot(p.to(DOT_DT), v_c, acc * alpha[:, None])
        m_i = m_new

    o = acc / l_i[:, None]
    out_base = b * qlen * H * DV + h * DV
    out_mask = row_valid[:, None] & (dv_ar[None, :] < DV)
    tl.store(
        out_ptr + out_base + m_offs[:, None] * (H * DV) + dv_ar[None, :],
        o.to(DOT_DT),
        mask=out_mask,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, kv_cache, block_table, cache_seqlens, headdim_v, causal):
        B, qlen, H, D = q.shape
        _, page_size, _, cache_dim = kv_cache.shape
        num_pages = block_table.shape[1]
        capacity = num_pages * page_size
        DV = headdim_v
        dtype = q.dtype

        # Kernel 1: gather paged KV into a dense [B, capacity, D] buffer.
        dense = torch.empty((B, capacity, D), dtype=dtype, device=q.device)
        kv_flat = kv_cache.reshape(-1, cache_dim)
        BLOCK_N_G = 32
        D_CHUNK = 64
        n_kv_tiles = triton.cdiv(capacity, BLOCK_N_G)
        _paged_kv_gather_kernel[(B * n_kv_tiles,)](
            kv_flat,
            block_table,
            dense,
            capacity,
            num_pages,
            page_size,
            D,
            n_kv_tiles,
            BLOCK_N_G,
            D_CHUNK,
        )

        # Kernel 2: MQA flash attention.
        out = torch.empty((B, qlen, H, DV), dtype=dtype, device=q.device)
        BLOCK_M = 16
        DV_PAD = _next_pow2(DV, 32)
        BLOCK_N = 16 if DV_PAD >= 512 else 32
        nqt = triton.cdiv(qlen, BLOCK_M)
        qk_scale = 1.0 / math.sqrt(D)
        _paged_mqa_attention_kernel[(B * H * nqt,)](
            q,
            dense,
            cache_seqlens,
            out,
            H,
            qlen,
            capacity,
            D,
            DV,
            qk_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            DK=64,
            DV_PAD=DV_PAD,
            CAUSAL=int(causal),
            DOT_DT=_tl_dtype(dtype),
        )
        return out
