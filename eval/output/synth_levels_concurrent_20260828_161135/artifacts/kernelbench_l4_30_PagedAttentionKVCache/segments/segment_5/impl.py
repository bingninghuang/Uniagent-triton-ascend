import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_pow2(n):
    if n > 1:
        return 1 << (n - 1).bit_length()
    return 1


@triton.jit
def _paged_attention_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    seqlens_ptr,
    pt_ptr,
    out_ptr,
    num_kv_heads,
    repeats,
    s_qb,
    s_qm,
    s_qh,
    s_qd,
    s_kb,
    s_ks,
    s_kh,
    s_kd,
    s_vb,
    s_vs,
    s_vh,
    s_vd,
    s_ob,
    s_om,
    s_oh,
    s_od,
    s_pb,
    s_pc,
    head_dim,
    Q_len,
    capacity,
    page_size,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DT: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    H = num_kv_heads * repeats
    b = pid // H
    h = pid % H
    kv_head = h // repeats

    offs_m = tl.arange(0, BLOCK_M)
    offs_d0 = tl.arange(0, BLOCK_D)
    m_mask = offs_m < Q_len
    d0_mask = offs_d0 < head_dim

    q0 = tl.load(
        q_ptr
        + b * s_qb
        + h * s_qh
        + offs_m[:, None] * s_qm
        + offs_d0[None, :] * s_qd,
        mask=m_mask[:, None] & d0_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    HAVE_TAIL: tl.constexpr = BLOCK_DT > 0
    if HAVE_TAIL:
        offs_d1 = tl.arange(0, BLOCK_DT)
        d1_mask = (BLOCK_D + offs_d1) < head_dim
        q1 = tl.load(
            q_ptr
            + b * s_qb
            + h * s_qh
            + offs_m[:, None] * s_qm
            + (BLOCK_D + offs_d1)[None, :] * s_qd,
            mask=m_mask[:, None] & d1_mask[None, :],
            other=0.0,
        ).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))
    seqlen = tl.load(seqlens_ptr + b).to(tl.float32)
    end = tl.minimum(seqlen.to(tl.int32), capacity)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc0 = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    if HAVE_TAIL:
        acc1 = tl.zeros([BLOCK_M, BLOCK_DT], tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    pt_base = pt_ptr + b * s_pb
    last = tl.cast(seqlen - tl.cast(Q_len, tl.float32), tl.int32) + offs_m

    for start in range(0, tl.maximum(end, 1), BLOCK_K):
        offs_j = start + offs_k
        j_valid = offs_j < end
        page = offs_j // page_size
        in_page = offs_j - page * page_size
        block = tl.load(pt_base + page * s_pc, mask=j_valid, other=0)

        kv0 = (
            block[:, None] * s_kb
            + in_page[:, None] * s_ks
            + kv_head * s_kh
            + offs_d0[None, :] * s_kd
        )
        v0b = (
            block[:, None]
            * s_vb
            + in_page[:, None] * s_vs
            + kv_head * s_vh
            + offs_d0[None, :] * s_vd
        )
        k0 = tl.load(
            k_ptr + kv0,
            mask=j_valid[:, None] & d0_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v0 = tl.load(
            v_ptr + v0b,
            mask=j_valid[:, None] & d0_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        s = tl.dot(q0, tl.trans(k0)) * scale
        if HAVE_TAIL:
            kv1 = (
                block[:, None] * s_kb
                + in_page[:, None] * s_ks
                + kv_head * s_kh
                + (BLOCK_D + offs_d1)[None, :] * s_kd
            )
            v1b = (
                block[:, None]
                * s_vb
                + in_page[:, None] * s_vs
                + kv_head * s_vh
                + (BLOCK_D + offs_d1)[None, :] * s_vd
            )
            k1 = tl.load(
                k_ptr + kv1,
                mask=j_valid[:, None] & d1_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            v1 = tl.load(
                v_ptr + v1b,
                mask=j_valid[:, None] & d1_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            s += tl.dot(q1, tl.trans(k1)) * scale

        if CAUSAL:
            key_ok = offs_j[None, :] <= last[:, None]
        else:
            key_ok = j_valid[None, :]
        allowed = j_valid[None, :] & key_ok
        s = tl.where(allowed, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp(m_i - m_new)
        p = tl.math.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc0 = acc0 * alpha[:, None] + tl.dot(p, v0)
        if HAVE_TAIL:
            acc1 = acc1 * alpha[:, None] + tl.dot(p, v1)
        m_i = m_new

    l_i = tl.maximum(l_i, 1.0e-30)
    tl.store(
        out_ptr
        + b * s_ob
        + h * s_oh
        + offs_m[:, None] * s_om
        + offs_d0[None, :] * s_od,
        acc0 / l_i[:, None],
        mask=m_mask[:, None] & d0_mask[None, :],
    )
    if HAVE_TAIL:
        tl.store(
            out_ptr
            + b * s_ob
            + h * s_oh
            + offs_m[:, None] * s_om
            + (BLOCK_D + offs_d1)[None, :] * s_od,
            acc1 / l_i[:, None],
            mask=m_mask[:, None] & d1_mask[None, :],
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, cache_seqlens, page_table, causal):
        B, Q, H, D = q.shape
        num_pages, page_size, K, _ = k_cache.shape
        repeats = H // K
        capacity = page_table.shape[1] * page_size

        out = torch.empty_like(q)

        BLOCK_M = _next_pow2(Q) if Q > 16 else 16
        BLOCK_D = _next_pow2(D) if D > 16 else 16
        BLOCK_D = 128 if D > 128 else BLOCK_D
        tail = D - BLOCK_D
        BLOCK_DT = _next_pow2(tail) if tail > 1 else (16 if tail > 0 else 0)
        BLOCK_K = 32 if tail > 0 else 64

        grid = (B * H,)
        _paged_attention_fwd[grid](
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            page_table,
            out,
            K,
            repeats,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            page_table.stride(0),
            page_table.stride(1),
            D,
            Q,
            capacity,
            page_size,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            BLOCK_DT=BLOCK_DT,
            CAUSAL=bool(causal),
        )
        return out
