import torch
import triton
import triton.language as tl


def _next_pow2(n):
    return 1 << max(0, (n - 1).bit_length())


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
    CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    H = num_kv_heads * repeats
    b = pid // H
    h = pid % H
    kv_head = h // repeats

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < Q_len
    d_mask = offs_d < head_dim

    q = tl.load(
        q_ptr
        + b * s_qb
        + h * s_qh
        + offs_m[:, None] * s_qm
        + offs_d[None, :] * s_qd,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    scale = 1.0 / tl.sqrt(tl.cast(head_dim, tl.float32))
    seqlen = tl.load(seqlens_ptr + b).to(tl.float32)
    end = tl.minimum(seqlen.to(tl.int32), capacity)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    pt_base = pt_ptr + b * s_pb
    last = tl.cast(seqlen - tl.cast(Q_len, tl.float32), tl.int32) + offs_m

    for start in range(0, tl.maximum(end, 1), BLOCK_K):
        offs_j = start + offs_k
        j_valid = offs_j < end
        page = offs_j // page_size
        in_page = offs_j - page * page_size
        block = tl.load(pt_base + page * s_pc, mask=j_valid, other=0)

        kv_off = (
            block[:, None] * s_kb
            + in_page[:, None] * s_ks
            + kv_head * s_kh
            + offs_d[None, :] * s_kd
        )
        k = tl.load(
            k_ptr + kv_off,
            mask=j_valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr
            + block[:, None]
            * s_vb
            + in_page[:, None] * s_vs
            + kv_head * s_vh
            + offs_d[None, :] * s_vd,
            mask=j_valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        s = tl.dot(q, tl.trans(k)) * scale
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
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    l_i = tl.maximum(l_i, 1.0e-30)
    out = acc / l_i[:, None]
    tl.store(
        out_ptr
        + b * s_ob
        + h * s_oh
        + offs_m[:, None] * s_om
        + offs_d[None, :] * s_od,
        out,
        mask=m_mask[:, None] & d_mask[None, :],
    )


class ModelNew:
    def __init__(self):
        pass

    def forward(self, q, k_cache, v_cache, cache_seqlens, page_table, causal):
        B, Q, H, D = q.shape
        num_pages, page_size, K, _ = k_cache.shape
        repeats = H // K
        capacity = page_table.shape[1] * page_size

        out = torch.empty_like(q)

        BLOCK_M = max(16, _next_pow2(Q))
        BLOCK_D = max(16, _next_pow2(D))
        BLOCK_K = 64

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
            CAUSAL=bool(causal),
        )
        return out
