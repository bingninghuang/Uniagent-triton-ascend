import torch
import triton
import triton.language as tl


@triton.jit
def hstu_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_offsets_ptr, nt_ptr, nc_ptr,
    B,                     # batch count (int)
    alpha,                 # float scalar
    scaling,               # float scalar
    W,                     # max_attn_len (int)
    H: tl.constexpr, D: tl.constexpr, V: tl.constexpr,
    BMAX: tl.constexpr,    # next_power_of_2(B)
    BM: tl.constexpr, BN: tl.constexpr,
    CAUSAL: tl.constexpr,
    TG: tl.constexpr,      # target_group_size (>=1)
    SCORE_TY: tl.constexpr,   # dtype of scores/v for the 2nd tl.dot
):
    b = tl.program_id(0).to(tl.int32)
    h = tl.program_id(1).to(tl.int32)
    ib = tl.program_id(2).to(tl.int32)

    s = tl.load(seq_offsets_ptr + b).to(tl.int32)
    e = tl.load(seq_offsets_ptr + b + 1).to(tl.int32)
    n = e - s

    # NOTE: no early `return` allowed on Ascend (G constraints).
    # Inactive programs (ib*BM >= n) are handled by masks: in_seq all
    # False -> loads/store fully masked, and jhi_max == -1 -> empty loop.

    # fallback scaling: max sequence length in this batch
    bi = tl.arange(0, BMAX)
    bo_mask = bi < B
    off0 = tl.load(seq_offsets_ptr + bi, mask=bo_mask, other=0)
    off1 = tl.load(seq_offsets_ptr + bi + 1, mask=bo_mask, other=0)
    max_len = tl.max(tl.where(bo_mask, off1 - off0, 0))
    if scaling < 0:
        scaling = max_len.to(tl.float32)
    else:
        scaling = tl.full((), scaling, dtype=tl.float32)

    nt = tl.load(nt_ptr + b).to(tl.int32)
    nc = tl.load(nc_ptr + b).to(tl.int32)

    max_ids_post = n - nc + 1 - nt

    # ---- per-row valid-column upper bound (for loop bound only) ----
    i = ib * BM + tl.arange(0, BM)
    in_seq = i < n
    # causal: last valid col = i (rows i>=nc); ctx rows (i<nc): max(i, n-nt-1)
    if CAUSAL:
        jhi = i
        jhi = tl.where(i < nc, tl.maximum(i, n - nt - 1), jhi)
    else:
        jhi = n - 1
    jhi = tl.where(in_seq, jhi, -1)
    jhi_max = tl.max(jhi, axis=0)

    d = tl.arange(0, D)
    dv = tl.arange(0, V)
    qm = ((s + i)[:, None]) * (H * D) + h * D + d[None, :]
    q = tl.load(q_ptr + qm, mask=in_seq[:, None], other=0.0)

    ids_i = tl.maximum(i - nc + 1, 0)
    tgt_i = tl.where(i >= (n - nt), (i - (n - nt)) // TG, -1)

    acc = tl.zeros((BM, V), dtype=tl.float32)
    for jb in range(0, jhi_max + 1, BN):
        jv = jb + tl.arange(0, BN)
        j_in = jv < n
        ids_j = tl.maximum(jv - nc + 1, 0)

        kk = ((s + jv)[:, None]) * (H * D) + h * D + d[None, :]
        k = tl.load(k_ptr + kk, mask=j_in[:, None], other=0.0)
        s2 = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        # silu(qk * alpha) / scaling
        s2 = s2 * alpha
        s2 = s2 / (1.0 + tl.exp(-s2))
        s2 = s2 / scaling

        ids_i2 = ids_i[:, None]
        ids_j2 = ids_j[None, :]
        dist = ids_i2 - ids_j2
        if CAUSAL:
            base_ok = (i[:, None] == jv[None, :]) | (dist > 0)
        else:
            base_ok = tl.full((BM, BN), True, dtype=tl.int1)
        # target groups
        tgt_j = tl.where(jv >= (n - nt), (jv - (n - nt)) // TG, -1)
        tg_ok = (tgt_i[:, None] == tgt_j[None, :]) | (tgt_i[:, None] < 0) \
            | (tgt_j[None, :] < 0)
        if W > 0:
            win_d = dist if CAUSAL else tl.abs(dist)
            win_ok = win_d <= W
        else:
            win_ok = tl.full((BM, BN), True, dtype=tl.int1)
        ctx_ok = (ids_i2 == 0) & (ids_j2 < max_ids_post)
        valid = (base_ok & tg_ok & win_ok) | ctx_ok
        valid = valid & (jv[None, :] < n) & in_seq[:, None]
        s2 = tl.where(valid, s2, 0.0)

        vv = ((s + jv)[:, None]) * (H * D) + h * D + dv[None, :]
        v = tl.load(v_ptr + vv, mask=j_in[:, None], other=0.0)
        acc += tl.dot(s2.to(SCORE_TY), v, out_dtype=tl.float32)

    om = ((s + i)[:, None]) * (H * V) + h * V + dv[None, :]
    tl.store(out_ptr + om, acc.to(out_ptr.dtype.element_ty), mask=in_seq[:, None])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, seq_offsets, num_targets, num_contextuals,
                alpha, causal, max_attn_len, target_group_size,
                scaling_seqlen):
        L, H, D = q.shape
        V = v.shape[2]
        B = seq_offsets.shape[0] - 1
        causal = bool(causal)

        out = torch.empty(L, H, V, dtype=q.dtype, device=q.device)

        # host-side scheduling constants (index bookkeeping only)
        BM = 64
        BN = 64
        BMAX = triton.next_power_of_2(B)
        max_rb = (L + BM - 1) // BM

        grid = (B, H, max_rb)
        hstu_attn_kernel[grid](
            q, k, v, out,
            seq_offsets, num_targets, num_contextuals,
            B,
            float(alpha),
            float(scaling_seqlen),
            int(max_attn_len),
            H=H, D=D, V=V,
            BMAX=BMAX,
            BM=BM, BN=BN,
            CAUSAL=causal,
            TG=int(target_group_size),
            SCORE_TY=tl.bfloat16,
        )
        return out
