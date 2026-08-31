import torch
import triton
import triton.language as tl


@triton.jit
def hstu_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,  # global tensors [L,H,*]
    s0,                  # batch start offset (host int)
    n,                   # batch length (host int; loop bound)
    nt,                  # num_targets of this batch (host int)
    nc,                  # num_contextuals of this batch (host int)
    alpha,               # float scalar
    inv_scaling,         # float scalar = 1 / scaling
    weff,                # int: max_attn_len if > 0 else INT_MAX
    H: tl.constexpr, D: tl.constexpr, V: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
    CAUSAL: tl.constexpr,
    TG: tl.constexpr,
    SCORE_TY: tl.constexpr,
):
    row0 = tl.program_id(0).to(tl.int32) * BM
    h = tl.program_id(1).to(tl.int32)

    li = row0 + tl.arange(0, BM)               # local row ids in [0, n)
    row_ok = li < n

    d = tl.arange(0, D)
    dv = tl.arange(0, V)
    g64 = s0.to(tl.int64) + li.to(tl.int64)    # global row ids

    q_off = (g64[:, None] * H + h) * D + d[None, :]
    q_b = tl.load(q_ptr + q_off, mask=row_ok[:, None], other=0.0)

    # ---- LOCAL coords, matching the reference mask semantics exactly ----
    row_ids = tl.maximum(li - nc + 1, 0)       # (BM,)
    mids = n - nc + 1                          # scalar
    hist = mids - nt                           # scalar
    t_i = tl.where(row_ids - mids + nt < 0,
                   -1,
                   (row_ids - mids + nt) // TG)   # (BM,)

    acc = tl.zeros((BM, V), dtype=tl.float32)

    for j0 in range(0, n, BN):
        lj = j0 + tl.arange(0, BN)             # local col ids
        j_ok = lj < n
        c64 = s0.to(tl.int64) + lj.to(tl.int64)

        col_ids = tl.maximum(lj - nc + 1, 0)   # (BN,)
        t_j = tl.where(col_ids - mids + nt < 0,
                       -1,
                       (col_ids - mids + nt) // TG)   # (BN,)

        k_off = (c64[None, :] * H + h) * D + d[:, None]    # (D, BN)
        k_b = tl.load(k_ptr + k_off, mask=j_ok[None, :], other=0.0)

        s2 = tl.dot(q_b, k_b, out_dtype=tl.float32)        # (BM, BN)
        s2 = s2 * alpha
        s2 = s2 / (1.0 + tl.exp(-s2))                      # SiLU
        s2 = s2 * inv_scaling

        dist = row_ids[:, None] - col_ids[None, :]         # (BM, BN)
        if CAUSAL:
            base_ok = dist >= 0
            win_ok = dist <= weff
        else:
            ad = tl.abs(dist)
            base_ok = ad >= 0
            win_ok = ad <= weff
        tg_ok = (t_i[:, None] == t_j[None, :]) | \
                (t_i[:, None] < 0) | (t_j[None, :] < 0)
        ctx_ok = (row_ids[:, None] == 0) & (col_ids[None, :] < hist)
        valid = (base_ok & tg_ok & win_ok) | ctx_ok

        v_off = (c64[:, None] * H + h) * V + dv[None, :]   # (BN, V)
        v_b = tl.load(v_ptr + v_off, mask=j_ok[:, None], other=0.0)

        s2 = tl.where(valid & j_ok[None, :], s2, 0.0)
        acc = tl.dot(s2.to(SCORE_TY), v_b, acc, out_dtype=tl.float32)

    o_off = (g64[:, None] * H + h) * V + dv[None, :]
    tl.store(out_ptr + o_off, acc.to(out_ptr.dtype.element_ty),
             mask=row_ok[:, None])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, seq_offsets, num_targets, num_contextuals,
                alpha, causal, max_attn_len, target_group_size,
                scaling_seqlen):
        L, H, D = q.shape
        V = v.shape[2]
        causal = bool(causal)

        seq_offsets = seq_offsets.to(q.device)
        num_targets = num_targets.to(q.device)
        num_contextuals = num_contextuals.to(q.device)

        offs = seq_offsets.tolist()
        nt_list = num_targets.tolist()
        nc_list = num_contextuals.tolist()

        scaling = int(scaling_seqlen)
        if scaling < 0:
            scaling = max(offs[b + 1] - offs[b]
                          for b in range(len(offs) - 1))
        inv_scaling = 1.0 / max(scaling, 1)
        weff = int(max_attn_len) if int(max_attn_len) > 0 else 2147483647

        out = torch.zeros(L, H, V, dtype=q.dtype, device=q.device)

        if q.dtype == torch.float16:
            stype = tl.float16
        elif q.dtype == torch.float32:
            stype = tl.float32
        else:
            stype = tl.bfloat16

        BM = 64
        BN = 32
        ntg = max(int(target_group_size), 1)
        fa = float(alpha)

        for b in range(len(offs) - 1):
            s0 = offs[b]
            n = offs[b + 1] - s0
            if n <= 0:
                continue
            grid = ((n + BM - 1) // BM, H)
            hstu_attn_kernel[grid](
                q, k, v, out,
                s0, n, int(nt_list[b]), int(nc_list[b]),
                fa, float(inv_scaling), weff,
                H=H, D=D, V=V,
                BM=BM, BN=BN,
                CAUSAL=causal,
                TG=ntg,
                SCORE_TY=stype,
            )
        return out
