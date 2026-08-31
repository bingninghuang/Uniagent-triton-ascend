import torch
import triton
import triton.language as tl


@triton.jit
def hstu_flat_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,    # [L, H, *] row-major
    so_ptr,                          # [B+1] int64 seq offsets
    nt_ptr,                          # [B] int32 num_targets
    ncb_ptr,                         # [B] int32 num_contextuals
    alpha,                           # f32 score scale
    weff,                            # i32 window (INT_MAX when disabled)
    scaling,                         # i32 raw scaling (<0 -> max seqlen)
    B,                               # i32 batch count
    H: tl.constexpr, D: tl.constexpr, V: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
    BMAX: tl.constexpr,
    CAUSAL: tl.constexpr,
    TG: tl.constexpr,
    SCORE_TY: tl.constexpr,
):
    pid0 = tl.program_id(0)
    h = tl.program_id(1).to(tl.int32)

    # ---- map flat row-block id (pid0) -> (batch, local row start) ----
    # batch b owns blocks [st_b, st_b + cb_b) where
    # cb_b = cdiv(n_b, BM), st_b = sum_{b'<b} cb_b'
    bi = tl.arange(0, BMAX)
    bmask = bi < B
    o0 = tl.load(so_ptr + bi, mask=bmask, other=0)          # (BMAX,) i64
    o1 = tl.load(so_ptr + bi + 1, mask=bmask, other=0)
    nb = o1 - o0                                            # (BMAX,) i64
    ntv = tl.load(nt_ptr + bi, mask=bmask, other=0)         # (BMAX,) i32
    ncv = tl.load(ncb_ptr + bi, mask=bmask, other=0)        # (BMAX,) i32
    cb = tl.where(nb > 0, (nb + BM - 1) // BM, 0).to(tl.int32)
    lt = bi[None, :] < bi[:, None]
    st = tl.sum(tl.where(lt, cb[None, :], 0), axis=1)       # (BMAX,) i32
    m = (pid0 >= st) & (pid0 < st + cb) & bmask
    s0 = tl.sum(tl.where(m, o0, 0)).to(tl.int32)            # 0-d batch start
    n = tl.sum(tl.where(m, nb, 0)).to(tl.int32)             # 0-d batch len
    nt = tl.sum(tl.where(m, ntv, 0))                        # 0-d int32
    nc = tl.sum(tl.where(m, ncv, 0))                        # 0-d int32
    st_r = tl.sum(tl.where(m, st, 0))                       # 0-d int32
    r0 = (pid0.to(tl.int32) - st_r) * BM                    # 0-d local row start

    # ---- scaling (in-kernel: no host max) ----
    maxn = tl.max(tl.where(bmask, nb, 0))                   # 0-d i64
    se = tl.where(scaling > 0, scaling.to(tl.int64), maxn)
    inv_scalar = 1.0 / se.to(tl.float32)

    li = r0 + tl.arange(0, BM)                              # (BM,) i32
    row_ok = li < n
    d = tl.arange(0, D)
    dv = tl.arange(0, V)
    g64 = s0.to(tl.int64) + li.to(tl.int64)

    q_off = (g64[:, None] * H + h) * D + d[None, :]
    q_b = tl.load(q_ptr + q_off, mask=row_ok[:, None], other=0.0)

    # ---- LOCAL coords: reference mask semantics, exactly -----
    row_ids = tl.maximum(li - nc + 1, 0)                    # (BM,)
    mids = n - nc + 1                                       # 0-d
    hist = mids - nt                                        # 0-d
    tv = row_ids - mids + nt
    t_i = tl.where(tv < 0, -1, tv // TG)                    # (BM,)

    acc = tl.zeros((BM, V), dtype=tl.float32)

    for j0 in range(0, n, BN):
        lj = j0 + tl.arange(0, BN)
        j_ok = lj < n
        c64 = s0.to(tl.int64) + lj.to(tl.int64)

        col_ids = tl.maximum(lj - nc + 1, 0)                # (BN,)
        tv2 = col_ids - mids + nt
        t_j = tl.where(tv2 < 0, -1, tv2 // TG)              # (BN,)

        k_off = (c64[None, :] * H + h) * D + d[:, None]     # (D, BN)
        k_b = tl.load(k_ptr + k_off, mask=j_ok[None, :], other=0.0)

        s2 = tl.dot(q_b, k_b, out_dtype=tl.float32)         # (BM, BN)
        s2 = s2 * alpha
        s2 = s2 / (1.0 + tl.exp(-s2))                       # SiLU
        s2 = s2 * inv_scalar

        dist = row_ids[:, None] - col_ids[None, :]          # (BM, BN)
        diag = li[:, None] == lj[None, :]
        if CAUSAL:
            base_ok = diag | (dist > 0)
            win_ok = dist <= weff
        else:
            ad = tl.abs(dist)
            base_ok = diag | (ad > 0)
            win_ok = ad <= weff
        tg_ok = (t_i[:, None] == t_j[None, :]) | \
                (t_i[:, None] < 0) | (t_j[None, :] < 0)
        ctx_ok = (row_ids[:, None] == 0) & (col_ids[None, :] < hist)
        valid = (base_ok & tg_ok & win_ok) | ctx_ok

        v_off = (c64[:, None] * H + h) * V + dv[None, :]
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
        B = seq_offsets.numel() - 1
        if B < 1:
            B = 1

        seq_offsets = seq_offsets.to(q.device)
        num_targets = num_targets.to(q.device)
        num_contextuals = num_contextuals.to(q.device)

        out = torch.zeros(L, H, V, dtype=q.dtype, device=q.device)

        if q.dtype == torch.float16:
            stype = tl.float16
        elif q.dtype == torch.float32:
            stype = tl.float32
        else:
            stype = tl.bfloat16

        BM = 64
        BN = 32
        tg = int(target_group_size)
        if tg < 1:
            tg = 1
        weff = int(max_attn_len)
        if weff <= 0:
            weff = 2147483647
        grid = ((L + BM - 1) // BM + B, H)
        hstu_flat_kernel[grid](
            q, k, v, out,
            seq_offsets, num_targets, num_contextuals,
            float(alpha), weff, int(scaling_seqlen), B,
            H=H, D=D, V=V,
            BM=BM, BN=BN,
            BMAX=triton.next_power_of_2(B),
            CAUSAL=causal,
            TG=tg,
            SCORE_TY=stype,
        )
        return out
