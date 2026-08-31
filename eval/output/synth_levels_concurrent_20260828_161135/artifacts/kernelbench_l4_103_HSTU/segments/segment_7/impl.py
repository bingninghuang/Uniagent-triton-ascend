import torch
import triton
import triton.language as tl


@triton.jit
def hstu_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_offsets_ptr, nt_ptr, nc_ptr,
    L,                      # global sequence length (host int; loop bound)
    alpha,                  # float scalar
    scaling,                # float scalar (<0 -> batch max seqlen)
    W,                      # max_attn_len (int)
    H: tl.constexpr, D: tl.constexpr, V: tl.constexpr,
    B: tl.constexpr,        # batch count
    BMAX: tl.constexpr,     # next_power_of_2(B)
    BM: tl.constexpr, BN: tl.constexpr,
    CAUSAL: tl.constexpr,
    TG: tl.constexpr,       # target_group_size (>= 1)
    SCORE_TY: tl.constexpr, # dtype of scores/v for the 2nd tl.dot
):
    row0 = tl.program_id(0).to(tl.int32) * BM
    h = tl.program_id(1).to(tl.int32)

    i = row0 + tl.arange(0, BM)                 # global row ids
    row_ok = i < L

    # ---- per-row batch lookup (in-kernel; no host device sync) ----
    bb = tl.arange(0, BMAX)
    b_ok = bb < B
    off_s = tl.load(seq_offsets_ptr + bb, mask=b_ok, other=0).to(tl.int32)
    off_e = tl.load(seq_offsets_ptr + bb + 1, mask=b_ok, other=0).to(tl.int32)
    nt_b = tl.load(nt_ptr + bb, mask=b_ok, other=0).to(tl.int32)
    nc_b = tl.load(nc_ptr + bb, mask=b_ok, other=0).to(tl.int32)

    inb = (i[:, None] >= off_s[None, :]) & (i[:, None] < off_e[None, :])
    inb = inb & b_ok[None, :]
    s_r = tl.sum(tl.where(inb, off_s[None, :], 0), axis=1).to(tl.int32)
    e_r = tl.sum(tl.where(inb, off_e[None, :], 0), axis=1).to(tl.int32)
    nt_r = tl.sum(tl.where(inb, nt_b[None, :], 0), axis=1).to(tl.int32)
    nc_r = tl.sum(tl.where(inb, nc_b[None, :], 0), axis=1).to(tl.int32)

    # scaling: scaling_seqlen < 0 -> max sequence length in the batch.
    # Vectorized 0-dim ops only: no runtime scf.if, no scalar rebind.
    max_len_f = tl.max(tl.where(b_ok, off_e - off_s, 0)).to(tl.float32)
    s0 = tl.full((), scaling, dtype=tl.float32)
    scaling_f = tl.where(s0 > 0.0, s0, max_len_f)
    inv_scaling = 1.0 / tl.maximum(scaling_f, 1.0)

    d = tl.arange(0, D)
    dv = tl.arange(0, V)
    i64 = i.to(tl.int64)

    # q stays bf16: tl.dot(bf16, bf16; fp32 acc) matches the golden qk.
    q_off = (i64[:, None] * H + h) * D + d[None, :]
    q_b = tl.load(q_ptr + q_off, mask=row_ok[:, None], other=0.0)

    row_ids = tl.maximum(i - nc_r + 1, 0)
    mids_r = (e_r - s_r) - nc_r + 1              # per-row (n - nc + 1)
    hist_r = mids_r - nt_r
    W_eff = tl.where(W > 0, W, 2147483647)

    acc = tl.zeros((BM, V), dtype=tl.float32)

    # KV columns over the GLOBAL index range; host-INT loop bound; per-row
    # mask restricts each row to its own batch span [s_r, e_r).
    for g0 in range(0, L, BN):
        j32 = g0 + tl.arange(0, BN)
        j64 = j32.to(tl.int64)
        j_ok = j32 < L
        jl = j32[None, :] - s_r[:, None]
        j_in = (jl >= 0) & (jl < (e_r - s_r)[:, None])
        j_in = j_in & row_ok[:, None] & j_ok[None, :]

        k_off = (j64[None, :] * H + h) * D + d[:, None]
        k_b = tl.load(k_ptr + k_off, mask=j_ok[None, :], other=0.0)

        s2 = tl.dot(q_b, k_b, out_dtype=tl.float32)
        s2 = s2 * alpha
        s2 = s2 / (1.0 + tl.exp(-s2))            # SiLU
        s2 = s2 * inv_scaling

        col_ids = tl.maximum(jl - nc_r[:, None] + 1, 0)
        dist = row_ids[:, None] - col_ids
        if CAUSAL:
            base_ok = dist >= 0
        else:
            base_ok = tl.abs(dist) >= 0

        t_i = row_ids - mids_r + nt_r
        t_i = tl.where(t_i < 0, -1, t_i // TG)
        t_j = col_ids - mids_r[:, None] + nt_r[:, None]
        t_j = tl.where(t_j < 0, -1, t_j // TG)
        tg_ok = (t_i[:, None] == t_j) | (t_i[:, None] < 0) | (t_j < 0)

        if CAUSAL:
            win_ok = dist <= W_eff
        else:
            win_ok = tl.abs(dist) <= W_eff

        ctx_ok = (row_ids[:, None] == 0) & (col_ids < hist_r[:, None])

        valid = ((base_ok & tg_ok & win_ok) | ctx_ok) & j_in
        s2 = tl.where(valid, s2, 0.0)

        v_off = (j64[:, None] * H + h) * V + dv[None, :]
        v_b = tl.load(v_ptr + v_off, mask=j_ok[:, None], other=0.0)
        acc += tl.dot(s2.to(SCORE_TY), v_b, out_dtype=tl.float32)

    o_off = (i64[:, None] * H + h) * V + dv[None, :]
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
        B = seq_offsets.shape[0] - 1
        causal = bool(causal)

        out = torch.empty(L, H, V, dtype=q.dtype, device=q.device)

        BM = 64
        BN = 64
        BMAX = triton.next_power_of_2(B)
        if q.dtype == torch.float16:
            stype = tl.float16
        elif q.dtype == torch.float32:
            stype = tl.float32
        else:
            stype = tl.bfloat16

        grid = ((L + BM - 1) // BM, H)
        hstu_attn_kernel[grid](
            q, k, v, out,
            seq_offsets, num_targets, num_contextuals,
            L,
            float(alpha),
            float(scaling_seqlen),
            int(max_attn_len),
            H=H, D=D, V=V,
            B=B, BMAX=BMAX,
            BM=BM, BN=BN,
            CAUSAL=causal,
            TG=int(target_group_size),
            SCORE_TY=stype,
        )
        return out
