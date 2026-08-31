import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

def _get_vec_core_num():
    try:
        import torch_npu
        lim = torch_npu.npu.npu_config.get_device_limit(0)
        n = int(lim.get("vector_core_num", 0) or 0)
        if n > 0:
            return n
    except Exception:
        pass
    return 48


LOG2E: tl.constexpr = 1.4426950408889634


# ---------------------------------------------------------------------------
# K1: per (b,h) sum of k over L  ->  [BH, P, DP] fp32 partials
# ---------------------------------------------------------------------------
@triton.jit
def sla_kmean_kernel(
    k_ptr, part_ptr,
    L, DP,
    s_bh, s_p, s_d,
    num_units, num_pids,
    BLK: tl.constexpr, DPX: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    bhp = num_units // num_pids
    rem = num_units - bhp * num_pids
    if pid < rem:
        ub_ = bhp + 1
        first = pid * (bhp + 1)
    else:
        ub_ = bhp
        first = rem * (bhp + 1) + (pid - rem) * bhp
    last = first + ub_
    for unit in range(first, last):
        bh = unit // P
        p = unit - bh * P
        ch = tl.cdiv(L, P)
        l0 = p * ch
        l1 = tl.minimum(l0 + ch, L)
        acc = tl.zeros([DPX], dtype=tl.float32)
        d_offs = tl.arange(0, DPX)
        l_offs = l0 + tl.arange(0, BLK)
        for l0c in range(l0, l1, BLK):
            msk = (l0c + tl.arange(0, BLK)) < l1
            ktile = tl.load(k_ptr + bh * s_bh + l_offs[:, None] * DP + d_offs[None, :],
                            mask=msk[:, None] & (d_offs[None, :] < DP), other=0.0)
            l_offs += BLK
            acc += tl.sum(ktile.to(tl.float32), axis=0)
        tl.store(part_ptr + unit * s_p + d_offs * s_d, acc, mask=d_offs < DP)


# ---------------------------------------------------------------------------
# K2: block means (q with BLKQ, or arg_k with BLKK) -> dtype
# ---------------------------------------------------------------------------
@triton.jit
def sla_blockmean_kernel(
    x_ptr, km_ptr, out_ptr,
    L, BQ,
    s_bh, s_l, s_b, s_bq, s_n, s_d,
    num_units, num_pids,
    IS_ARGK: tl.constexpr, DP: tl.constexpr, DPX: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    bpu = num_units // num_pids
    rem = num_units - bpu * bpu
    if pid < rem:
        ub_ = bpu + 1
        first = pid * (bpu + 1)
    else:
        ub_ = bpu
        first = rem * (bpu + 1) + (pid - rem) * bpu
    last = first + ub_
    d_offs = tl.arange(0, DPX)
    d_msk = d_offs < DP
    for unit in range(first, last):
        bh = unit // BQ
        nb = unit - bh * BQ
        base = bh * s_bh + nb * s_l
        acc = tl.zeros([DPX], dtype=tl.float32)
        for i in range(0, BQ, 32):
            off = nb * BQ + i + tl.arange(0, 32)
            msk = (off < L) & d_msk
            if IS_ARGK:
                krow = tl.load(km_ptr + bh * s_l + d_offs, mask=d_msk, other=0.0).to(tl.float32)
                xt = tl.load(x_ptr + base + off[:, None] * s_b + d_offs[None, :],
                             mask=msk, other=0.0)
                diff = (xt - krow[None, :].to(xt.dtype)).to(tl.float32)
                diff = tl.where(off[:, None] < L, diff, 0.0)
            else:
                diff = tl.load(x_ptr + base + off[:, None] * s_b + d_offs[None, :],
                               mask=msk, other=0.0).to(tl.float32)
                diff = tl.where(off[:, None] < L, diff, 0.0)
            acc += tl.sum(diff, axis=0)
        cnt = tl.minimum(BQ, L - nb * BQ).to(tl.float32)
        bmean = (acc / cnt).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + bh * s_bq + nb * s_n + d_offs, bmean, mask=d_msk)


# ---------------------------------------------------------------------------
# K3: pooled = qm_f @ km_f^T stored in dtype  [B, H, MQ, NK]
# ---------------------------------------------------------------------------
@triton.jit
def sla_pooled_kernel(
    qm_ptr, km_ptr, pooled_ptr,
    MQ, NK,
    s_bh, s_m, s_n, s_k, s_nd,
    num_units, num_pids,
    DP: tl.constexpr, DPX: tl.constexpr,
    BMQ: tl.constexpr, BNK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    bpu = num_units // num_pids
    rem = num_units - bpu * bpu
    if pid < rem:
        ub_ = bpu + 1
        first = pid * (bpu + 1)
    else:
        ub_ = bpu
        first = rem * (bpu + 1) + (pid - rem) * bpu
    last = first + ub_
    d_offs = tl.arange(0, DPX)
    m_offs = tl.arange(0, BMQ)
    n_offs = tl.arange(0, BNK)
    for unit in range(first, last):
        bh = unit // (MQ // BMQ)
        tile = unit - bh * (MQ // BMQ)
        tm = tile // (NK // BNK)
        tn = tile - tm * (NK // BNK)
        qbase = bh * s_bh + (tm * BMQ) * s_m
        kbase = bh * s_k + (tn * BNK) * s_nd
        qf = tl.load(qm_ptr + qbase + m_offs[:, None] * s_n + d_offs[None, :],
                     mask=(m_offs[:, None] < MQ) & (d_offs[None, :] < DP), other=0.0).to(tl.float32)
        kf = tl.load(km_ptr + kbase + n_offs[:, None] * s_nd + d_offs[None, :],
                     mask=(n_offs[:, None] < NK) & (d_offs[None, :] < DP), other=0.0).to(tl.float32)
        acc = tl.dot(qf, tl.trans(kf), out_dtype=tl.float32)
        accd = acc.to(pooled_ptr.dtype.element_ty)
        pb = bh * s_bh + (tm * BMQ) * s_n
        tl.store(pooled_ptr + pb + m_offs[:, None] * s_nd + (tn * BNK + n_offs)[None, :],
                 accd, mask=(m_offs[:, None] < MQ) & (n_offs[None, :] < NK))


# ---------------------------------------------------------------------------
# K4: iterative top-k per row of pooled -> lut int32 [B, H, MQ, K]
# ---------------------------------------------------------------------------
@triton.jit
def sla_topk_kernel(
    pooled_ptr, lut_ptr,
    NK, K,
    s_bh, s_m, s_s, s_nk,
    num_units, num_pids,
    KPOW2: tl.constexpr, BNK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    bpu = num_units // num_pids
    rem = num_units - bpu * bpu
    if pid < rem:
        ub_ = bpu + 1
        first = pid * (bpu + 1)
    else:
        ub_ = bpu
        first = rem * (bpu + 1) + (pid - rem) * bpu
    last = first + ub_
    n_offs = tl.arange(0, BNK)
    s_offs = tl.arange(0, KPOW2)
    for unit in range(first, last):
        bh = unit // NK
        m = unit - bh * NK
        base = bh * s_bh + m * s_s
        cand = tl.load(pooled_ptr + base + n_offs * s_nk,
                       mask=n_offs < NK, other=float("-inf")).to(tl.float32)
        cand = tl.where(n_offs < NK, cand, float("-inf"))
        idx_vec = tl.zeros([KPOW2], dtype=tl.int32)
        for i in range(0, KPOW2):
            pick = i < K
            mv = tl.max(tl.where(pick, cand, float("-inf")), axis=0)
            bi = tl.argmax(tl.where(pick, cand, float("-inf")), axis=0)
            idx_vec = tl.where(s_offs == i, bi.to(tl.int32), idx_vec)
            cand = tl.where(pick & (n_offs == bi), float("-inf"), cand)
        tl.store(lut_ptr + base + s_offs, idx_vec, mask=s_offs < K)


# ---------------------------------------------------------------------------
# K5: feature map over whole seq (softmax / elu / relu) -> fp32
# ---------------------------------------------------------------------------
@triton.jit
def sla_fmap_kernel(
    x_ptr, y_ptr,
    L,
    s_bh, s_l,
    num_units, num_pids,
    FM: tl.constexpr, DP: tl.constexpr, DPX: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    bpu = num_units // num_pids
    rem = num_units - bpu * bpu
    if pid < rem:
        ub_ = bpu + 1
        first = pid * (bpu + 1)
    else:
        ub_ = bpu
        first = rem * (bpu + 1) + (pid - rem) * bpu
    last = first + ub_
    d_offs = tl.arange(0, DPX)
    d_msk = d_offs < DP
    for unit in range(first, last):
        bh = unit // L
        l0 = unit - bh * L
        for l0c in range(l0, l0 + L, BLK):
            lo = l0c + tl.arange(0, BLK)
            msk = (lo < l0 + L) & d_msk
            x = tl.load(x_ptr + bh * s_bh + lo[:, None] * s_l + d_offs[None, :],
                        mask=msk, other=0.0).to(tl.float32)
            if FM == 0:
                mx = tl.max(x, axis=1)[:, None]
                e = tl.exp(x - mx)
                e = tl.where(lo[:, None] < l0 + L, e, 0.0)
                s = tl.sum(e, axis=1)[:, None]
                y = e / s
            elif FM == 1:
                e = tl.exp(x)
                y = tl.where(x > 0, x, e - 1.0) + 1.0
            else:
                y = tl.where(x > 0, x, 0.0)
            tl.store(y_ptr + bh * s_bh + lo[:, None] * s_l + d_offs[None, :],
                     y, mask=msk)


# ---------------------------------------------------------------------------
# K6: partials  kvsum_p[bh,p,i,j] = sum_l c_k[l,i]*v[l,j], ksum_p[bh,p,i]
# ---------------------------------------------------------------------------
@triton.jit
def sla_partials_kernel(
    ck_ptr, v_ptr, kvsum_ptr, ksum_ptr,
    L, D,
    s_ck, s_v, s_p, s_i, s_j, s_k, s_ks,
    num_units, num_pids,
    DP: tl.constexpr, DPX: tl.constexpr,
    BLK: tl.constexpr, DTILE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    bpu = num_units // num_pids
    rem = num_units - bpu * bpu
    if pid < rem:
        ub_ = bpu + 1
        first = pid * (bpu + 1)
    else:
        ub_ = bpu
        first = rem * (bpu + 1) + (pid - rem) * bpu
    last = first + ub_
    for unit in range(first, last):
        bh = unit // P
        p = unit - bh * P
        ch = tl.cdiv(L, P)
        l0 = p * ch
        l1 = tl.minimum(l0 + ch, L)
        l_offs = l0 + tl.arange(0, BLK)
        for it in range(0, DP, DTILE):
            i_offs = it + tl.arange(0, DTILE)
            i_msk = i_offs < DP
            acc = tl.zeros([DTILE, DPX], dtype=tl.float32)
            ks = tl.zeros([DTILE], dtype=tl.float32)
            lo = l_offs
            for l0c in range(l0, l1, BLK):
                lmsk = lo < l1
                ck = tl.load(ck_ptr + bh * s_ck + lo[:, None] * s_i + i_offs[None, :],
                             mask=lmsk[:, None] & i_msk[None, :], other=0.0).to(tl.float32)
                vt = tl.load(v_ptr + bh * s_v + lo[:, None] * s_j + tl.arange(0, DPX)[None, :],
                             mask=lmsk[:, None] & (tl.arange(0, DPX)[None, :] < DP), other=0.0).to(tl.float32)
                acc += tl.dot(tl.trans(ck), vt, out_dtype=tl.float32)
                ks += tl.sum(ck, axis=0)
                lo += BLK
            tl.store(kvsum_ptr + unit * s_p + i_offs[:, None] * s_j + tl.arange(0, DPX)[None, :] * (1),
                     acc, mask=i_msk[:, None] & (tl.arange(0, DPX)[None, :] < DP))
            tl.store(ksum_ptr + unit * s_k + i_offs, ks, mask=i_msk)


# ---------------------------------------------------------------------------
# K7: reduce partials -> kvsum [BH, DP, DP], ksum [BH, DP]  (fp32)
# ---------------------------------------------------------------------------
@triton.jit
def sla_reduce_kernel(
    kvsum_p_ptr, ksum_p_ptr, kvsum_ptr, ksum_ptr,
    D,
    s_bhp, s_i, s_j, s_bh, s_bs,
    num_units, num_pids,
    DP: tl.constexpr, DPX: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    bpu = num_units // num_pids
    rem = num_units - bpu * bpu
    if pid < rem:
        ub_ = bpu + 1
        first = pid * (bpu + 1)
    else:
        ub_ = bpu
        first = rem * (bpu + 1) + (pid - rem) * bpu
    last = first + ub_
    j_offs = tl.arange(