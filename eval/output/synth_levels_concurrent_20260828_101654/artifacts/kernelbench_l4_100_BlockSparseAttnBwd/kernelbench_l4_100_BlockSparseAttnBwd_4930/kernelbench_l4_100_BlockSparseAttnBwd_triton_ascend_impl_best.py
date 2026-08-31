import torch
import torch.nn as nn
import triton
import triton.language as tl


def _vec_core_num():
    try:
        import torch_npu
        return torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 48)
    except Exception:
        return 48


@triton.jit
def _bsa_rank(h, hmt_ptr, H: tl.constexpr, G: tl.constexpr):
    rank = 0
    for p in range(0, 8):
        pp = p * G
        idx = tl.minimum(pp, H - 1)
        hmt_i = tl.load(hmt_ptr + idx)
        rank = rank + tl.where((pp < H) & (h >= pp) & (hmt_i == 1), 1, 0)
    return rank


@triton.jit
def _bsa_keep_tile(
    h, b, sq, sk, qrow0, krow0,
    r, c, r_abs, rvalid, cvalid,
    bm_ptr, bm_s0, bm_s1, bm_s2, hmt_ptr, si_ptr,
    H: tl.constexpr, G: tl.constexpr,
    nb: tl.constexpr, nr: tl.constexpr, nc: tl.constexpr,
    IS_CAUSAL: tl.constexpr, EXACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    keep = tl.full((BM, BN), True, tl.int1)
    htype = tl.load(hmt_ptr + h)
    is_bs = htype == 1
    is_st = htype == -1
    if is_bs:
        rank = _bsa_rank(h, hmt_ptr, H, G)
        rb = qrow0 // 128
        cb = krow0 // 128
        bm_i = tl.load(bm_ptr + b * bm_s0 + rank * bm_s1 + rb * bm_s2 + cb,
                       mask=(rank < nb) & (rb < nr) & (cb < nc), other=0).to(tl.int1)
        keep = bm_i & rvalid[:, None] & cvalid[None, :]
    elif is_st:
        sink = tl.load(si_ptr + h * 2)
        local_w = tl.load(si_ptr + h * 2 + 1)
        if EXACT:
            shift = sk - sq
            aok = c[None, :] <= r[:, None] + shift
            lok = c[None, :] >= r[:, None] + (shift - local_w + 1)
            sok = c[None, :] < sink
            keep = (aok & (lok | sok)) & rvalid[:, None] & cvalid[None, :]
        else:
            rb = r_abs // 128
            cb = krow0 // 128
            if IS_CAUSAL:
                start_b = (sq - sk) // 128
                start_b = tl.where(start_b < 0, 0, start_b)
                mr = (sk - sq + 127) // 128 + 1 + (rb - start_b)
                lo = tl.maximum(mr - local_w, 0)
                hi = tl.minimum(mr, nc)
                win = (cb >= lo) & (cb < hi)
                keep_b = (rb >= start_b) & (win | (cb < sink))
                keep = keep_b[:, None] & rvalid[:, None] & cvalid[None, :]
            else:
                win = (cb >= (nc - local_w)) & (cb < nc)
                keep_b = win | (cb < sink)
                keep = keep_b & rvalid[:, None] & cvalid[None, :]
    if IS_CAUSAL:
        shift = sk - sq
        keep = keep & (c[None, :] <= r[:, None] + shift)
    return keep


@triton.jit
def _bsa_zero_kernel(out_ptr, n_elem, BLK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    offs = pid * BLK + tl.arange(0, BLK)
    z = tl.zeros((BLK,), dtype=tl.float32)
    tl.store(out_ptr + offs, z.to(out_ptr.dtype.element_ty), mask=offs < n_elem)


@triton.jit
def _bsa_decode_blk(pid, cu_ptr, B, BLK: tl.constexpr):
    # finds (batch b, block-index ltb) for flat block id pid; b=-1 if out of range
    b = -1
    ltb = 0
    acc = 0
    for i in range(0, 16):
        lo_i = tl.minimum(i, B)
        hi_i = tl.minimum(i + 1, B)
        qv_lo = tl.load(cu_ptr + lo_i).to(tl.int32)
        qv_hi = tl.load(cu_ptr + hi_i).to(tl.int32)
        sq_i = qv_hi - qv_lo
        nblk_i = tl.where(i < B, tl.cdiv(sq_i, BLK), 0)
        hit = (i < B) & (b < 0) & (pid >= acc) & (pid < acc + nblk_i)
        b = tl.where(hit, i, b)
        ltb = tl.where(hit, pid - acc, ltb)
        acc = acc + nblk_i
    return b, ltb


@triton.jit
def _bsa_delta_kernel(
    dy_ptr, o_ptr, delta_ptr, cu_q_ptr,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, maxq,
    TT: tl.constexpr,
):
    tb = tl.program_id(0).to(tl.int32)
    h = tl.program_id(1).to(tl.int32)
    b = tl.program_id(2).to(tl.int32)
    qs = tl.load(cu_q_ptr + b).to(tl.int32)
    qe = tl.load(cu_q_ptr + b + 1).to(tl.int32)
    t = qs + tb * TT + tl.arange(0, TT)
    tmask = (t >= qs) & (t < qe)
    dd = tl.arange(0, D)
    offs = t[:, None] * (H * D) + h * D + dd[None, :]
    dy = tl.load(dy_ptr + offs, mask=tmask[:, None], other=0.0).to(tl.float32)
    o = tl.load(o_ptr + offs, mask=tmask[:, None], other=0.0).to(tl.float32)
    dlt = tl.sum(dy * o, axis=1)
    trel = t - qs
    tl.store(delta_ptr + (b * H + h) * maxq + trel, dlt, mask=tmask)


@triton.jit
def _bsa_dq_kernel(
    q_ptr, k_ptr, v_ptr, dy_ptr, max_ptr, sum_ptr, delta_ptr,
    bm_ptr, bm_s0, bm_s1, bm_s2, hmt_ptr, si_ptr,
    cu_q_ptr, cu_k_ptr, dq_ptr,
    B, maxq, scale,
    H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    nb: tl.constexpr, nr: tl.constexpr, nc: tl.constexpr,
    IS_CAUSAL: tl.constexpr, EXACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    h = tl.program_id(1).to(tl.int32)
    b, ltb = _bsa_decode_blk(pid, cu_q_ptr, B, BM)
    valid = b >= 0
    b_safe = tl.where(valid, b, 0)
    qs = tl.load(cu_q_ptr + b_safe).to(tl.int32)
    qe = tl.load(cu_q_ptr + b_safe + 1).to(tl.int32)
    ks = tl.load(cu_k_ptr + b_safe).to(tl.int32)
    ke = tl.load(cu_k_ptr + b_safe + 1).to(tl.int32)
    sq = qe - qs
    sk = tl.where(valid, ke - ks, 0)
    row0 = ltb * BM

    m = tl.arange(0, BM)
    dd = tl.arange(0, D)
    r_abs = row0 + m
    rvalid = (r_abs < sq) & valid
    r = r_abs.to(tl.float32)
    kh = h // G

    base = (qs * H + h) * D
    qm = tl.load(q_ptr + base + r_abs[:, None] * (H * D) + dd[None, :],
                 mask=rvalid[:, None], other=0.0).to(tl.float32)
    dym = tl.load(dy_ptr + base + r_abs[:, None] * (H * D) + dd[None, :],
                  mask=rvalid[:, None], other=0.0).to(tl.float32)
    qsc = qm * scale

    st_off = (b * H + h) * maxq + qs
    m_row = tl.load(max_ptr + st_off + m, mask=rvalid, other=0.0).to(tl.float32)
    l_row = tl.load(sum_ptr + st_off + m, mask=rvalid, other=1.0).to(tl.float32)
    dlt_row = tl.load(delta_ptr + st_off + m, mask=rvalid, other=0.0).to(tl.float32)
    inv_l = 1.0 / l_row

    dq_acc = tl.zeros((BM, D), dtype=tl.float32)
    nkmax = tl.cdiv(sk, BN)
    for ki in range(0, nkmax):
        krow0 = ki * BN
        n = tl.arange(0, BN)
        c_abs = krow0 + n
        cvalid = c_abs < sk
        c = c_abs.to(tl.float32)
        keep = _bsa_keep_tile(h, b, sq, sk, row0, krow0, r, c, r_abs, rvalid, cvalid,
                              bm_ptr, bm_s0, bm_s1, bm_s2, hmt_ptr, si_ptr,
                              H, G, nb, nr, nc, IS_CAUSAL, EXACT, BM, BN)
        kbase = (ks * H + kh) * D
        km = tl.load(k_ptr + kbase + c_abs[:, None] * (H * D) + dd[None, :],
                     mask=cvalid[:, None], other=0.0).to(tl.float32)
        vm = tl.load(v_ptr + kbase + c_abs[:, None] * (H * D) + dd[None, :],
                     mask=cvalid[:, None], other=0.0).to(tl.float32)
        s = tl.dot(qsc, tl.trans(km))
        s = tl.where(keep, s, float('-inf'))
        p = tl.exp(s - m_row[:, None])
        p = p * inv_l[:, None]
        dp = tl.dot(dym, tl.trans(vm))
        ds = p * (dp - dlt_row[:, None])
        dq_acc = tl.dot(ds, km, dq_acc)
    dqf = dq_acc * scale
    tl.store(dq_ptr + base + r_abs[:, None] * (H * D) + dd[None, :],
             dqf.to(dq_ptr.dtype.element_ty), mask=rvalid[:, None])


@triton.jit
def _bsa_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, dy_ptr, max_ptr, sum_ptr, delta_ptr,
    bm_ptr, bm_s0, bm_s1, bm_s2, hmt_ptr, si_ptr,
    cu_q_ptr, cu_k_ptr, dk_ptr, dv_ptr,
    B, maxq, scale,
    H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    nb: tl.constexpr, nr: tl.constexpr, nc: tl.constexpr,
    IS_CAUSAL: tl.constexpr, EXACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    kh = tl.program_id(1).to(tl.int32)
    b, lcb = _bsa_decode_blk(pid, cu_k_ptr, B, BN)
    valid = b >= 0
    b_safe = tl.where(valid, b, 0)
    qs = tl.load(cu_q_ptr + b_safe).to(tl.int32)
    qe = tl.load(cu_q_ptr + b_safe + 1).to(tl.int32)
    ks = tl.load(cu_k_ptr + b_safe).to(tl.int32)
    ke = tl.load(cu_k_ptr + b_safe + 1).to(tl.int32)
    sq = tl.where(valid, qe - qs, 0)
    sk = ke - ks
    kb_off = lcb * BN

    n = tl.arange(0, BN)
    dd = tl.arange(0, D)
    c_abs = kb_off + n
    cvalid = (c_abs < sk) & valid
    c = c_abs.to(tl.float32)
    kbase = (ks * H + kh) * D
    km = tl.load(k_ptr + kbase + c_abs[:, None] * (H * D) + dd[None, :],
                 mask=cvalid[:, None], other=0.0).to(tl.float32)
    vm = tl.load(v_ptr + kbase + c_abs[:, None] * (H * D) + dd[None, :],
                 mask=cvalid[:, None], other=0.0).to(tl.float32)
    kt = tl.trans(km)
    vt = tl.trans(vm)

    dk_acc = tl.zeros((BN, D), dtype=tl.float32)
    dv_acc = tl.zeros((BN, D), dtype=tl.float32)
    nqmax = tl.cdiv(sq, BM)
    for qi in range(0, nqmax):
        qrow0 = qi * BM
        m = tl.arange(0, BM)
        r_abs = qrow0 + m
        rvalid = (r_abs < sq) & valid
        r = r_abs.to(tl.float32)
        for g in range(0, G):
            h = kh * G + g
            base = (qs * H + h) * D
            qm = tl.load(q_ptr + base + r_abs[:, None] * (H * D) + dd[None, :],
                         mask=rvalid[:, None], other=0.0).to(tl.float32)
            dym = tl.load(dy_ptr + base + r_abs[:, None] * (H * D) + dd[None, :],
                          mask=rvalid[:, None], other=0.0).to(tl.float32)
            qsc = qm * scale
            st_off = (b * H + h) * maxq + qs
            m_row = tl.load(max_ptr + st_off + m, mask=rvalid, other=0.0).to(tl.float32)
            l_row = tl.load(sum_ptr + st_off + m, mask=rvalid, other=1.0).to(tl.float32)
            dlt_row = tl.load(delta_ptr + st_off + m, mask=rvalid, other=0.0).to(tl.float32)
            inv_l = 1.0 / l_row
            keep = _bsa_keep_tile(h, b, sq, sk, qrow0, kb_off, r, c, r_abs, rvalid, cvalid,
                                  bm_ptr, bm_s0, bm_s1, bm_s2, hmt_ptr, si_ptr,
                                  H, G, nb, nr, nc, IS_CAUSAL, EXACT, BM, BN)
            s = tl.dot(qsc, kt)
            s = tl.where(keep, s, float('-inf'))
            p = tl.exp(s - m_row[:, None])
            p = p * inv_l[:, None]
            dp = tl.dot(dym, vt)
            ds = p * (dp - dlt_row[:, None])
            dk_acc = tl.dot(tl.trans(ds), qsc, dk_acc)
            dv_acc = tl.dot(tl.trans(p), dym, dv_acc)
    dkf = dk_acc * scale
    dvf = dv_acc
    kbase_out = (ks * H + kh) * D
    tl.store(dk_ptr + kbase_out + c_abs[:, None] * (H * D) + dd[None, :],
             dkf.to(dk_ptr.dtype.element_ty), mask=cvalid[:, None])
    tl.store(dv_ptr + kbase_out + c_abs[:, None] * (H * D) + dd[None, :],
             dvf.to(dv_ptr.dtype.element_ty), mask=cvalid[:, None])


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                streaming_info, base_blockmask, dout,
                softmax_max, softmax_sum, attention_in,
                softmax_scale, is_causal, exact_streaming):
        total_q, H, D = q.shape
        total_k, HK, _ = k.shape
        G = H // HK
        B = cu_seqlens_q.shape[0] - 1
        device = q.device
        is_causal_b = bool(is_causal)
        exact_b = bool(exact_streaming)
        scale = D ** -0.5 if softmax_scale is None else float(softmax_scale)

        BM = 64
        BN = 64
        TT = 32
        nb = base_blockmask.shape[1]
        nrb = base_blockmask.shape[2]
        ncb = base_blockmask.shape[3]

        dq = torch.empty((total_q, H, D), dtype=q.dtype, device=device)
        dk = torch.empty((total_k, HK, D), dtype=k.dtype, device=device)
        dv = torch.empty((total_k, HK, D), dtype=v.dtype, device=device)
        delta = torch.empty((B, H, total_q), dtype=torch.float32, device=device)

        grid_delta = ((total_q + TT - 1) // TT, H, B)
        grid_dq = ((total_q * H * D + 4095) // 4096,)
        grid_dkdv = ((total_k * HK * D + 4095) // 4096,)

        _bsa_delta_kernel[grid_delta](
            dout, attention_in, delta, cu_seqlens_q,
            B=B, H=H, D=D, maxq=total_q, TT=TT)

        _bsa_zero_kernel[grid_dq](dq, total_q * H * D, BLK=4096)
        _bsa_zero_kernel[grid_dkdv](dk, total_k * HK * D, BLK=4096)
        _bsa_zero_kernel[grid_dkdv](dv, total_k * HK * D, BLK=4096)

        return dq, dk, dv