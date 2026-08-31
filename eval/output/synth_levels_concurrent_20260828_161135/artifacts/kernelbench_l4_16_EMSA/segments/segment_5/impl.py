import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu


def _get_vec_core_num():
    try:
        return torch_npu.npu.npu_config.get_device_limit(0).get(
            "vector_core_num", 40
        )
    except Exception:
        return 40


@triton.jit
def _proj_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = tl.cdiv(M, BLOCK_M) * NUM_N
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_N
        block_n = block_idx % NUM_N
        m0 = block_m * BLOCK_M
        n0 = block_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a_off = (m0 + rm)[:, None] * K + (k0 + rk)[None, :]
            a_mask = ((m0 + rm)[:, None] < M) & ((k0 + rk)[None, :] < K)
            a = tl.load(x_ptr + a_off, mask=a_mask, other=0.0).to(tl.float32)
            b_off = (n0 + rn)[None, :] * K + (k0 + rk)[:, None]
            b_mask = ((n0 + rn)[None, :] < N) & ((k0 + rk)[:, None] < K)
            b = tl.load(w_ptr + b_off, mask=b_mask, other=0.0).to(tl.float32)
            acc += tl.dot(a, b)
        bias = tl.load(b_ptr + n0 + rn, mask=(n0 + rn) < N, other=0.0).to(
            tl.float32
        )
        acc += bias[None, :]
        y_off = (m0 + rm)[:, None] * N + (n0 + rn)[None, :]
        y_mask = ((m0 + rm)[:, None] < M) & ((n0 + rn)[None, :] < N)
        tl.store(y_ptr + y_off, acc, mask=y_mask)


@triton.jit
def _sr_kernel(
    x_ptr, sw_ptr, sb_ptr, gamma_ptr, beta_ptr, y_ptr,
    B, L, KR, KW, H, W, DM,
    R: tl.constexpr,
    BLOCK_DM: tl.constexpr,
    num_cores: tl.constexpr,
):
    pid = tl.program_id(0)
    cd = tl.arange(0, BLOCK_DM)
    cmask = cd < DM
    total = B * KR
    R2 = (R + 1) * (R + 1)
    PAD = R // 2
    dm_f = DM.to(tl.float32)
    for p in range(pid, total, num_cores):
        bt = p // KR
        p2 = p % KR
        oy = p2 // KW
        ox = p2 % KW
        acc = tl.zeros((BLOCK_DM,), dtype=tl.float32)
        for ky in range(0, R + 1):
            iy = oy * R + ky - PAD
            if (iy >= 0) & (iy < H):
                for kx in range(0, R + 1):
                    ix = ox * R + kx - PAD
                    if (ix >= 0) & (ix < W):
                        row = iy * W + ix
                        xv = tl.load(
                            x_ptr + (bt * L + row) * DM + cd,
                            mask=cmask, other=0.0,
                        ).to(tl.float32)
                        wv = tl.load(
                            sw_ptr + cd * R2 + ky * (R + 1) + kx,
                            mask=cmask, other=0.0,
                        ).to(tl.float32)
                        acc += wv * xv
        sb = tl.load(sb_ptr + cd, mask=cmask, other=0.0).to(tl.float32)
        acc += sb
        mean = tl.sum(acc, axis=0) / dm_f
        xc = acc - mean
        var = tl.sum(xc * xc, axis=0) / dm_f
        rstd = 1.0 / tl.sqrt(tl.maximum(var, 0.0) + 1e-5)
        gamma = tl.load(gamma_ptr + cd, mask=cmask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + cd, mask=cmask, other=0.0).to(tl.float32)
        res = xc * rstd * gamma + beta
        tl.store(y_ptr + p * DM + cd, res, mask=cmask)


@triton.jit
def _attn_plain_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, L, KL, NH, DK, DV,
    scale,
    num_cores: tl.constexpr,
    BH: tl.constexpr,
    BDJ: tl.constexpr,
    BDV: tl.constexpr,
    BT: tl.constexpr,
):
    pid = tl.program_id(0)
    dj = tl.arange(0, BDJ)
    dm = tl.arange(0, BDV)
    bh = tl.arange(0, BH)
    tt = tl.arange(0, BT)
    hmask = bh < NH
    dmask = dj < DK
    dvmask = dm < DV
    total = B * L
    for r in range(pid, total, num_cores):
        bt = r // L
        i = r % L
        m_h = tl.full((BH,), float("-inf"), dtype=tl.float32)
        z_h = tl.zeros((BH,), dtype=tl.float32)
        oacc = tl.zeros((BH, BDV), dtype=tl.float32)
        q_base = q_ptr + (bt * L + i) * NH * DK
        qh_acc = tl.zeros((BH, BDJ), dtype=tl.float32)
        for h in range(0, NH):
            qh = tl.load(q_base + h * DK + dj, mask=dmask, other=0.0).to(
                tl.float32
            )
            qh_acc += qh[None, :] * ((bh == h) & hmask)[:, None]
        for t0 in range(0, KL, BT):
            tmask = (t0 + tt) < KL
            s3 = tl.zeros((BT, BH), dtype=tl.float32)
            for h in range(0, NH):
                qh = tl.sum(qh_acc * ((bh == h) & hmask)[None, :], axis=0)
                kh = tl.load(
                    k_ptr + (bt * KL + t0 + tt)[:, None] * NH * DK + h * DK
                    + dj[None, :],
                    mask=tmask[:, None] & dmask[None, :], other=0.0,
                ).to(tl.float32)
                s = tl.sum(kh * qh[None, :], axis=1) * scale
                s3 += s[:, None] * ((bh == h) & hmask)[None, :]
            s3 = tl.where(tmask[:, None], s3, float("-inf"))
            sm = tl.max(s3, axis=0)
            new_m = tl.maximum(m_h, sm)
            rfac = tl.exp(m_h - new_m)
            e = tl.exp(s3 - new_m[None, :])
            z_h = z_h * rfac + tl.sum(e, axis=0)
            oacc = oacc * rfac[None, :]
            for h in range(0, NH):
                pcol = tl.sum(e * ((bh == h) & hmask)[None, :], axis=1)
                vt = tl.load(
                    v_ptr + (bt * KL + t0 + tt)[:, None] * NH * DV + h * DV
                    + dm[None, :],
                    mask=tmask[:, None] & dvmask[None, :], other=0.0,
                ).to(tl.float32)
                oadd = tl.sum(vt * pcol[:, None], axis=0)
                oacc = oacc + oadd[None, :] * ((bh == h) & hmask)[:, None]
            m_h = new_m
        z_h = tl.where(z_h == 0.0, 1.0, z_h)
        o = oacc / z_h[:, None]
        o_off = (bt * L + i) * NH * DV + (bh * DV + dm)[None, :]
        tl.store(
            o_ptr + o_off, o,
            mask=hmask[:, None] & dvmask[None, :],
        )


@triton.jit
def _attn_mix_rows_kernel(
    q_ptr, k_ptr, w1_ptr, b1_ptr, p_ptr,
    B, L, KL, NH, DK,
    scale,
    num_cores: tl.constexpr,
    BH: tl.constexpr,
    BDJ: tl.constexpr,
    BT: tl.constexpr,
):
    pid = tl.program_id(0)
    dj = tl.arange(0, BDJ)
    bh = tl.arange(0, BH)
    tt = tl.arange(0, BT)
    hmask = bh < NH
    dmask = dj < DK
    total = B * L
    for r in range(pid, total, num_cores):
        bt = r // L
        i = r % L
        m_h = tl.full((BH,), float("-inf"), dtype=tl.float32)
        z_h = tl.zeros((BH,), dtype=tl.float32)
        q_base = q_ptr + (bt * L + i) * NH * DK
        qh_acc = tl.zeros((BH, BDJ), dtype=tl.float32)
        for h in range(0, NH):
            qh = tl.load(q_base + h * DK + dj, mask=dmask, other=0.0).to(
                tl.float32
            )
            qh_acc += qh[None, :] * ((bh == h) & hmask)[:, None]
        wm = tl.load(
            w1_ptr + bh[None, :] * NH + bh[:, None],
            mask=hmask[:, None] & hmask[None, :], other=0.0,
        ).to(tl.float32)
        wmT = tl.zeros((BH, BH), dtype=tl.float32)
        for h in range(0, NH):
            wcol = tl.sum(wm * ((bh == h) & hmask)[None, :], axis=1)
            wmT += wcol[:, None] * ((bh == h) & hmask)[None, :]
        tb = tl.load(b1_ptr + bh, mask=hmask, other=0.0).to(tl.float32)
        for t0 in range(0, KL, BT):
            tmask = (t0 + tt) < KL
            smat = tl.zeros((BT, BH), dtype=tl.float32)
            for h in range(0, NH):
                qh = tl.sum(qh_acc * ((bh == h) & hmask)[None, :], axis=0)
                kh = tl.load(
                    k_ptr + (bt * KL + t0 + tt)[:, None] * NH * DK + h * DK
                    + dj[None, :],
                    mask=tmask[:, None] & dmask[None, :], other=0.0,
                ).to(tl.float32)
                s = tl.sum(kh * qh[None, :], axis=1)
                smat += s[:, None] * ((bh == h) & hmask)[None, :]
            s3 = tl.dot(smat * scale, wmT) + tb[None, :] * tmask[:, None]
            s3 = tl.where(tmask[:, None], s3, float("-inf"))
            sm = tl.max(s3, axis=0)
            new_m = tl.maximum(m_h, sm)
            rfac = tl.exp(m_h - new_m)
            e = tl.exp(s3 - new_m[None, :])
            z_h = z_h * rfac + tl.sum(e, axis=0)
            m_h = new_m
        z_safe = tl.where(z_h == 0.0, 1.0, z_h)
        for t0 in range(0, KL, BT):
            tmask = (t0 + tt) < KL
            smat = tl.zeros((BT, BH), dtype=tl.float32)
            for h in range(0, NH):
                qh = tl.sum(qh_acc * ((bh == h) & hmask)[None, :], axis=0)
                kh = tl.load(
                    k_ptr + (bt * KL + t0 + tt)[:, None] * NH * DK + h * DK
                    + dj[None, :],
                    mask=tmask[:, None] & dmask[None, :], other=0.0,
                ).to(tl.float32)
                s = tl.sum(kh * qh[None, :], axis=1)
                smat += s[:, None] * ((bh == h) & hmask)[None, :]
            s3 = tl.dot(smat * scale, wmT) + tb[None, :] * tmask[:, None]
            e = tl.exp(s3 - m_h[None, :])
            pn = e / z_safe[None, :]
            p_off = (bt * L + i) * NH * KL + (bh * KL)[None, :] + (t0 + tt)[:, None]
            pmask = hmask[None, :] & tmask[:, None]
            tl.store(p_ptr + p_off, pn, mask=pmask)


@triton.jit
def _attn_stats_partial_kernel(
    p_ptr, partial_ptr,
    B, L, KL, NH,
    NT,
    BIR: tl.constexpr,
    BT: tl.constexpr,
    num_cores: tl.constexpr,
):
    pid = tl.program_id(0)
    ri = tl.arange(0, BIR)
    tt = tl.arange(0, BT)
    num_i = tl.cdiv(L, BIR)
    num_t = tl.cdiv(KL, BT)
    nblk_h = num_i * num_t
    total = B * NH * nblk_h
    for blk in range(pid, total, num_cores):
        bt = blk // (NH * nblk_h)
        rest = blk % (NH * nblk_h)
        h = rest // nblk_h
        it = rest % nblk_h
        i0 = (it // num_t) * BIR
        t0 = (it % num_t) * BT
        tile = tl.load(
            p_ptr + (bt * L + i0 + ri)[:, None] * NH * KL + h * KL
            + (t0 + tt)[None, :],
            mask=((i0 + ri)[:, None] < L) & ((t0 + tt)[None, :] < KL),
            other=0.0,
        ).to(tl.float32)
        s1 = tl.sum(tl.sum(tile, axis=1), axis=0)
        s2 = tl.sum(tl.sum(tile * tile, axis=1), axis=0)
        part_off = ((bt * NH + h) * NT + it) * 2
        tl.store(partial_ptr + part_off, s1)
        tl.store(partial_ptr + part_off + 1, s2)


@triton.jit
def _attn_stats_finalize_kernel(
    partial_ptr, stat_ptr,
    B, NH,
    NT,
    L, KL,
    num_cores: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.arange(0, 128)
    count = (L * KL).to(tl.float32)
    for bh_idx in range(pid, B * NH, num_cores):
        b = bh_idx // NH
        h = bh_idx % NH
        s1 = 0.0
        s2 = 0.0
        for it0 in range(0, NT, 128):
            imask = (it0 + idx) < NT
            base = ((b * NH + h) * NT + it0 + idx) * 2
            v1 = tl.load(partial_ptr + base, mask=imask, other=0.0)
            v2 = tl.load(partial_ptr + base + 1, mask=imask, other=0.0)
            s1 += tl.sum(v1, axis=0)
            s2 += tl.sum(v2, axis=0)
        mean = s1 / count
        var = s2 / count - mean * mean
        rstd = 1.0 / tl.sqrt(tl.maximum(var, 0.0) + 1e-5)
        off = (b * NH + h) * 2
        tl.store(stat_ptr + off, mean)
        tl.store(stat_ptr + off + 1, rstd)


@triton.jit
def _attn_pv_kernel(
    p_ptr, v_ptr, stat_ptr, o_ptr,
    B, L, KL, NH, DK, DV,
    scale,
    num_cores: tl.constexpr,
    BH: tl.constexpr,
    BDV: tl.constexpr,
    BT: tl.constexpr,
):
    pid = tl.program_id(0)
    dm = tl.arange(0, BDV)
    bh = tl.arange(0, BH)
    tt = tl.arange(0, BT)
    hmask = bh < NH
    dvmask = dm < DV
    total = B * L
    for r in range(pid, total, num_cores):
        bt = r // L
        i = r % L
        oacc = tl.zeros((BH, BDV), dtype=tl.float32)
        for h in range(0, NH):
            sel = (bh == h) & hmask
            mean_s = tl.load(stat_ptr + bt * NH * 2 + h).to(tl.float32)
            rstd_s = tl.load(stat_ptr + bt * NH * 2 + h + 1).to(tl.float32)
            acc_h = tl.zeros((BDV,), dtype=tl.float32)
            for t0 in range(0, KL, BT):
                tmask = (t0 + tt) < KL
                pt = tl.load(
                    p_ptr + (bt * L + i) * NH * KL + h * KL + t0 + tt,
                    mask=tmask, other=0.0,
                ).to(tl.float32)
                vt = tl.load(
                    v_ptr + (bt * KL + t0 + tt)[:, None] * NH * DV + h * DV
                    + dm[None, :],
                    mask=tmask[:, None] & dvmask[None, :], other=0.0,
                ).to(tl.float32)
                pn = (pt - mean_s) * rstd_s
                oadd = tl.sum(vt * pn[:, None], axis=0)
                acc_h += oadd
            oacc = oacc + acc_h[None, :] * sel[:, None]
        o_off = (bt * L + i) * NH * DV + (bh * DV + dm)[None, :]
        tl.store(
            o_ptr + o_off, oacc, mask=hmask[:, None] & dvmask[None, :],
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.VEC_CORE_NUM = _get_vec_core_num()

    def forward(
        self,
        queries,
        keys,
        values,
        n_heads,
        height,
        width,
        ratio,
        d_k,
        d_v,
        apply_transform,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        out_weight,
        out_bias,
        sr_weight,
        sr_bias,
        sr_norm_weight,
        sr_norm_bias,
        transform_weight,
        transform_bias,
    ):
        batch, query_length, d_model = queries.shape
        key_length = keys.shape[1]
        if query_length != height * width:
            raise ValueError("height * width must match the query length")

        nh = n_heads
        hgt = height
        wid = width
        ratio = int(ratio)
        dk = d_k
        dv = d_v
        apply_transform = bool(apply_transform)
        L = query_length
        device = queries.device
        vec = self.VEC_CORE_NUM

        q_ncol = nh * dk
        v_ncol = nh * dv

        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 32

        q_proj = torch.empty(
            (batch, L, q_ncol), device=device, dtype=torch.float32
        )
        nblocks = triton.cdiv(batch * L, BLOCK_M) * triton.cdiv(q_ncol,
                                                                BLOCK_N)
        grid_size = nblocks if nblocks < vec else vec
        _proj_kernel[grid_size](
            queries, q_weight, q_bias, q_proj,
            batch * L, q_ncol, d_model,
            num_cores=grid_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        k_in = keys
        v_in = values
        key_len = key_length
        if ratio > 1:
            kh = (hgt + 2 * (ratio // 2) - (ratio + 1)) // ratio + 1
            kw = (wid + 2 * (ratio // 2) - (ratio + 1)) // ratio + 1
            key_len = kh * kw
            red = torch.empty(
                (batch, key_len, d_model), device=device,
                dtype=torch.float32,
            )
            dm_np2 = triton.next_power_of_2(d_model)
            BLOCK_DM = dm_np2 if dm_np2 >= 16 else 16
            sr_grid = (batch * key_len) if (batch * key_len) < vec else vec
            _sr_kernel[sr_grid](
                queries, sr_weight, sr_bias, sr_norm_weight, sr_norm_bias,
                red,
                batch, L, key_len, kw, hgt, wid, d_model,
                R=ratio, BLOCK_DM=BLOCK_DM,
                num_cores=sr_grid,
            )
            k_in = red
            v_in = red

        k_proj = torch.empty(
            (batch, key_len, q_ncol), device=device, dtype=torch.float32
        )
        nblocks = triton.cdiv(batch * key_len, BLOCK_M) * triton.cdiv(
            q_ncol, BLOCK_N
        )
        grid_size = nblocks if nblocks < vec else vec
        _proj_kernel[grid_size](
            k_in, k_weight, k_bias, k_proj,
            batch * key_len, q_ncol, d_model,
            num_cores=grid_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        v_proj = torch.empty(
            (batch, key_len, v_ncol), device=device, dtype=torch.float32
        )
        nblocks = triton.cdiv(batch * key_len, BLOCK_M) * triton.cdiv(
            v_ncol, BLOCK_N
        )
        grid_size = nblocks if nblocks < vec else vec
        _proj_kernel[grid_size](
            v_in, v_weight, v_bias, v_proj,
            batch * key_len, v_ncol, d_model,
            num_cores=grid_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        out_mid = torch.empty(
            (batch, L, v_ncol), device=device, dtype=torch.float32
        )
        scale = 1.0 / math.sqrt(dk)

        if apply_transform and nh > 1:
            p_buf = torch.empty(
                (batch, L, nh, key_len), device=device,
                dtype=torch.float32,
            )
            nh_np2 = triton.next_power_of_2(nh)
            BH = 64 if nh_np2 > 64 else (nh_np2 if nh_np2 >= 16 else 16)
            dk_np2 = triton.next_power_of_2(dk)
            BDJ = 128 if dk_np2 > 128 else (dk_np2 if dk_np2 >= 16 else 16)
            BT = 32
            row_work = batch * L
            grid_size = row_work if row_work < vec else vec
            _attn_mix_rows_kernel[grid_size](
                q_proj, k_proj, transform_weight, transform_bias, p_buf,
                batch, L, key_len, nh, dk, scale,
                num_cores=grid_size,
                BH=BH, BDJ=BDJ, BT=BT,
            )

            BIR = 32
            BT_S = 32
            num_i = triton.cdiv(L, BIR)
            num_t = triton.cdiv(key_len, BT_S)
            nt = num_i * num_t
            partial = torch.empty(
                (batch * nh * nt * 2,), device=device,
                dtype=torch.float32,
            )
            part_work = batch * nh * nt
            grid_size = part_work if part_work < vec else vec
            _attn_stats_partial_kernel[grid_size](
                p_buf, partial, batch, L, key_len, nh, nt,
                BIR=BIR, BT=BT_S,
                num_cores=grid_size,
            )
            stat = torch.empty(
                (batch * nh * 2,), device=device, dtype=torch.float32
            )
            bh_work = batch * nh
            grid_size = bh_work if bh_work < vec else vec
            _attn_stats_finalize_kernel[grid_size](
                partial, stat, batch, nh, nt, L, key_len,
                num_cores=grid_size,
            )
            dv_np2 = triton.next_power_of_2(dv)
            BDV = 128 if dv_np2 > 128 else (dv_np2 if dv_np2 >= 16 else 16)
            pv_work = batch * L
            grid_size = pv_work if pv_work < vec else vec
            _attn_pv_kernel[grid_size](
                p_buf, v_proj, stat, out_mid,
                batch, L, key_len, nh, dk, dv, scale,
                num_cores=grid_size,
                BH=BH, BDV=BDV, BT=BT,
            )
        else:
            nh_np2 = triton.next_power_of_2(nh)
            BH = 64 if nh_np2 > 64 else (nh_np2 if nh_np2 >= 16 else 16)
            dk_np2 = triton.next_power_of_2(dk)
            BDJ = 128 if dk_np2 > 128 else (dk_np2 if dk_np2 >= 16 else 16)
            dv_np2 = triton.next_power_of_2(dv)
            BDV = 128 if dv_np2 > 128 else (dv_np2 if dv_np2 >= 16 else 16)
            BT = 32
            row_work = batch * L
            grid_size = row_work if row_work < vec else vec
            _attn_plain_kernel[grid_size](
                q_proj, k_proj, v_proj, out_mid,
                batch, L, key_len, nh, dk, dv, scale,
                num_cores=grid_size,
                BH=BH, BDJ=BDJ, BDV=BDV, BT=BT,
            )

        out = torch.empty(
            (batch, L, d_model), device=device, dtype=queries.dtype
        )
        nblocks = triton.cdiv(batch * L, BLOCK_M) * triton.cdiv(d_model,
                                                                BLOCK_N)
        grid_size = nblocks if nblocks < vec else vec
        _proj_kernel[grid_size](
            out_mid, out_weight, out_bias, out,
            batch * L, d_model, v_ncol,
            num_cores=grid_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out