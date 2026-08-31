import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


def _num_cores():
    try:
        import torch_npu
        return max(1, int(torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)))
    except Exception:
        try:
            return max(1, triton.runtime.driver.active.utils.get_device_properties(0).multi_processor_count)
        except Exception:
            return 24


def _block_mean(x, BLK):
    # fp32 块均值（尾块按实际 token 数），结果回铸 x.dtype —— 与参考
    # _block_mean（F.pad + view + sum/cnt）语义一致。
    device = x.device
    B, H, L, D = x.shape
    LB = (L + BLK - 1) // BLK
    pad = LB * BLK - L
    xp = F.pad(x.float(), (0, 0, 0, pad)).view(B, H, LB, BLK, D)
    cnt = torch.full((LB,), BLK, dtype=torch.float32, device=device)
    if pad:
        cnt[-1] = BLK - pad
    return (xp.sum(3) / cnt.view(1, 1, LB, 1)).to(x.dtype)


def _build_lut(q, k, topk, BLKQ, BLKK):
    """Block selection, bit-exact with the reference (see get_block_map).

    Returns (tok32, real_topk, MQ):
      tok32: int32 [B*H, MQ, real_topk*BK] raw key token positions
             (values may be >= L for padded/tail blocks; kernels mask them).
    """
    device = q.device
    dtype = q.dtype
    B, H, L, D = q.shape
    arg_k = k - k.float().mean(dim=-2, keepdim=True).to(dtype)
    qm = _block_mean(q, BLKQ)
    km = _block_mean(arg_k, BLKK)
    pooled = (qm.float() @ km.float().transpose(-1, -2)).to(dtype)
    MQ, NK = pooled.shape[2], pooled.shape[-1]
    real_topk = int(topk * NK)
    if real_topk > NK:
        real_topk = NK
    assert real_topk >= 1, "real_topk=0 yields NaN in official kernel (undefined)"
    lut = torch.topk(pooled.float(), real_topk, dim=-1, sorted=False).indices
    lut = lut.sort(dim=-1).values  # [B,H,MQ,real_topk] ascending
    BK = triton.next_power_of_2(BLKK)
    if BK == BLKK:
        tok = (lut[..., None] * BLKK
               + torch.arange(BK, device=device)).reshape(B * H, MQ, real_topk * BK)
    else:
        # pad key block if BLKK not power of two (tok layout must be RT*BK);
        # pad with sentinel block index NK so tok >= L (masked invalid)
        lutp = F.pad(lut, (0, BK - BLKK), value=NK)
        tok = (lutp.reshape(B * H, MQ, -1, 1) * BLKK
               + torch.arange(BK, device=device)).reshape(B * H, MQ, real_topk * BK)
    return tok.to(torch.int32).contiguous(), real_topk, MQ


@triton.jit
def _sla_fmap_kernel(
    x_ptr, y_ptr, L,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    T: tl.constexpr, FM: tl.constexpr,
    N_TILES_BH, N_PER_BH, GRID,
):
    """Per-token feature map. x: [BH, L, D] half/bf16, y: [BH, L, D] fp32.
    FM: 0=relu, 1=elu+1, 2=softmax(dim=-1)."""
    for i in range(tl.program_id(0), N_TILES_BH, GRID):
        bh = i // N_PER_BH
        t0 = (i % N_PER_BH) * T
        rows = t0 + tl.arange(0, T)
        d = tl.arange(0, BLOCK_D)
        off = bh * L * D
        ptrs = off + rows[:, None] * D + d[None, :]
        mask = (rows[:, None] < L) & (d[None, :] < D)
        x = tl.load(x_ptr + ptrs, mask=mask, other=-1e30).to(tl.float32)
        if FM == 0:
            c = tl.maximum(x, 0.0)
        elif FM == 1:
            c = tl.where(x > 0.0, x + 1.0, tl.exp(x))
        else:
            mx = tl.max(x, 1)
            e = tl.exp(x - mx[:, None])
            c = e / tl.sum(e, 1)[:, None]
        tl.store(y_ptr + ptrs, c, mask=mask)


@triton.jit
def _sla_kv_kernel(
    ck_ptr, v_ptr, w_ptr, al_ptr, ks_ptr,
    L, N_PER_BH,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    T: tl.constexpr,
):
    """kvsum[D,D] = c_k^T @ v, ksum[D] = sum of c_k over L,
    A_l = kvsum @ W_l^T (fp32). pid == bh."""
    pid = tl.program_id(0)
    Loff = pid * L * D
    d = tl.arange(0, BLOCK_D)
    dmask = d < D
    kv = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    ks = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t in range(N_PER_BH):
        rows = t * T + tl.arange(0, T)
        mrow = (rows[:, None] < L) & dmask[None, :]
        off = Loff + rows[:, None] * D + d[None, :]
        ck = tl.load(ck_ptr + off, mask=mrow, other=0.0)
        vv = tl.load(v_ptr + off, mask=mrow, other=0.0).to(tl.float32)
        kv = tl.dot(tl.trans(ck), vv, kv)
        ks += tl.sum(ck, axis=0)
    # A_l[i, j] = sum_d kv[i, d] * W[j, d]  ==  dot(kv, trans(Wfull))
    wf = tl.load(w_ptr + d[:, None] * D + d[None, :], mask=dmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
    al = tl.dot(kv, tl.trans(wf))
    tl.store(al_ptr + pid * BLOCK_D * BLOCK_D + d[:, None] * BLOCK_D + d[None, :],
             al, mask=dmask[:, None] & dmask[None, :])
    tl.store(ks_ptr + pid * BLOCK_D + d, ks, mask=dmask)


@triton.jit
def _sla_sparse_kernel(
    q_ptr, k_ptr, v_ptr, tok_ptr, os_ptr,
    L, MQ, RT, BH,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr, BLKK: tl.constexpr,
    LOG_SCALE: tl.constexpr, NSUB: tl.constexpr,
    GRID,
):
    """Flash-style sparse attention. tok: [BH, MQ, RT*BK] int32 (raw positions,
    may be >= L for tail block). o_s stored fp32 [BH, L, D]."""
    SUBQ: tl.constexpr = BQ // NSUB
    for i in range(tl.program_id(0), BH * MQ, GRID):
        m = i % MQ
        bh = i // MQ
        Loff = bh * L * D
        d = tl.arange(0, BLOCK_D)
        dmask = d < D
        t0 = i * RT * BK
        kt = tl.arange(0, BK)
        toks_all = tok_ptr + t0 + kt
        for qs in range(NSUB):
            rows = m * BQ + qs * SUBQ + tl.arange(0, SUBQ)
            rmask = rows < L
            rowoff = Loff + rows[:, None] * D
            qm = tl.load(q_ptr + rowoff + d[None, :],
                         mask=rmask[:, None] & dmask[None, :], other=0.0)
            acc = tl.zeros((SUBQ, BLOCK_D), dtype=tl.float32)
            mi = tl.full((SUBQ,), -1e30, tl.float32)
            li = tl.zeros((SUBQ,), dtype=tl.float32)
            for j in range(RT):
                toks = tl.load(toks_all + j * BK, mask=(kt < BLKK), other=L)
                valid = toks < L
                pos = tl.minimum(toks, L - 1)
                kseg = tl.load(k_ptr + Loff + pos[:, None] * D + d[None, :],
                               mask=dmask[None, :], other=0.0)
                vseg = tl.load(v_ptr + Loff + pos[:, None] * D + d[None, :],
                               mask=dmask[None, :], other=0.0).to(tl.float32)
                # QK: low-precision dot (inputs exact w.r.t. fp32 reference, fp32 acc)
                s2 = tl.dot(qm, tl.trans(kseg)) * LOG_SCALE
                s2 = tl.where(valid[None, :], s2, -1e30)
                mn = tl.maximum(mi, tl.max(s2, 1))
                alpha = tl.exp2(mi - mn)
                p = tl.exp2(s2 - mn[:, None])
                p = tl.where(valid[None, :], p, 0.0)
                li = li * alpha + tl.sum(p, 1)
                # PV: fp32 dot (p has no low-precision cast; matches fp32 reference)
                acc = acc * alpha[:, None] + tl.dot(p, vseg)
                mi = mn
            out = acc / li[:, None]
            tl.store(os_ptr + rowoff + d[None, :], out,
                     mask=rmask[:, None] & dmask[None, :])


@triton.jit
def _sla_out_kernel(
    cq_ptr, al_ptr, ks_ptr, os_ptr, out_ptr,
    L, N_TILES_BH, N_PER_BH,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
    T: tl.constexpr, GRID,
):
    """Rowwise: out = o_s + (c_q @ A_l) / (1e-5 + sum_d c_q * ksum)."""
    for i in range(tl.program_id(0), N_TILES_BH, GRID):
        bh = i // N_PER_BH
        t0 = (i % N_PER_BH) * T
        rows = t0 + tl.arange(0, T)
        d = tl.arange(0, BLOCK_D)
        rmask = rows < L
        off = bh * L * D + rows[:, None] * D + d[None, :]
        mmask = rmask[:, None] & (d[None, :] < D)
        cq = tl.load(cq_ptr + off, mask=mmask, other=0.0)
        os_ = tl.load(os_ptr + off, mask=mmask, other=0.0)
        al = tl.load(al_ptr + bh * BLOCK_D * BLOCK_D + d[:, None] * BLOCK_D + d[None, :])
        ksum = tl.load(ks_ptr + bh * BLOCK_D + d, mask=(d < D), other=0.0)
        x = tl.dot(cq, al)
        denom = 1e-5 + tl.sum(cq * ksum[None, :], 1)
        res = os_ + x / denom[:, None]
        tl.store(out_ptr + off, res.to(out_ptr.dtype.element_ty),
                 mask=rmask[:, None] & (d[None, :] < D))


class ModelNew(nn.Module):
    """SLA forward on Ascend NPU via Triton kernels.

    Block selection is reproduced bit-exactly with torch (selection is
    rounding-sensitive and dominates correctness); the heavy compute
    (sparse attention, linear branch, fusion) runs in Triton kernels.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, q, k, v, topk, feature_map, BLKQ, BLKK, W_l):
        device = q.device
        B, H, L, D = q.shape
        assert k.shape == v.shape == q.shape, "SLA requires q/k/v same shape (MHA)"
        assert q.dtype == k.dtype == v.dtype
        assert isinstance(feature_map, str) and feature_map in ("elu", "relu", "softmax")
        assert 0.0 < topk <= 1.0

        dtype = q.dtype
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        W_l = W_l.contiguous()

        # ---- 1. block selection (bit-exact with reference) ----
        tok32, real_topk, MQ = _build_lut(q, k, topk, BLKQ, BLKK)

        FM = 0 if feature_map == "relu" else (1 if feature_map == "elu" else 2)
        BD = triton.next_power_of_2(D)
        BK = triton.next_power_of_2(BLKK)
        scale = D ** -0.5

        BH = B * H
        n_tiles = (L + 63) // 64
        cores = _num_cores()

        os_buf = torch.empty(B * H, L, D, dtype=torch.float32, device=device)
        cq = torch.empty(B * H, L, D, dtype=torch.float32, device=device)
        ck = torch.empty(B * H, L, D, dtype=torch.float32, device=device)
        al = torch.empty(B * H, BD, BD, dtype=torch.float32, device=device)
        ks = torch.empty(B * H, BD, dtype=torch.float32, device=device)
        out = torch.empty(B, H, L, D, dtype=dtype, device=device)

        # ---- 2. feature maps ----
        g_fm = BH * n_tiles
        if g_fm > cores:
            g_fm = cores
        _sla_fmap_kernel[(g_fm,)](
            q, cq, L, D, BD, 64, FM, BH * n_tiles, n_tiles, g_fm,
            num_warps=4,
        )
        _sla_fmap_kernel[(g_fm,)](
            k, ck, L, D, BD, 64, FM, BH * n_tiles, n_tiles, g_fm,
            num_warps=4,
        )

        # ---- 3. linear branch reduce (kvsum/ksum/Al per bh) ----
        g_kv = BH
        if g_kv > cores:
            g_kv = cores
        _sla_kv_kernel[(g_kv,)](
            ck, v, W_l, al, ks, L, n_tiles, D, BD, 64,
            num_warps=4,
        )

        # ---- 4. sparse attention ----
        # NSUB: split query rows into sub-tiles to bound UB usage
        NSUB = 2 if (BLKQ == 128 and BD == 128) else 1
        g_sp = BH * MQ
        if g_sp > cores:
            g_sp = cores
        _sla_sparse_kernel[(g_sp,)](
            q, k, v, tok32, os_buf, L, MQ, real_topk, BH,
            D, BD, BLKQ, BK, BLKK, scale * _LOG2E, NSUB, g_sp,
            num_warps=4,
        )

        # ---- 5. fusion ----
        g_out = BH * n_tiles
        if g_out > cores:
            g_out = cores
        _sla_out_kernel[(g_out,)](
            cq, al, ks, os_buf, out, L, BH * n_tiles, n_tiles, D, BD, 64, g_out,
            num_warps=4,
        )

        return out