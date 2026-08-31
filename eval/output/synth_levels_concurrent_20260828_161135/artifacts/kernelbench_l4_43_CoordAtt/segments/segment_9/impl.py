import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Coordinate Attention (KernelBench 43) fused Triton implementation for Ascend.
#
# Pipeline (mirrors the reference semantics exactly):
#   x_h  = avg over W   -> (B, C, H)
#   x_w  = avg over H   -> (B, C, W)
#   y    = 1x1conv(cat([x_h, x_w])) + BN(batch stats) + h_swish   (B, mip, H+W)
#   x_h', x_w' = split(y, [H, W])
#   a_h  = sigmoid(1x1conv(x_h'))   (B, C, H)
#   a_w  = sigmoid(1x1conv(x_w'))   (B, C, W)
#   out  = x * a_h[:, :, :, None] * a_w[:, :, None, :]
#
# Kernels:
#   k_pool     : spatial avg pooling -> pool (B, C, P), P = H + W
#   k_affine1  : GEMM yraw = W1 @ pool + b1 per (b, o-tile); fp32 acc;
#                atomic partial sums (sum / sumsq) for batch-norm stats
#   k_stats    : finalize mean / rstd per feature (biased variance, eps)
#   k_final    : fused tail: normalize + h_swish of yraw (in-kernel),
#                two small GEMMs + sigmoid for a_h / a_w, multiply with x
# ---------------------------------------------------------------------------


@triton.jit
def k_pool(x_ptr, pool_ptr, H, W, P, n_planes, n_prog,
           BH: tl.constexpr, BW: tl.constexpr):
    """Pure-vector plane pooling; grid capped at the vector-core count.

    Program pid handles the contiguous plane-id range [t0, t1) (contiguous
    task partition, no interleaving).  For each plane: load the (H, W)
    spatial plane (padded to BH x BW), sum along W and H (fp32 accumulation)
    and write pool[t, 0:H] and pool[t, H:P].
    """
    pid = tl.program_id(0)
    offs_i = tl.arange(0, BH)
    offs_j = tl.arange(0, BW)
    i_ok = offs_i < H
    j_ok = offs_j < W
    x_mask = i_ok[:, None] & j_ok[None, :]

    t0 = (n_planes * pid) // n_prog
    t1 = (n_planes * (pid + 1)) // n_prog
    for t in range(t0, t1):
        x_offs = t * (H * W) + offs_i[:, None] * W + offs_j[None, :]
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

        xh = tl.sum(x, axis=1) / W          # (BH,) mean over W
        xw = tl.sum(x, axis=0) / H          # (BW,) mean over H

        pool_base = t * P
        tl.store(pool_ptr + pool_base + offs_i, xh, mask=i_ok)
        tl.store(pool_ptr + pool_base + H + offs_j, xw, mask=j_ok)


@triton.jit
def k_affine1(pool_ptr, w1_ptr, b1_ptr, yr_ptr,
              C, P, mip, n_o_tiles, n_tasks, n_prog,
              BK: tl.constexpr, BP: tl.constexpr):
    """yraw[b, o, p] = sum_c W1[o, c] * pool[b, c, p] + b1[o].

    Tasks are (b, o-tile of 16) pairs; grid capped at the cube-core count.
    Program pid handles the contiguous task range [t0, t1).  GEMM tile
    (16, BP) x K-loop over C.
    """
    pid = tl.program_id(0)
    offs_p = tl.arange(0, BP)
    p_ok = offs_p < P
    offs_k = tl.arange(0, BK)

    t0 = (n_tasks * pid) // n_prog
    t1 = (n_tasks * (pid + 1)) // n_prog
    for t in range(t0, t1):
        b = t // n_o_tiles
        ot = t - (t // n_o_tiles) * n_o_tiles

        offs_o = ot * 16 + tl.arange(0, 16)
        o_ok = offs_o < mip

        acc = tl.zeros((16, BP), dtype=tl.float32)
        pool_base = pool_ptr + b * C * P
        for kc in range(0, C, BK):
            k_offs = kc + offs_k
            k_ok = k_offs < C
            a = tl.load(w1_ptr + offs_o[:, None] * C + k_offs[None, :],
                        mask=o_ok[:, None] & k_ok[None, :], other=0.0)
            bm = tl.load(pool_base + k_offs[:, None] * P + offs_p[None, :],
                         mask=k_ok[:, None] & p_ok[None, :], other=0.0)
            acc = tl.dot(a, bm, acc)

        bias = tl.load(b1_ptr + offs_o, mask=o_ok, other=0.0).to(tl.float32)
        acc = acc + bias[:, None]

        store_mask = o_ok[:, None] & p_ok[None, :]
        tl.store(yr_ptr + (b * mip + offs_o[:, None]) * P + offs_p[None, :],
                 acc, mask=store_mask)


@triton.jit
def k_stats(yr_ptr, mean_ptr, rstd_ptr, B, P, mip, n_total, n_prog,
            BM: tl.constexpr, BP: tl.constexpr, EPS: tl.constexpr):
    """Deterministic BN stats by reducing yraw (B, mip, P) directly.

    mean_o = sum_{b,p} y[b,o,p] / (B*P);  var_o = E[y^2] - mean^2 (biased);
    rstd = 1/sqrt(max(var,0)+EPS).  Grid capped at the vector-core count;
    program pid handles contiguous 16-feature tiles (no atomics, fully
    deterministic).
    """
    pid = tl.program_id(0)
    n_feats = (mip + BM - 1) // BM
    offs_p = tl.arange(0, BP)
    p_ok = offs_p < P

    t0 = (n_feats * pid) // n_prog
    t1 = (n_feats * (pid + 1)) // n_prog
    for t in range(t0, t1):
        offs_m = t * BM + tl.arange(0, BM)
        m_ok = offs_m < mip

        s = tl.zeros((BM,), dtype=tl.float32)
        sq = tl.zeros((BM,), dtype=tl.float32)
        for bb in range(0, B):
            y = tl.load(yr_ptr + (bb * mip + offs_m[:, None]) * P + offs_p[None, :],
                        mask=m_ok[:, None] & p_ok[None, :], other=0.0)
            y = y.to(tl.float32)
            s += tl.sum(y, axis=1)
            sq += tl.sum(y * y, axis=1)

        mean = s / n_total
        var = sq / n_total - mean * mean
        var = tl.maximum(var, 0.0)
        rstd = tl.rsqrt(var + EPS)

        tl.store(mean_ptr + offs_m, mean, mask=m_ok)
        tl.store(rstd_ptr + offs_m, rstd, mask=m_ok)


@triton.jit
def k_final(x_ptr, yr_ptr, mean_ptr, rstd_ptr,
            bnv_ptr, bnt_ptr, wh_ptr, bh_ptr, ww_ptr, bw_ptr, out_ptr,
            C, H, W, P, mip, n_ct, n_tasks, n_prog,
            BO: tl.constexpr, BK: tl.constexpr,
            N_I: tl.constexpr, N_J: tl.constexpr):
    """Fused tail with loop-invariant hoisting: y2 = hswish(BN(yraw));
    a_h = sigmoid(Wh @ y2_h + bh); a_w = sigmoid(Ww @ y2_w + bw);
    out = x * a_h * a_w.

    Tasks are (b, o-tile of BO) pairs; grid capped at the cube-core count.
    Program pid handles the contiguous task range [t0, t1).  The a_h / a_w
    tiles (N_I = cdiv(H, 16), N_J = cdiv(W, 16) 16-wide tiles, at most 2
    each for the covered shapes) are computed once per task up front; the
    following multiply/store loop is pure memory + VEC work.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BK)
    offs_i = tl.arange(0, 16)
    offs_j = tl.arange(0, 16)
    hw = H * W
    dt: tl.constexpr = yr_ptr.dtype.element_ty

    i0 = offs_i
    i0_ok = i0 < H
    i1 = 16 + offs_i
    i1_ok = i1 < H
    j0 = H + offs_j
    j0_ok = j0 < P
    j1 = H + 16 + offs_j
    j1_ok = j1 < P

    t0 = (n_tasks * pid) // n_prog
    t1 = (n_tasks * (pid + 1)) // n_prog
    for t in range(t0, t1):
        b = t // n_ct
        ct = t - (t // n_ct) * n_ct

        offs_oc = ct * BO + tl.arange(0, BO)
        oc_ok = offs_oc < C
        yr_base = yr_ptr + b * mip * P

        # ---- a_h tile 0 (y cols [0, 16)); tile 1 when H > 16 ----
        acc_h0 = tl.zeros((BO, 16), dtype=tl.float32)
        if N_I == 2:
            acc_h1 = tl.zeros((BO, 16), dtype=tl.float32)
        for km in range(0, mip, BK):
            m_offs = km + offs_m
            m_ok = m_offs < mip
            wh = tl.load(wh_ptr + offs_oc[:, None] * mip + m_offs[None, :],
                         mask=oc_ok[:, None] & m_ok[None, :], other=0.0)
            mean_m = tl.load(mean_ptr + m_offs, mask=m_ok, other=0.0)
            rstd_m = tl.load(rstd_ptr + m_offs, mask=m_ok, other=0.0)
            bnv = tl.load(bnv_ptr + m_offs, mask=m_ok, other=0.0).to(tl.float32)
            bnt = tl.load(bnt_ptr + m_offs, mask=m_ok, other=0.0).to(tl.float32)

            yh = tl.load(yr_base + m_offs[:, None] * P + i0[None, :],
                         mask=m_ok[:, None] & i0_ok[None, :], other=0.0)
            yf = yh.to(tl.float32)
            norm = (yf - mean_m[:, None]) * rstd_m[:, None]
            norm = norm * bnv[:, None] + bnt[:, None]          # affine BN
            hsw = norm * tl.clamp(norm + 3.0, 0.0, 6.0) / 6.0  # h_swish
            # re-round like the reference (h_swish output stored in input dtype)
            hsw = tl.where(m_ok[:, None], hsw.to(dt), 0.0)
            acc_h0 = tl.dot(wh, hsw, acc_h0)
            if N_I == 2:
                yh = tl.load(yr_base + m_offs[:, None] * P + i1[None, :],
                             mask=m_ok[:, None] & i1_ok[None, :], other=0.0)
                yf = yh.to(tl.float32)
                norm = (yf - mean_m[:, None]) * rstd_m[:, None]
                norm = norm * bnv[:, None] + bnt[:, None]
                hsw = norm * tl.clamp(norm + 3.0, 0.0, 6.0) / 6.0
                hsw = tl.where(m_ok[:, None], hsw.to(dt), 0.0)
                acc_h1 = tl.dot(wh, hsw, acc_h1)

        bhv = tl.load(bh_ptr + offs_oc, mask=oc_ok, other=0.0).to(tl.float32)
        # sigmoid (fp32) then re-round to storage dtype, then back to fp32
        a_h0 = tl.sigmoid(acc_h0 + bhv[:, None]).to(dt).to(tl.float32)
        if N_I == 2:
            a_h1 = tl.sigmoid(acc_h1 + bhv[:, None]).to(dt).to(tl.float32)

        # ---- a_w tile 0 (y cols [H, H+16)); tile 1 when W > 16 ----
        acc_w0 = tl.zeros((BO, 16), dtype=tl.float32)
        if N_J == 2:
            acc_w1 = tl.zeros((BO, 16), dtype=tl.float32)
        for km in range(0, mip, BK):
            m_offs = km + offs_m
            m_ok = m_offs < mip
            ww = tl.load(ww_ptr + offs_oc[:, None] * mip + m_offs[None, :],
                         mask=oc_ok[:, None] & m_ok[None, :], other=0.0)
            mean_m = tl.load(mean_ptr + m_offs, mask=m_ok, other=0.0)
            rstd_m = tl.load(rstd_ptr + m_offs, mask=m_ok, other=0.0)
            bnv = tl.load(bnv_ptr + m_offs, mask=m_ok, other=0.0).to(tl.float32)
            bnt = tl.load(bnt_ptr + m_offs, mask=m_ok, other=0.0).to(tl.float32)

            yw = tl.load(yr_base + m_offs[:, None] * P + j0[None, :],
                         mask=m_ok[:, None] & j0_ok[None, :], other=0.0)
            yf = yw.to(tl.float32)
            norm = (yf - mean_m[:, None]) * rstd_m[:, None]
            norm = norm * bnv[:, None] + bnt[:, None]
            hsw = norm * tl.clamp(norm + 3.0, 0.0, 6.0) / 6.0
            hsw = tl.where(m_ok[:, None], hsw.to(dt), 0.0)
            acc_w0 = tl.dot(ww, hsw, acc_w0)
            if N_J == 2:
                yw = tl.load(yr_base + m_offs[:, None] * P + j1[None, :],
                             mask=m_ok[:, None] & j1_ok[None, :], other=0.0)
                yf = yw.to(tl.float32)
                norm = (yf - mean_m[:, None]) * rstd_m[:, None]
                norm = norm * bnv[:, None] + bnt[:, None]
                hsw = norm * tl.clamp(norm + 3.0, 0.0, 6.0) / 6.0
                hsw = tl.where(m_ok[:, None], hsw.to(dt), 0.0)
                acc_w1 = tl.dot(ww, hsw, acc_w1)

        bwv = tl.load(bw_ptr + offs_oc, mask=oc_ok, other=0.0).to(tl.float32)
        a_w0 = tl.sigmoid(acc_w0 + bwv[:, None]).to(dt).to(tl.float32)
        if N_J == 2:
            a_w1 = tl.sigmoid(acc_w1 + bwv[:, None]).to(dt).to(tl.float32)

        # ---- multiply & store: up to 4 (16 x 16) spatial tiles ----
        x_base = x_ptr + (b * C + offs_oc[:, None, None]) * hw
        o_base = out_ptr + (b * C + offs_oc[:, None, None]) * hw

        x_offs = x_base + i0[None, :, None] * W + j0[None, None, :]
        x_mask = oc_ok[:, None, None] & i0_ok[None, :, None] & j0_ok[None, None, :]
        xv = tl.load(x_offs, mask=x_mask, other=0.0).to(tl.float32)
        tl.store(o_base + i0[None, :, None] * W + j0[None, None, :],
                 xv * a_h0[:, :, None] * a_w0[:, None, :], mask=x_mask)

        if N_J == 2:
            x_offs = x_base + i0[None, :, None] * W + j1[None, None, :]
            x_mask = oc_ok[:, None, None] & i0_ok[None, :, None] & j1_ok[None, None, :]
            xv = tl.load(x_offs, mask=x_mask, other=0.0).to(tl.float32)
            tl.store(o_base + i0[None, :, None] * W + j1[None, None, :],
                     xv * a_h0[:, :, None] * a_w1[:, None, :], mask=x_mask)

        if N_I == 2:
            x_offs = x_base + i1[None, :, None] * W + j0[None, None, :]
            x_mask = oc_ok[:, None, None] & i1_ok[None, :, None] & j0_ok[None, None, :]
            xv = tl.load(x_offs, mask=x_mask, other=0.0).to(tl.float32)
            tl.store(o_base + i1[None, :, None] * W + j0[None, None, :],
                     xv * a_h1[:, :, None] * a_w0[:, None, :], mask=x_mask)
            if N_J == 2:
                x_offs = x_base + i1[None, :, None] * W + j1[None, None, :]
                x_mask = oc_ok[:, None, None] & i1_ok[None, :, None] & j1_ok[None, None, :]
                xv = tl.load(x_offs, mask=x_mask, other=0.0).to(tl.float32)
                tl.store(o_base + i1[None, :, None] * W + j1[None, None, :],
                         xv * a_h1[:, :, None] * a_w1[:, None, :], mask=x_mask)


# ---------------------------------------------------------------------------
# Torch fallback for shapes outside the fused fast path (safety net; the
# benchmark shapes are all covered by the Triton kernels).
# ---------------------------------------------------------------------------

def _conv1x1_ref(x, weight, bias=None):
    orig_dtype = x.dtype
    x = x.float()
    weight = weight.squeeze(-1).squeeze(-1).float()
    out = torch.einsum('oc,bchw->bohw', weight, x)
    if bias is not None:
        out = out + bias.float().view(1, -1, 1, 1)
    return out.to(orig_dtype)


def _coordatt_torch(x, reduction, conv1_w, conv1_b, bn_w, bn_b,
                    conv_h_w, conv_h_b, conv_w_w, conv_w_b):
    import torch.nn.functional as F
    identity = x
    _, c, h, w = x.shape
    mip = max(8, c // int(reduction))
    x_h = F.adaptive_avg_pool2d(x, (None, 1))
    x_w = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
    y = torch.cat([x_h, x_w], dim=2)
    y = _conv1x1_ref(y, conv1_w, conv1_b)
    y = F.batch_norm(y, torch.zeros_like(bn_w), torch.ones_like(bn_w),
                     bn_w, bn_b, training=True, momentum=0.1, eps=1e-5)
    y = y * (F.relu6(y + 3) / 6)
    x_h, x_w = torch.split(y, [h, w], dim=2)
    x_w = x_w.permute(0, 1, 3, 2)
    a_h = torch.sigmoid(_conv1x1_ref(x_h, conv_h_w, conv_h_b))
    a_w = torch.sigmoid(_conv1x1_ref(x_w, conv_w_w, conv_w_b))
    return identity * a_w * a_h


def _next_pow2(n):
    p = 16
    while p < n:
        p *= 2
    return p


_CORE_COUNTS = None


def _core_counts():
    """Dynamically read (vector_cores, cube_cores); never hardcode counts."""
    global _CORE_COUNTS
    if _CORE_COUNTS is None:
        try:
            import torch_npu
            dev_limit = torch_npu.npu.npu_config.get_device_limit(0)
            vec = int(dev_limit.get('vector_core_num', 40))
            cube = int(dev_limit.get('cube_core_num', 40))
            if vec > 0 and cube > 0:
                _CORE_COUNTS = (vec, cube)
            else:
                _CORE_COUNTS = (40, 40)
        except Exception:
            _CORE_COUNTS = (40, 40)
    return _CORE_COUNTS


class ModelNew(nn.Module):
    """Coordinate Attention with the same interface as the reference Model."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, inp, oup, reduction,
                conv1_w, conv1_b, bn_w, bn_b,
                conv_h_w, conv_h_b, conv_w_w, conv_w_b):
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return _coordatt_torch(x, reduction, conv1_w, conv1_b, bn_w, bn_b,
                                   conv_h_w, conv_h_b, conv_w_w, conv_w_b)

        b, c, h, w = x.shape
        reduction = int(reduction)
        mip = c // reduction
        if mip < 8:
            mip = 8
        P = h + w

        # fast-path shape limits (all benchmark cases fit: H,W <= 23)
        if h > 32 or w > 32 or b * c > 60000 or mip > 4096 or b * P > 4096:
            return _coordatt_torch(x, reduction, conv1_w, conv1_b, bn_w, bn_b,
                                   conv_h_w, conv_h_b, conv_w_w, conv_w_b)

        x = x.contiguous()
        dev = x.device
        dt = x.dtype

        pool = torch.empty((b, c, P), device=dev, dtype=dt)
        yraw = torch.empty((b, mip, P), device=dev, dtype=dt)
        mean = torch.empty((mip,), device=dev, dtype=torch.float32)
        rstd = torch.empty((mip,), device=dev, dtype=torch.float32)

        # grids capped at the physical core count; in-kernel contiguous
        # task loops (no interleaving) cover any remaining tasks
        vec_cores, cube_cores = _core_counts()

        # ---- kernel 1: pooling ------------------------------------------------
        BH = _next_pow2(h)
        BW = _next_pow2(w)
        n_planes = b * c
        n_prog_pool = n_planes if n_planes <= vec_cores else vec_cores
        k_pool[(n_prog_pool,)](x, pool, h, w, P, n_planes, n_prog_pool,
                               BH=BH, BW=BW)

        # ---- kernel 2: 1x1 conv (mip x C) @ pool (C x P) + bias -------------
        BP = _next_pow2(P)
        n_o_tiles = triton.cdiv(mip, 16)
        n_task_a1 = b * n_o_tiles
        n_prog_a1 = n_task_a1 if n_task_a1 <= cube_cores else cube_cores
        k_affine1[(n_prog_a1,)](
            pool, conv1_w, conv1_b, yraw,
            c, P, mip, n_o_tiles, n_task_a1, n_prog_a1,
            BK=64, BP=BP)

        # ---- kernel 3: batch-norm stats (deterministic, no atomics) ----------
        n_st = triton.cdiv(mip, 16)
        n_prog_st = n_st if n_st <= vec_cores else vec_cores
        k_stats[(n_prog_st,)](yraw, mean, rstd, b, P, mip, b * P, n_prog_st,
                              BM=16, BP=BP, EPS=1e-5)

        # ---- kernel 4: fused tail (BN+hswish on the fly, 2 GEMMs, sigmoid,
        #                 multiply with x) -------------------------------------
        out = torch.empty_like(x)
        n_ct = triton.cdiv(c, 32)
        n_task_f = b * n_ct
        n_prog_f = n_task_f if n_task_f <= cube_cores else cube_cores
        n_i_f = triton.cdiv(h, 16)   # 16-wide h tiles, at most 2 (h <= 32)
        n_j_f = triton.cdiv(w, 16)   # 16-wide w tiles, at most 2 (w <= 32)
        k_final[(n_prog_f,)](
            x, yraw, mean, rstd,
            bn_w, bn_b, conv_h_w, conv_h_b, conv_w_w, conv_w_b, out,
            c, h, w, P, mip, n_ct, n_task_f, n_prog_f,
            BO=32, BK=64, N_I=n_i_f, N_J=n_j_f)

        return out


