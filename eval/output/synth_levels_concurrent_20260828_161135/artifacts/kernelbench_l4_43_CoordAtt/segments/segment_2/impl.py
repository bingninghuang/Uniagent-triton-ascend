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
def k_pool(x_ptr, pool_ptr, H, W, P,
           BH: tl.constexpr, BW: tl.constexpr):
    """One program per (b, c) pair.

    Loads the (H, W) spatial plane (padded to BH x BW), sums along W and H
    (fp32 accumulation) and writes pool[b, c, 0:H] and pool[b, c, H:H+W].
    """
    pid = tl.program_id(0)
    offs_i = tl.arange(0, BH)
    offs_j = tl.arange(0, BW)
    i_ok = offs_i < H
    j_ok = offs_j < W

    x_offs = pid * (H * W) + offs_i[:, None] * W + offs_j[None, :]
    x_mask = i_ok[:, None] & j_ok[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

    xh = tl.sum(x, axis=1) / W          # (BH,) mean over W
    xw = tl.sum(x, axis=0) / H          # (BW,) mean over H

    pool_base = pid * P
    tl.store(pool_ptr + pool_base + offs_i, xh, mask=i_ok)
    tl.store(pool_ptr + pool_base + H + offs_j, xw, mask=j_ok)


@triton.jit
def k_affine1(pool_ptr, w1_ptr, b1_ptr, yr_ptr, stat_ptr,
              C, P, mip, n_o_tiles,
              BK: tl.constexpr, BP: tl.constexpr):
    """yraw[b, o, p] = sum_c W1[o, c] * pool[b, c, p] + b1[o].

    Program layout: (b, o-tile of 16).  GEMM tile (16, BP) x K-loop over C.
    Also accumulates per-feature partial sum / sumsq (over p) of the fp32
    affine output via atomic_add into stat_ptr[0:mip] / stat_ptr[mip:2*mip].
    """
    pid = tl.program_id(0)
    b = pid // n_o_tiles
    ot = pid % n_o_tiles

    offs_o = ot * 16 + tl.arange(0, 16)
    o_ok = offs_o < mip
    offs_p = tl.arange(0, BP)
    p_ok = offs_p < P
    offs_k = tl.arange(0, BK)

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

    accm = tl.where(p_ok[None, :], acc, 0.0)
    s = tl.sum(accm, axis=1)            # (16,) partial sum over p for this b
    sq = tl.sum(accm * accm, axis=1)
    tl.atomic_add(stat_ptr + offs_o, s, mask=o_ok)
    tl.atomic_add(stat_ptr + mip + offs_o, sq, mask=o_ok)


@triton.jit
def k_stats(stat_ptr, mean_ptr, rstd_ptr, mip, n_total, eps):
    """mean_o = S[N]/N ;  var_o = S[N^2]/N - mean^2 (biased); rstd = 1/sqrt(var+eps).

    n_total = B * P (number of values each feature is reduced over).
    """
    ot = tl.program_id(0)
    offs_o = ot * 16 + tl.arange(0, 16)
    o_ok = offs_o < mip

    s = tl.load(stat_ptr + offs_o, mask=o_ok, other=0.0)
    sq = tl.load(stat_ptr + mip + offs_o, mask=o_ok, other=0.0)

    mean = s / n_total
    var = sq / n_total - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    tl.store(mean_ptr + offs_o, mean, mask=o_ok)
    tl.store(rstd_ptr + offs_o, rstd, mask=o_ok)


@triton.jit
def k_final(x_ptr, yr_ptr, mean_ptr, rstd_ptr,
            bnv_ptr, bnt_ptr, wh_ptr, bh_ptr, ww_ptr, bw_ptr, out_ptr,
            C, H, W, P, mip, n_ct,
            BO: tl.constexpr, BK: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    """Fused tail: y2 = hswish(BN(yraw)); a_h = sigmoid(Wh @ y2_h + bh);
    a_w = sigmoid(Ww @ y2_w + bw);  out = x * a_h * a_w.

    Program layout: (b, o-tile of BO).  Inside, tiles over the spatial grid
    (BI x BJ) recomputing the small mip-reductions on the fly.
    """
    pid = tl.program_id(0)
    b = pid // n_ct
    ct = pid % n_ct

    offs_oc = ct * BO + tl.arange(0, BO)
    oc_ok = offs_oc < C
    offs_m = tl.arange(0, BK)
    offs_bi = tl.arange(0, BI)
    offs_bj = tl.arange(0, BJ)

    n_i = (H + BI - 1) // BI
    n_j = (W + BJ - 1) // BJ

    yr_base = yr_ptr + b * mip * P
    hw = H * W

    for si in range(0, n_i):
        i_offs = si * BI + offs_bi
        i_ok = i_offs < H

        acc_h = tl.zeros((BO, BI), dtype=tl.float32)
        for km in range(0, mip, BK):
            m_offs = km + offs_m
            m_ok = m_offs < mip
            wh = tl.load(wh_ptr + offs_oc[:, None] * mip + m_offs[None, :],
                         mask=oc_ok[:, None] & m_ok[None, :], other=0.0)
            yh = tl.load(yr_base + m_offs[:, None] * P + i_offs[None, :],
                         mask=m_ok[:, None] & i_ok[None, :], other=0.0)
            mean_m = tl.load(mean_ptr + m_offs, mask=m_ok, other=0.0)
            rstd_m = tl.load(rstd_ptr + m_offs, mask=m_ok, other=0.0)
            bnv = tl.load(bnv_ptr + m_offs, mask=m_ok, other=0.0).to(tl.float32)
            bnt = tl.load(bnt_ptr + m_offs, mask=m_ok, other=0.0).to(tl.float32)

            yf = yh.to(tl.float32)
            norm = (yf - mean_m[:, None]) * rstd_m[:, None]
            norm = norm * bnv[:, None] + bnt[:, None]          # affine BN
            hsw = norm * tl.clamp(norm + 3.0, 0.0, 6.0) / 6.0  # h_swish
            # re-round like the reference (h_swish output stored in input dtype)
            hsw = hsw.to(yr_ptr.dtype.element_ty)
            hsw = tl.where(m_ok[:, None], hsw, 0.0)
            acc_h = tl.dot(wh, hsw, acc_h)

        bhv = tl.load(bh_ptr + offs_oc, mask=oc_ok, other=0.0).to(tl.float32)
        # sigmoid (fp32) then re-round to storage dtype, then back to fp32
        a_h = tl.sigmoid(acc_h + bhv[:, None])
        a_h = a_h.to(yr_ptr.dtype.element_ty).to(tl.float32)

        for sj in range(0, n_j):
            j_offs = sj * BJ + offs_bj
            j_ok = j_offs < W

            acc_w = tl.zeros((BO, BJ), dtype=tl.float32)
            for km in range(0, mip, BK):
                m_offs = km + offs_m
                m_ok = m_offs < mip
                ww = tl.load(ww_ptr + offs_oc[:, None] * mip + m_offs[None, :],
                             mask=oc_ok[:, None] & m_ok[None, :], other=0.0)
                yw = tl.load(yr_base + m_offs[:, None] * P + H + j_offs[None, :],
                             mask=m_ok[:, None] & j_ok[None, :], other=0.0)
                mean_m = tl.load(mean_ptr + m_offs, mask=m_ok, other=0.0)
                rstd_m = tl.load(rstd_ptr + m_offs, mask=m_ok, other=0.0)
                bnv = tl.load(bnv_ptr + m_offs, mask=m_ok, other=0.0).to(tl.float32)
                bnt = tl.load(bnt_ptr + m_offs, mask=m_ok, other=0.0).to(tl.float32)

                yf = yw.to(tl.float32)
                norm = (yf - mean_m[:, None]) * rstd_m[:, None]
                norm = norm * bnv[:, None] + bnt[:, None]
                hsw = norm * tl.clamp(norm + 3.0, 0.0, 6.0) / 6.0
                hsw = hsw.to(yr_ptr.dtype.element_ty)
                hsw = tl.where(m_ok[:, None], hsw, 0.0)
                acc_w = tl.dot(ww, hsw, acc_w)

            bwv = tl.load(bw_ptr + offs_oc, mask=oc_ok, other=0.0).to(tl.float32)
            a_w = tl.sigmoid(acc_w + bwv[:, None])
            a_w = a_w.to(yr_ptr.dtype.element_ty).to(tl.float32)

            # spatial tile of x : (BO, BI, BJ)
            x_offs = (b * C + offs_oc[:, None, None]) * hw \
                + i_offs[None, :, None] * W + j_offs[None, None, :]
            x_mask = oc_ok[:, None, None] & i_ok[None, :, None] & j_ok[None, None, :]
            xv = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)
            out = xv * a_h[:, :, None] * a_w[:, None, :]
            tl.store(out_ptr + x_offs, out, mask=x_mask)


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
        stat = torch.zeros((2 * mip,), device=dev, dtype=torch.float32)
        mean = torch.empty((mip,), device=dev, dtype=torch.float32)
        rstd = torch.empty((mip,), device=dev, dtype=torch.float32)

        # ---- kernel 1: pooling ------------------------------------------------
        BH = _next_pow2(h)
        BW = _next_pow2(w)
        k_pool[(b * c,)](x, pool, h, w, P, BH=BH, BW=BW)

        # ---- kernel 2: 1x1 conv (mip x C) @ pool (C x P) + bias --- partial sums
        BP = _next_pow2(P)
        n_o_tiles = triton.cdiv(mip, 16)
        k_affine1[(b * n_o_tiles,)](
            pool, conv1_w, conv1_b, yraw, stat,
            c, P, mip, n_o_tiles,
            BK=64, BP=BP)

        # ---- kernel 3: batch-norm stats (biased variance) -------------------
        k_stats[(n_o_tiles,)](stat, mean, rstd, mip, b * P, 1e-5)

        # ---- kernel 4: fused tail (BN+hswish on the fly, 2 GEMMs, sigmoid,
        #                 multiply with x) -------------------------------------
        out = torch.empty_like(x)
        n_ct = triton.cdiv(c, 32)
        k_final[(b * n_ct,)](
            x, yraw, mean, rstd,
            bn_w, bn_b, conv_h_w, conv_h_b, conv_w_w, conv_w_b, out,
            c, h, w, P, mip, n_ct,
            BO=32, BK=64, BI=16, BJ=16)

        if not getattr(self, '_dbg_done', False):
            self._dbg_done = True
            _debug_dump(x, inp, oup, reduction,
                        conv1_w, conv1_b, bn_w, bn_b,
                        conv_h_w, conv_h_b, conv_w_w, conv_w_b,
                        pool, yraw, mean, rstd, out)
        return out


def _debug_dump(x, inp, oup, reduction,
                conv1_w, conv1_b, bn_w, bn_b,
                conv_h_w, conv_h_b, conv_w_w, conv_w_b,
                pool, yraw, mean, rstd, out):
    import traceback
    import torch.nn.functional as F
    lines = []
    try:
        b, c, h, w = x.shape
        xf = x.float()
        x_h = F.adaptive_avg_pool2d(xf, (None, 1)).flatten(3)
        x_w = F.adaptive_avg_pool2d(xf, (1, None)).permute(0, 1, 3, 2).flatten(3)
        P = h + w
        pool_ref = torch.cat([x_h, x_w], dim=2)
        mip = max(8, c // int(reduction))
        W1 = conv1_w.squeeze(-1).squeeze(-1).float()
        yraw_ref = (torch.einsum('oc,bchw->bohw', W1, pool_ref.unsqueeze(3))
                    + conv1_b.float().view(1, -1, 1, 1))[..., 0]
        N = b * P
        y3 = yraw_ref.reshape(b, mip, N // b)
        mean_ref = y3.mean(dim=(0, 2))
        var_ref = y3.var(dim=(0, 2), unbiased=False)
        rstd_ref = 1.0 / torch.sqrt(var_ref + 1e-5)
        ynorm = (y3 - mean_ref.view(1, -1, 1)) * rstd_ref.view(1, -1, 1) \
                * bn_w.float().view(1, -1, 1) + bn_b.float().view(1, -1, 1)
        yhs = ynorm * F.relu6(ynorm + 3.0) / 6.0
        Wh = conv_h_w.squeeze(-1).squeeze(-1).float()
        Ww = conv_w_w.squeeze(-1).squeeze(-1).float()
        a_h_ref = torch.sigmoid(yhs[:, :, :h].transpose(1, 2) @ Wh.transpose(0, 1)
                                + conv_h_b.float().view(-1, 1)).transpose(1, 2)
        a_w_ref = torch.sigmoid(yhs[:, :, h:h + w].transpose(1, 2) @ Ww.transpose(0, 1)
                                + conv_w_b.float().view(-1, 1)).transpose(1, 2)
        out_ref = xf * a_h_ref.unsqueeze(3) * a_w_ref.unsqueeze(2)

        from kernelbench_l4_43_CoordAtt import Model as _RefModel
        ref_out = _RefModel().forward(x, inp, oup, reduction,
                                      conv1_w, conv1_b, bn_w, bn_b,
                                      conv_h_w, conv_h_b, conv_w_w, conv_w_b)

        def line(tag, r, m):
            d = (m.float() - r).abs()
            lines.append("%s: maxdiff=%.6g mine[0]=%.6g ref[0]=%.6g"
                         % (tag, d.max().item(), m.flatten()[0].item(),
                            r.flatten()[0].item()))

        line('POOL', pool_ref, pool)
        line('YRAW', yraw_ref, yraw)
        line('MEAN', mean_ref, mean)
        line('RSTD', rstd_ref, rstd)

        d = (out.float() - out_ref).abs()
        lines.append('OUT manref: maxdiff=%.6g meandiff=%.6g' %
                     (d.max().item(), d.mean().item()))
        d2 = (out.float() - ref_out).abs()
        lines.append('OUT npuref: maxdiff=%.6g meandiff=%.6g' %
                     (d2.max().item(), d2.mean().item()))
        i = d2.flatten().argmax().item()
        lines.append('worst: ref=%.6g mine=%.6g idx=%d' %
                     (ref_out.flatten()[i].item(), out.flatten()[i].item(), i))

        # a_h / a_w recomputed from MY yraw with ref stats (isolate k_final)
        y3_mine = yraw.float().reshape(b, mip, N // b)
        ynorm_m = (y3_mine - mean_ref.view(1, -1, 1)) * rstd_ref.view(1, -1, 1) \
                  * bn_w.float().view(1, -1, 1) + bn_b.float().view(1, -1, 1)
        yhs_m = ynorm_m * F.relu6(ynorm_m + 3.0) / 6.0
        a_h_m = torch.sigmoid(yhs_m[:, :, :h].transpose(1, 2) @ Wh.transpose(0, 1)
                              + conv_h_b.float().view(-1, 1)).transpose(1, 2)
        a_w_m = torch.sigmoid(yhs_m[:, :, h:h + w].transpose(1, 2) @ Ww.transpose(0, 1)
                              + conv_w_b.float().view(-1, 1)).transpose(1, 2)
        line('A_H (ref vs from-my-yraw)', a_h_ref, a_h_m)
        line('A_W (ref vs from-my-yraw)', a_w_ref, a_w_m)
    except Exception:
        lines.append('DBG EXC:\n' + traceback.format_exc())
    with open('/opt/workspace_card2/agent_workdir/debug_dump.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')