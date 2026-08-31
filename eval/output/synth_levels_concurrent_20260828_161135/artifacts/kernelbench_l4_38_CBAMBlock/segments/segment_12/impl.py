import torch
import torch_npu
import torch.nn as nn
import triton
import triton.language as tl

try:
    torch.npu.conv.allow_hf32 = False
except Exception:
    pass


# ---------------------------------------------------------------------------
# Kernel 1: per (b, c-block): pool x to (avg, max) and form partial
#   part1[(blk, b, h)] = sum_{c in blk} W1[h, c] * avg[b, c]
#   part2[(blk, b, h)] = sum_{c in blk} W1[h, c] * maxp[b, c]
# Grid: B * nblk
@triton.jit
def cbam_pool_w1_kernel(
    x_ptr, w1_ptr, p1_ptr, p2_ptr,
    inv_hw, B, C, HW, hidden, nblk,
    BC: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    b = pid // nblk
    blk = pid - b * nblk
    ic = tl.arange(0, BC).to(tl.int32)
    ihw = tl.arange(0, BT).to(tl.int32)
    bh = tl.arange(0, BH).to(tl.int32)
    mh = bh.to(tl.float32) < hidden
    s = tl.zeros((BC,), dtype=tl.float32)
    m = tl.full((BC,), -3.0e30, dtype=tl.float32)
    for hw0 in range(0, HW, BT):
        offs = (b * C + blk * BC + ic[:, None]) * HW + (hw0 + ihw)[None, :]
        mask = (hw0 + ihw)[None, :].to(tl.float32) < HW
        t = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(t, axis=1)
        m = tl.maximum(m, tl.max(tl.where(mask, t, -3.0e30), axis=1))
    avg = s * inv_hw
    w1t = tl.load(w1_ptr + bh[:, None] * C + blk * BC + ic[None, :],
                  mask=mh[:, None], other=0.0).to(tl.float32)
    q1 = tl.sum(w1t * avg[None, :], axis=1)
    q2 = tl.sum(w1t * m[None, :], axis=1)
    po = (blk * B + b) * hidden + bh
    tl.store(p1_ptr + po, q1, mask=mh)
    tl.store(p2_ptr + po, q2, mask=mh)


# ---------------------------------------------------------------------------
# Kernel 2: second layer of the shared MLP (per b, c-block out):
#   hsum[b, h] = relu(sum_blk part1) + relu(sum_blk part2)
#   ca[b, c]   = sigmoid(sum_h W2[c, h] * hsum[b, h])
# Grid: B * nblk2
@triton.jit
def cbam_mlp2_kernel(
    p1_ptr, p2_ptr, w2_ptr, ca_ptr,
    B, C, hidden, nblk,
    BC2: tl.constexpr, BH: tl.constexpr, NB: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    nblk2 = C // BC2
    b = pid // nblk2
    blk2 = pid - b * nblk2
    iblk = tl.arange(0, NB).to(tl.int32)
    ib = tl.arange(0, BH).to(tl.int32)
    ic = tl.arange(0, BC2).to(tl.int32)
    mm = (iblk.to(tl.float32) < nblk)[:, None] & \
        (ib.to(tl.float32) < hidden)[None, :]
    po = (iblk[:, None] * B + b) * hidden + ib[None, :]
    q1 = tl.load(p1_ptr + po, mask=mm, other=0.0)
    q2 = tl.load(p2_ptr + po, mask=mm, other=0.0)
    hsum = tl.maximum(tl.sum(q1, axis=0), 0.0) + \
        tl.maximum(tl.sum(q2, axis=0), 0.0)
    mt = (ic[:, None].to(tl.float32) < C) & \
        (ib.to(tl.float32) < hidden)[None, :]
    w2t = tl.load(w2_ptr + (blk2 * BC2 + ic[:, None]) * hidden + ib[None, :],
                  mask=mt, other=0.0).to(tl.float32)
    s = tl.sum(w2t * hsum[None, :], axis=1)
    cav = 1.0 / (1.0 + tl.exp(-s))
    tl.store(ca_ptr + b * C + blk2 * BC2 + ic, cav,
             mask=ic.to(tl.float32) < C)


# ---------------------------------------------------------------------------
# Kernel 3: per (b, h) row: channel stats of ca-weighted features
#   st[b, 0, h, w] = max_c ca[b, c] * x[...],  st[b, 1, h, w] = mean_c
# Grid: B * H
@triton.jit
def cbam_stats_kernel(
    x_ptr, ca_ptr, st_ptr,
    inv_c, B, H, W, C,
    BC: tl.constexpr, WT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    b = pid // H
    h = pid - b * H
    iw = tl.arange(0, WT).to(tl.int32)
    ic = tl.arange(0, BC).to(tl.int32)
    mw = iw.to(tl.float32) < W
    mx = tl.full((WT,), -3.0e30, dtype=tl.float32)
    sm = tl.zeros((WT,), dtype=tl.float32)
    for c0 in range(0, C, BC):
        cm = (c0 + ic).to(tl.float32) < C
        cat = tl.load(ca_ptr + b * C + c0 + ic, mask=cm, other=0.0)
        offs = ((b * C + c0 + ic[:, None]) * H + h) * W + iw[None, :]
        om = cm[:, None] & mw[None, :]
        t = tl.load(x_ptr + offs, mask=om, other=0.0).to(tl.float32)
        t = t * cat[:, None]
        mx = tl.maximum(mx, tl.max(tl.where(om, t, -3.0e30), axis=0))
        sm += tl.sum(t, axis=0)
    base = (b * 2) * (H * W) + h * W + iw
    tl.store(st_ptr + base, mx, mask=mw)
    tl.store(st_ptr + base + H * W, sm * inv_c, mask=mw)


# ---------------------------------------------------------------------------
# Kernel 4a: fast spatial conv when every output pixel sees the full
# (H x W) st region and all sw taps used are in-bounds (K large).
#   out[b, h, w] = sum_{c2, j} st[b, c2, i, j] * sw[c2, i + pad - h + ...
# per (b, h): for c2 in {0,1}, for j in [0, W):
#     ST[ih]   = st[b, c2, ih, j]
#     S2[ih, iw] = sw[c2, pad - h + ih, j + pad - iw]
#     acc[iw] += sum_ih ST[ih] * S2[ih, iw]
# then sigmoid(acc + bias) -> sp[b, h, w]
# Grid: B * H
@triton.jit
def cbam_conv_fast_kernel(
    st_ptr, sw_ptr, sb_ptr, sp_ptr,
    B, H, W, K, pad,
    BH: tl.constexpr, BW: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    b = pid // H
    h = pid - b * H
    ih = tl.arange(0, BH).to(tl.int32)
    iw = tl.arange(0, BW).to(tl.int32)
    mrow = ih.to(tl.float32) < H
    mcol = iw.to(tl.float32) < W
    acc = tl.zeros((BW,), dtype=tl.float32)
    ky0 = pad - h
    for c2 in range(2):
        for j in range(W):
            stv = tl.load(st_ptr + ((b * 2 + c2) * H + ih) * W + j,
                          mask=mrow, other=0.0)
            woff = (c2 * K + ky0 + ih[:, None]) * K + (j + pad - iw[None, :])
            wst = tl.load(sw_ptr + woff,
                          mask=mrow[:, None] & mcol[None, :], other=0.0)
            acc += tl.sum(stv[:, None] * wst, axis=0)
    bias = tl.load(sb_ptr).to(tl.float32)
    sav = 1.0 / (1.0 + tl.exp(-(acc + bias)))
    tl.store(sp_ptr + (b * H + h) * W + iw, sav, mask=mcol)


# ---------------------------------------------------------------------------
# Kernel 4b: general spatial conv (small K): per (b, h) loop over (c2, ky)
#   acc[iw] += sum_ik wrow[c2, ky, ik] * st[b, c2, h + ky - pad, iw + ik - pad]
# Grid: B * H
@triton.jit
def cbam_conv_gen_kernel(
    st_ptr, sw_ptr, sb_ptr, sp_ptr,
    B, H, W, K, pad,
    BK: tl.constexpr, WT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    b = pid // H
    h = pid - b * H
    iw = tl.arange(0, WT).to(tl.int32)
    ik = tl.arange(0, BK).to(tl.int32)
    mcol = iw.to(tl.float32) < W
    mok = ik.to(tl.float32) < K
    acc = tl.zeros((WT,), dtype=tl.float32)
    for c2 in range(2):
        for ky in range(K):
            hi = h + ky - pad
            hiok = (hi >= 0) & (hi < H)
            hic = tl.where(hiok, hi, 0)
            wrow = tl.load(sw_ptr + (c2 * K + ky) * K + ik, mask=mok, other=0.0)
            jf = (iw[None, :] + ik[:, None] - pad)
            om = mok[:, None] & hiok & (jf >= 0) & (jf < W)
            t = tl.load(st_ptr + ((b * 2 + c2) * H + hic) * W +
                        iw[None, :] + ik[:, None] - pad,
                        mask=om, other=0.0)
            acc += tl.sum(wrow[:, None] * t, axis=0)
    bias = tl.load(sb_ptr).to(tl.float32)
    sav = 1.0 / (1.0 + tl.exp(-(acc + bias)))
    tl.store(sp_ptr + (b * H + h) * W + iw, sav, mask=mcol)


# ---------------------------------------------------------------------------
# Kernel 5: final elementwise  out = x * (ca * s + 1)
# Grid: min(B * ceil(HW/BT), VEC) interleaved
@triton.jit
def cbam_final_kernel(
    x_ptr, ca_ptr, sp_ptr, out_ptr,
    B, C, HW, nprogs,
    BC: tl.constexpr, BT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    ic = tl.arange(0, BC).to(tl.int32)
    ih = tl.arange(0, BT).to(tl.int32)
    nht = (HW + BT - 1) // BT
    total = B * nht
    for t in range(pid, total, nprogs):
        nb = t // nht
        b = nb
        hw0 = (t - nb * nht) * BT
        hwoff = hw0 + ih
        hm = hwoff.to(tl.float32) < HW
        s_t = tl.load(sp_ptr + b * HW + hwoff, mask=hm, other=0.0)
        for c0 in range(0, C, BC):
            cm = (c0 + ic).to(tl.float32) < C
            ca_t = tl.load(ca_ptr + b * C + c0 + ic, mask=cm, other=0.0)
            offs = (b * C + c0 + ic[:, None]) * HW + hwoff[None, :]
            msk = cm[:, None] & hm[None, :]
            x_t = tl.load(x_ptr + offs, mask=msk, other=0.0)
            y = x_t.to(tl.float32) * (ca_t[:, None] * s_t[None, :] + 1.0)
            tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=msk)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = int(
                torch_npu.npu.npu_config.get_device_limit(0).get(
                    "vector_core_num", 48))
        except Exception:
            self.VEC_CORE_NUM = 48

    def forward(self, x, channel, reduction, kernel_size,
                ca_fc1_weight, ca_fc2_weight, sa_conv_weight, sa_conv_bias):
        x = x.contiguous()
        B, C, H, W = x.shape
        HW = H * W
        hidden = C // int(reduction)
        hidden = hidden if hidden > 0 else 1
        K = int(kernel_size)
        pad = K // 2
        dev = x.device
        f32 = torch.float32

        w1 = ca_fc1_weight.contiguous().view(hidden, C)
        w2 = ca_fc2_weight.contiguous().view(C, hidden)
        sw = sa_conv_weight.contiguous().view(2, K, K)
        sb = sa_conv_bias.contiguous()

        BC1 = 32
        nblk1 = C // BC1
        BHh = triton.next_power_of_2(hidden)
        p1 = torch.empty((nblk1, B, hidden), device=dev, dtype=f32)
        p2 = torch.empty((nblk1, B, hidden), device=dev, dtype=f32)
        cbam_pool_w1_kernel[(B * nblk1,)](
            x, w1, p1, p2, 1.0 / HW, B, C, HW, hidden, nblk1,
            BC=BC1, BT=128, BH=BHh)

        BC2 = 64 if C >= 64 else 32
        nblk2 = C // BC2
        ca = torch.empty((B, C), device=dev, dtype=f32)
        cbam_mlp2_kernel[(B * nblk2,)](
            p1, p2, w2, ca, B, C, hidden, nblk1,
            BC2=BC2, BH=BHh, NB=triton.next_power_of_2(nblk1))

        WT = triton.next_power_of_2(W)
        st = torch.empty((B, 2, H, W), device=dev, dtype=f32)
        cbam_stats_kernel[(B * H,)](
            x, ca, st, 1.0 / C, B, H, W, C,
            BC=128, WT=WT)

        sp = torch.empty((B, H, W), device=dev, dtype=f32)
        if (pad >= H - 1) and (pad + H <= K) and \
                (pad >= W - 1) and (pad + W <= K):
            cbam_conv_fast_kernel[(B * H,)](
                st, sw, sb, sp, B, H, W, K, pad,
                BH=triton.next_power_of_2(H), BW=WT)
        else:
            cbam_conv_gen_kernel[(B * H,)](
                st, sw, sb, sp, B, H, W, K, pad,
                BK=triton.next_power_of_2(K), WT=WT)

        out = torch.empty_like(x)
        BT4 = 32
        nht = triton.cdiv(HW, BT4)
        total4 = B * nht
        g4 = total4 if total4 < self.VEC_CORE_NUM else self.VEC_CORE_NUM
        cbam_final_kernel[(g4,)](
            x, ca, sp, out, B, C, HW, g4,
            BC=128, BT=BT4)

        return out