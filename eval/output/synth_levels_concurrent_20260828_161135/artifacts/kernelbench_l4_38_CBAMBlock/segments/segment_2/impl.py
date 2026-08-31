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
# Kernel 1: per-batch channel attention:
#   - pool x[b] to (C,) mean / max over (H*W)
#   - shared MLP: hsum = relu(W1^T avg) + relu(W1^T max)  (per b)
#   - ca = sigmoid(W2^T hsum)
# Grid: (B,)
@triton.jit
def cbam_channel_attn_kernel(
    x_ptr, w1_ptr, w2_ptr, ca_ptr,
    C, HW, hidden, inv_hw,
    CC: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int32)
    ic = tl.arange(0, CC).to(tl.int32)
    ih = tl.arange(0, BT).to(tl.int32)
    ib = tl.arange(0, BH).to(tl.int32)

    h1 = tl.zeros((BH,), dtype=tl.float32)
    h2 = tl.zeros((BH,), dtype=tl.float32)
    for cc0 in range(0, C, CC):
        s_acc = tl.zeros((CC,), dtype=tl.float32)
        m_acc = tl.full((CC,), -3.0e30, dtype=tl.float32)
        for hw0 in range(0, HW, BT):
            offs = (b * C + cc0 + ic[:, None]) * HW + (hw0 + ih)[None, :]
            mask = (hw0 + ih)[None, :] < HW
            tile = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            s_acc += tl.sum(tile, axis=1)
            mx = tl.max(tl.where(mask, tile, -3.0e30), axis=1)
            m_acc = tl.maximum(m_acc, mx)
        avg_c = s_acc * inv_hw
        max_c = m_acc
        w1offs = ib[:, None] * C + (cc0 + ic)[None, :]
        w1mask = ib[:, None] < hidden
        w1t = tl.load(w1_ptr + w1offs, mask=w1mask, other=0.0).to(tl.float32)
        h1 += tl.sum(w1t * avg_c[None, :], axis=1)
        h2 += tl.sum(w1t * max_c[None, :], axis=1)
    hsum = tl.maximum(h1, 0.0) + tl.maximum(h2, 0.0)

    for c0 in range(0, C, CC):
        w2offs = (c0 + ic[:, None]) * hidden + ib[None, :]
        w2mask = ib[None, :] < hidden
        w2t = tl.load(w2_ptr + w2offs, mask=w2mask, other=0.0).to(tl.float32)
        s_c = tl.sum(w2t * hsum[None, :], axis=1)
        ca_c = 1.0 / (1.0 + tl.exp(-s_c))
        tl.store(ca_ptr + b * C + c0 + ic, ca_c)


# ---------------------------------------------------------------------------
# Kernel 2: per-pixel channel stats of weighted features
#   mx[b, hw] = max_c (ca[b,c] * x[b,c,hw]),  av[b, hw] = mean_c (...)
#   stored as st (B, 2, H, W): channel 0 = max, channel 1 = avg
# Grid: min(B*H, VEC) interleaved over (b, h) rows
@triton.jit
def cbam_spatial_stats_kernel(
    x_ptr, ca_ptr, st_ptr,
    B, H, W, HW, C, inv_c, nprogs,
    BC: tl.constexpr, WT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    iw = tl.arange(0, WT).to(tl.int32)
    ic = tl.arange(0, BC).to(tl.int32)
    for t in range(pid, B * H, nprogs):
        b = (t // H).to(tl.int32)
        h = (t % H).to(tl.int32)
        mx = tl.full((WT,), -3.0e30, dtype=tl.float32)
        sm = tl.zeros((WT,), dtype=tl.float32)
        for c0 in range(0, C, BC):
            cm = (c0 + ic) < C
            ca_t = tl.load(ca_ptr + b * C + c0 + ic, mask=cm, other=0.0)
            offs = ((b * C + c0 + ic[:, None]) * H + h) * W + iw[None, :]
            om = cm[:, None] & (iw[None, :] < W)
            wt = tl.load(x_ptr + offs, mask=om, other=0.0).to(tl.float32)
            wt = wt * ca_t[:, None]
            mx = tl.maximum(mx, tl.max(tl.where(cm[:, None], wt, -3.0e30), axis=0))
            sm += tl.sum(wt, axis=0)
        base = b * (2 * HW) + h * W
        tl.store(st_ptr + base + iw, mx, mask=iw < W)
        tl.store(st_ptr + base + HW + iw, sm * inv_c, mask=iw < W)


# ---------------------------------------------------------------------------
# Kernel 3: spatial attention conv  ch 2 -> 1, kernel KxK, pad = K//2
#   s[b, hw] = sigmoid(bias + sum sw[c2,ky,kx] * st[b, c2, h+ky-pad, w+kx-pad])
# Grid: min(B*H, VEC) interleaved over (b, h) rows
@triton.jit
def cbam_spatial_conv_kernel(
    st_ptr, sw_ptr, sb_ptr, sp_ptr,
    B, H, W, HW, K, pad, nprogs,
    BK: tl.constexpr, WT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    iw = tl.arange(0, WT).to(tl.int32)
    ik = tl.arange(0, BK).to(tl.int32)
    for t in range(pid, B * H, nprogs):
        b = (t // H).to(tl.int32)
        h = (t % H).to(tl.int32)
        acc = tl.zeros((WT,), dtype=tl.float32)
        for c2 in range(2):
            for ky in range(0, K):
                hi = h - pad + ky
                rowok = (hi >= 0) & (hi < H)
                hi_c = tl.where(rowok, hi, 0)
                base = ((b * 2 + c2) * H + hi_c) * W
                wrow = tl.load(
                    sw_ptr + c2 * K * K + ky * K + ik,
                    mask=(ik < K) & rowok, other=0.0).to(tl.float32)
                joff = ik[:, None] + (iw - pad)[None, :]
                m2 = (ik[:, None] < K) & (iw[None, :] < W) & rowok & \
                    (joff >= 0) & (joff < W)
                in_t = tl.load(st_ptr + base + joff, mask=m2, other=0.0)
                acc += tl.sum(wrow[:, None] * in_t, axis=0)
        bias = tl.load(sb_ptr)
        s = 1.0 / (1.0 + tl.exp(-(acc + bias)))
        tl.store(sp_ptr + b * HW + h * W + iw, s, mask=iw < W)


# ---------------------------------------------------------------------------
# Kernel 4: final elementwise  out = x * (ca * s + 1)
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
        b = (t // nht).to(tl.int32)
        hw0 = ((t % nht) * BT).to(tl.int32)
        hwoff = hw0 + ih
        hm = hwoff < HW
        s_t = tl.load(sp_ptr + b * HW + hwoff, mask=hm, other=0.0)
        for c0 in range(0, C, BC):
            cm = (c0 + ic) < C
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
        hidden = C // reduction
        hidden = hidden if hidden > 0 else 1
        K = int(kernel_size)
        pad = K // 2
        dev = x.device

        w1 = ca_fc1_weight.contiguous().view(hidden, C)
        w2 = ca_fc2_weight.contiguous().view(C, hidden)
        sw = sa_conv_weight.contiguous().view(2, K, K)
        sb = sa_conv_bias.contiguous()

        ca = torch.empty((B, C), device=dev, dtype=torch.float32)
        st = torch.empty((B, 2, HW), device=dev, dtype=torch.float32)
        sp = torch.empty((B, HW), device=dev, dtype=torch.float32)
        out = torch.empty_like(x)

        BT1 = min(128, triton.next_power_of_2(HW))
        cbam_channel_attn_kernel[(B,)](
            x, w1, w2, ca, C, HW, hidden, 1.0 / HW,
            CC=32, BT=BT1, BH=triton.next_power_of_2(hidden))

        WT = triton.next_power_of_2(W)
        nrow = B * H
        g2 = nrow if nrow < self.VEC_CORE_NUM else self.VEC_CORE_NUM
        cbam_spatial_stats_kernel[(g2,)](
            x, ca, st, B, H, W, HW, C, 1.0 / C, g2,
            BC=64, WT=WT)

        BK = triton.next_power_of_2(K)
        cbam_spatial_conv_kernel[(g2,)](
            st, sw, sb, sp, B, H, W, HW, K, pad, g2,
            BK=BK, WT=WT)

        BT4 = 32
        nht = triton.cdiv(HW, BT4)
        total4 = B * nht
        g4 = total4 if total4 < self.VEC_CORE_NUM else self.VEC_CORE_NUM
        cbam_final_kernel[(g4,)](
            x, ca, sp, out, B, C, HW, g4,
            BC=128, BT=BT4)

        return out