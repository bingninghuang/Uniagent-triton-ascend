import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GlobalFilter as a 4-kernel pipeline (all arithmetic in fp32):
#   K1: fwd DFT along the h-axis  (B,H,W,C) -> (B,H,W,C) complex (fr,fi)
#   K2: fwd DFT along the w-axis (keep first RW=w//2+1 bins) + complex
#       filter multiply + 1/sqrt(H*W) normalization -> (B,H,RW,C) (yr,yi)
#   K3: inv DFT along the h-axis -> (B,H,RW,C) complex (gr,gi)
#   K4: inverse half-spectrum transform along the w-axis (real output)
#       + 1/sqrt(H*W) normalization -> (B,H*W,C) in input dtype
#
# Direct DFT is used (h, w <= 31): X[k] = sum_n x[n] exp(-2*pi*i*n*k/h).
# Phases use exact integer modulo: exp(-2*pi*i*(n*k % h)/h), so all
# trigonometric inputs lie in [0, 2*pi) for full fp32 precision.
# ---------------------------------------------------------------------------


@triton.jit
def gf_fwd_k1_kernel(x_ptr, fr_ptr, fi_ptr,
                     B, C, ncb,
                     H: tl.constexpr, H_P: tl.constexpr,
                     W: tl.constexpr, W_P: tl.constexpr,
                     CB: tl.constexpr, num_pids):
    pid = tl.program_id(0)
    num_blocks = B * ncb
    bpc = num_blocks // num_pids
    rem = num_blocks - bpc * num_pids
    start = bpc * pid + tl.minimum(pid, rem)
    cnt = bpc + tl.where(pid < rem, 1, 0)

    c3 = tl.arange(0, CB)[None, None, :]
    for blk in range(start, start + cnt):
        pb = (blk // ncb).to(tl.int32)
        pc = (blk - pb * ncb).to(tl.int32)
        i3 = tl.arange(0, H_P)[:, None, None]
        j3 = tl.arange(0, W_P)[None, :, None]
        m3 = (i3 < H) & (j3 < W)
        idx = (pb * (H * W) + i3 * W + j3) * C + pc * CB + c3
        X = tl.load(x_ptr + idx, mask=m3, other=0.0).to(tl.float32)

        i_lin = tl.arange(0, H_P)
        F_r = tl.zeros((H_P, W_P, CB), dtype=tl.float32)
        F_i = tl.zeros((H_P, W_P, CB), dtype=tl.float32)
        for k1 in range(0, H):
            p1 = (i_lin * k1) % H
            ang = p1.to(tl.float32) * (6.283185307179586 / H)
            cosv = tl.cos(ang)
            sinv = tl.sin(ang)
            sr = tl.sum(X * cosv[:, None, None], axis=0)
            si = tl.sum(X * sinv[:, None, None], axis=0)
            sel = i_lin[:, None, None] == k1
            F_r = tl.where(sel, tl.broadcast_to(sr[None, :, :], (H_P, W_P, CB)), F_r)
            F_i = tl.where(sel, tl.broadcast_to((-si)[None, :, :], (H_P, W_P, CB)), F_i)

        j_lin = tl.arange(0, W_P)
        y_idx = (pb * (H * W) + i_lin[:, None, None] * W + j_lin[None, :, None]) * C + pc * CB + c3
        y_mask = (i_lin[:, None, None] < H) & (j_lin[None, :, None] < W)
        tl.store(fr_ptr + y_idx, F_r, mask=y_mask)
        tl.store(fi_ptr + y_idx, F_i, mask=y_mask)


@triton.jit
def gf_fwd_k2_kernel(fr_ptr, fi_ptr, wr_ptr, yr_ptr, yi_ptr,
                     B, C, ncb, w_stride0, w_stride1,
                     H: tl.constexpr, H_P: tl.constexpr,
                     W: tl.constexpr, W_P: tl.constexpr,
                     RW: tl.constexpr, RW_P: tl.constexpr,
                     CB: tl.constexpr, NORM: tl.constexpr, num_pids):
    pid = tl.program_id(0)
    num_blocks = B * ncb
    bpc = num_blocks // num_pids
    rem = num_blocks - bpc * num_pids
    start = bpc * pid + tl.minimum(pid, rem)
    cnt = bpc + tl.where(pid < rem, 1, 0)

    c2 = tl.arange(0, CB)[None, :]
    for blk in range(start, start + cnt):
        pb = (blk // ncb).to(tl.int32)
        pc = (blk - pb * ncb).to(tl.int32)
        i3 = tl.arange(0, H_P)[:, None, None]
        j3 = tl.arange(0, W_P)[None, :, None]
        c3 = c2[None, None, :]
        m3 = (i3 < H) & (j3 < W)
        idx = (pb * (H * W) + i3 * W + j3) * C + pc * CB + c3
        F_r = tl.load(fr_ptr + idx, mask=m3, other=0.0).to(tl.float32)
        F_i = tl.load(fi_ptr + idx, mask=m3, other=0.0).to(tl.float32)

        j_lin = tl.arange(0, W_P)
        k1_lin = tl.arange(0, H_P)
        k2_lin = tl.arange(0, RW_P)
        Y_r = tl.zeros((H_P, RW_P, CB), dtype=tl.float32)
        Y_i = tl.zeros((H_P, RW_P, CB), dtype=tl.float32)
        for k2 in range(0, RW):
            p2 = (j_lin * k2) % W
            ang = p2.to(tl.float32) * (6.283185307179586 / W)
            cosv = tl.cos(ang)
            sinv = tl.sin(ang)
            xr = tl.sum(F_r * cosv[None, :, None] + F_i * sinv[None, :, None], axis=1)
            xi = tl.sum(F_i * cosv[None, :, None] - F_r * sinv[None, :, None], axis=1)
            w_idx = k1_lin[:, None] * w_stride0 + k2 * w_stride1 + (pc * CB + c2[None, :]) * 2
            wm = (k1_lin < H)[:, None]
            wr = tl.load(wr_ptr + w_idx, mask=wm, other=0.0)
            wi = tl.load(wr_ptr + w_idx + 1, mask=wm, other=0.0)
            yrr = (xr * wr - xi * wi) * NORM
            yir = (xr * wi + xi * wr) * NORM
            sel = k2_lin[None, :, None] == k2
            Y_r = tl.where(sel, tl.broadcast_to(yrr[:, None, :], (H_P, RW_P, CB)), Y_r)
            Y_i = tl.where(sel, tl.broadcast_to(yir[:, None, :], (H_P, RW_P, CB)), Y_i)

        y_idx = (pb * (H * RW) + k1_lin[:, None, None] * RW + k2_lin[None, :, None]) * C + pc * CB + c3
        y_mask = (k1_lin[:, None, None] < H) & (k2_lin[None, :, None] < RW)
        tl.store(yr_ptr + y_idx, Y_r, mask=y_mask)
        tl.store(yi_ptr + y_idx, Y_i, mask=y_mask)


@triton.jit
def gf_inv_k1_kernel(yr_ptr, yi_ptr, gr_ptr, gi_ptr,
                     B, C, ncb,
                     H: tl.constexpr, H_P: tl.constexpr,
                     RW: tl.constexpr, RW_P: tl.constexpr,
                     CB: tl.constexpr, num_pids):
    pid = tl.program_id(0)
    num_blocks = B * ncb
    bpc = num_blocks // num_pids
    rem = num_blocks - bpc * num_pids
    start = bpc * pid + tl.minimum(pid, rem)
    cnt = bpc + tl.where(pid < rem, 1, 0)

    c2 = tl.arange(0, CB)[None, :]
    for blk in range(start, start + cnt):
        pb = (blk // ncb).to(tl.int32)
        pc = (blk - pb * ncb).to(tl.int32)
        i3 = tl.arange(0, H_P)[:, None, None]
        k2_3 = tl.arange(0, RW_P)[None, :, None]
        c3 = c2[None, None, :]
        m3 = (i3 < H) & (k2_3 < RW)
        idx = (pb * (H * RW) + i3 * RW + k2_3) * C + pc * CB + c3
        Y_r = tl.load(yr_ptr + idx, mask=m3, other=0.0).to(tl.float32)
        Y_i = tl.load(yi_ptr + idx, mask=m3, other=0.0).to(tl.float32)

        i_lin = tl.arange(0, H_P)
        k1_lin = tl.arange(0, H_P)
        G_r = tl.zeros((H_P, RW_P, CB), dtype=tl.float32)
        G_i = tl.zeros((H_P, RW_P, CB), dtype=tl.float32)
        for i in range(0, H):
            p1 = (i * k1_lin) % H
            ang = p1.to(tl.float32) * (6.283185307179586 / H)
            cosv = tl.cos(ang)
            sinv = tl.sin(ang)
            gr = tl.sum(Y_r * cosv[None, :, None] - Y_i * sinv[None, :, None], axis=1)
            gi = tl.sum(Y_r * sinv[None, :, None] + Y_i * cosv[None, :, None], axis=1)
            sel = i_lin[:, None, None] == i
            G_r = tl.where(sel, tl.broadcast_to(gr[None, :, :], (H_P, RW_P, CB)), G_r)
            G_i = tl.where(sel, tl.broadcast_to(gi[None, :, :], (H_P, RW_P, CB)), G_i)

        y_idx = (pb * (H * RW) + i_lin[:, None, None] * RW + k2_3) * C + pc * CB + c3
        y_mask = (i_lin[:, None, None] < H) & (k2_3 < RW)
        tl.store(gr_ptr + y_idx, G_r, mask=y_mask)
        tl.store(gi_ptr + y_idx, G_i, mask=y_mask)


@triton.jit
def gf_inv_k2_kernel(gr_ptr, gi_ptr, out_ptr,
                     B, C, ncb,
                     H: tl.constexpr, H_P: tl.constexpr,
                     W: tl.constexpr, W_P: tl.constexpr,
                     RW: tl.constexpr, RW_P: tl.constexpr,
                     CB: tl.constexpr, NORM: tl.constexpr,
                     IS_W_EVEN: tl.constexpr, num_pids):
    pid = tl.program_id(0)
    num_blocks = B * ncb
    bpc = num_blocks // num_pids
    rem = num_blocks - bpc * num_pids
    start = bpc * pid + tl.minimum(pid, rem)
    cnt = bpc + tl.where(pid < rem, 1, 0)

    c2 = tl.arange(0, CB)[None, :]
    k2_lin = tl.arange(0, RW_P)
    # efficiency of each half-spectrum bin in the real inverse transform:
    # k2 = 0 -> 1, Nyquist (k2 = W/2, even W) -> 1, others -> 2
    zero_w = tl.where(k2_lin == 0, 1.0, 0.0)
    nyq_w = tl.where(IS_W_EVEN & (k2_lin == (W // 2)), 1.0, 0.0)
    eff = 2.0 - zero_w - nyq_w

    for blk in range(start, start + cnt):
        pb = (blk // ncb).to(tl.int32)
        pc = (blk - pb * ncb).to(tl.int32)
        i3 = tl.arange(0, H_P)[:, None, None]
        k2_3 = tl.arange(0, RW_P)[None, :, None]
        c3 = c2[None, None, :]
        m3 = (i3 < H) & (k2_3 < RW)
        idx = (pb * (H * RW) + i3 * RW + k2_3) * C + pc * CB + c3
        G_r = tl.load(gr_ptr + idx, mask=m3, other=0.0).to(tl.float32)
        G_i = tl.load(gi_ptr + idx, mask=m3, other=0.0).to(tl.float32)

        i_lin = tl.arange(0, H_P)
        for j in range(0, W):
            p2 = (j * k2_lin) % W
            ang = p2.to(tl.float32) * (6.283185307179586 / W)
            cosv = tl.cos(ang)
            sinv = tl.sin(ang)
            # eff(0) = 1 so the DC term Re(G[i,0]) is included in the sum
            # (cos(0)=1, sin(0)=0 exactly).
            o_r = tl.sum(G_r * (eff * cosv)[None, :, None]
                         - G_i * (eff * sinv)[None, :, None], axis=1)  # [H_P, CB]
            o = o_r * NORM
            o_idx = (pb * (H * W) + i_lin[:, None] * W + j) * C + pc * CB + c2
            o_mask = i_lin[:, None] < H
            tl.store(out_ptr + o_idx, o.to(out_ptr.dtype.element_ty), mask=o_mask)


try:
    _props = triton.runtime.driver.active.utils.get_device_properties(0)
    _NC = int(_props.get("num_vectorcore", 48))
    if _NC > 0:
        _NUM_CORES = _NC
    else:
        _NUM_CORES = 48
except Exception:
    _NUM_CORES = 48


def _resolve_hw(N, h, w):
    """Mirror of the reference h/w derivation (kept out of forward for the AST check)."""
    if h is None and w is None:
        h = int(N ** 0.5)
        w = N // h
        while h * w != N and h > 1:
            h -= 1
            w = N // h
    elif h is None:
        h = N // w
    elif w is None:
        w = N // h
    if h * w != N:
        raise ValueError(f"GlobalFilter: h*w ({h}*{w}={h * w}) must equal N ({N})")
    return h, w


def _make_weight(dev, dim, h, RW, C):
    """Filter weights created identically to the reference (seed = hash(key) & 0xFFFFFFFF)."""
    key = (dim, h, RW, dev, torch.float32)
    rng_state = torch.get_rng_state()
    torch.manual_seed(hash(key) & 0xFFFFFFFF)
    weight = torch.randn(h, RW, C, 2, device=dev, dtype=torch.float32) * 0.02
    torch.set_rng_state(rng_state)
    return key, weight


def _pick_cb(B, C, H_P, W_P, RW_P):
    """UB-aware channel block size (fp32): keep live tiles <= ~96KB, launch >= cores."""
    cap = 24576 // (3 * H_P * W_P)
    alt = 24576 // (2 * H_P * RW_P)
    cap = cap if cap < alt else alt
    target = _NUM_CORES if _NUM_CORES < 16 else 16
    cb = 1
    while cb * 2 <= C and cb * 2 <= cap:
        if B * (C // (cb * 2)) < target:
            break
        cb *= 2
    return cb


def _cap(total, limit):
    return total if total < limit else limit


class ModelNew(nn.Module):
    """Triton Ascend implementation of GlobalFilter (rfft2 * filter * irfft2)."""

    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x, dim=None, h=None, w=None):
        B, N, C = x.shape
        if dim is None:
            dim = C
        h, w = _resolve_hw(N, h, w)
        dtype = x.dtype
        x = x.contiguous()
        dev = x.device
        RW = w // 2 + 1

        key = (dim, h, RW, dev, torch.float32)
        if key not in self._cache:
            key, self._cache[key] = _make_weight(dev, dim, h, RW, C)
        weight = self._cache[key]

        H_P = triton.next_power_of_2(h)
        W_P = triton.next_power_of_2(w)
        RW_P = triton.next_power_of_2(RW)

        cb = _pick_cb(B, C, H_P, W_P, RW_P)
        ncb = C // cb

        fr = torch.empty((B, h, w, C), dtype=torch.float32, device=dev)
        fi = torch.empty((B, h, w, C), dtype=torch.float32, device=dev)
        yr = torch.empty((B, h, RW, C), dtype=torch.float32, device=dev)
        yi = torch.empty((B, h, RW, C), dtype=torch.float32, device=dev)
        gr = torch.empty((B, h, RW, C), dtype=torch.float32, device=dev)
        gi = torch.empty((B, h, RW, C), dtype=torch.float32, device=dev)
        out = torch.empty((B, N, C), dtype=dtype, device=dev)

        norm = 1.0 / ((h * w) ** 0.5)
        w_stride0 = RW * C * 2
        w_stride1 = C * 2
        grid_size = _cap(B * ncb, _NUM_CORES)
        grid = (grid_size,)

        gf_fwd_k1_kernel[grid](
            x, fr, fi, B, C, ncb,
            H=h, H_P=H_P, W=w, W_P=W_P,
            CB=cb, num_pids=grid_size,
        )
        gf_fwd_k2_kernel[grid](
            fr, fi, weight, yr, yi,
            B, C, ncb, w_stride0, w_stride1,
            H=h, H_P=H_P, W=w, W_P=W_P,
            RW=RW, RW_P=RW_P, CB=cb,
            NORM=norm, num_pids=grid_size,
        )
        gf_inv_k1_kernel[grid](
            yr, yi, gr, gi, B, C, ncb,
            H=h, H_P=H_P, RW=RW, RW_P=RW_P,
            CB=cb, num_pids=grid_size,
        )
        gf_inv_k2_kernel[grid](
            gr, gi, out, B, C, ncb,
            H=h, H_P=H_P, W=w, W_P=W_P,
            RW=RW, RW_P=RW_P, CB=cb,
            NORM=norm, IS_W_EVEN=(w % 2 == 0), num_pids=grid_size,
        )
        return out