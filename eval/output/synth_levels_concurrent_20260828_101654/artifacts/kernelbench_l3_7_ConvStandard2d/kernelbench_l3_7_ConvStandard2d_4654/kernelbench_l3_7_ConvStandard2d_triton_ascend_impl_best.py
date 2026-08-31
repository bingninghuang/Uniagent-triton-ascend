import torch
import torch.nn as nn
import triton
import triton.language as tl

_CONV_CACHE = {}


def _build_conv(key, device):
    conv = _CONV_CACHE.get(key)
    if conv is None:
        (in_channels, out_channels, kernel_size, stride, padding,
         dilation, groups, bias) = key
        rng_state = torch.get_rng_state()
        torch.manual_seed(hash(key) & 0xFFFFFFFF)
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        torch.set_rng_state(rng_state)
        conv = conv.to(device)
        _CONV_CACHE[key] = conv
    return conv


@triton.jit
def _conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M_total,           # N * H_out * W_out
    C_in, H_in, W_in,  # full input channels, spatial
    C_out, H_out, W_out,
    kh, kw, sh, sw, ph, pw, dh, dw,
    ICG, OGC,          # C_in // groups, C_out // groups
    KPG,               # ICG * kh * kw (K per group)
    Khw,               # kh * kw
    tiles_per_group,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_oc = tl.program_id(1)

    # ---- decompose spatial positions (M dim) ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    HW_out = H_out * W_out
    n = offs_m // HW_out
    hw = offs_m % HW_out
    h = hw // W_out
    w = hw % W_out

    # ---- decompose oc tile (never crosses group boundary) ----
    g = pid_oc // tiles_per_group
    local = pid_oc % tiles_per_group
    oc_start = g * OGC + local * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    oc_valid = offs_oc < (g + 1) * OGC

    acc = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)

    m_valid = offs_m < M_total

    for k0 in range(0, KPG, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        k_valid = kk < KPG
        c = kk // Khw
        rs = kk % Khw
        r = rs // kw
        s = rs % kw
        ic = g * ICG + c

        hi = h[:, None] * sh + r[None, :] * dh - ph
        wi = w[:, None] * sw + s[None, :] * dw - pw
        a_mask = (m_valid[:, None] & k_valid[None, :]
                  & (hi >= 0) & (hi < H_in) & (wi >= 0) & (wi < W_in))
        a_off = (n[:, None] * (C_in * H_in * W_in)
                 + ic[None, :] * (H_in * W_in)
                 + hi * W_in + wi)
        a = tl.load(x_ptr + a_off, mask=a_mask, other=0.0)

        b_mask = k_valid[:, None] & oc_valid[None, :]
        b_off = (offs_oc[None, :] * (ICG * Khw)
                 + ic[:, None] * Khw
                 + r[:, None] * kw + s[:, None])
        b = tl.load(w_ptr + b_off, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + offs_oc, mask=oc_valid, other=0.0)
        acc += bias_val[None, :]

    o_mask = m_valid[:, None] & oc_valid[None, :]
    o_off = (n[:, None] * (C_out * H_out * W_out)
             + offs_oc[None, :] * (H_out * W_out)
             + h[:, None] * W_out + w[:, None])
    tl.store(out_ptr + o_off, acc, mask=o_mask)


def _next_pow2(x, lo=16, hi=128):
    p = lo
    while p < x and p < hi:
        p *= 2
    return p


@triton.jit
def _diag_kernel(out_ptr, M_total, C_out, H_out, W_out,
                 BLOCK_M: tl.constexpr, BLOCK_OC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_oc = tl.program_id(1)
    HW_out = H_out * W_out
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = offs_m // HW_out
    hw = offs_m % HW_out
    h = hw // W_out
    w = hw % W_out
    sig = (pid_m + 1) * 100000 + (pid_oc + 1)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    o_off = (n[:, None] * (C_out * HW_out)
             + offs_oc[None, :] * HW_out
             + h[:, None] * W_out + w[:, None])
    m_valid = (offs_m < M_total)[:, None] & (offs_oc < C_out)[None, :]
    tl.store(out_ptr + o_off,
             sig + tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32),
             mask=m_valid)


@triton.jit
def _probe1(out_ptr):
    i = tl.arange(0, 4)
    tl.store(out_ptr + i, 1.0 + i.to(tl.float32))


@triton.jit
def _probe2(out_ptr):
    pid1 = tl.program_id(1)
    j = tl.arange(0, 2)
    tl.store(out_ptr + 4 + 2 * pid1 + j, 40.0 + pid1 * 10.0 + j.to(tl.float32))


@triton.jit
def _probe3(out_ptr):
    pid0 = tl.program_id(0)
    tl.store(out_ptr + 8 + pid0 + tl.arange(0, 1),
             60.0 + pid0 * 10.0 + tl.zeros((1,), dtype=tl.float32))


class ModelNew(nn.Module):
    r"""Triton Ascend implementation of ConvStandard2d (torch.nn.Conv2d)."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs) -> torch.Tensor:
        (x, in_channels, out_channels, kernel_size, stride, padding,
         dilation, groups, bias) = inputs

        key = (in_channels, out_channels, kernel_size, stride, padding,
               dilation, groups, bias)
        conv = _build_conv(key, x.device)

        weight = conv.weight
        bias_param = conv.bias

        N, C_in, H_in, W_in = x.shape
        C_out = out_channels
        kh, kw = kernel_size, kernel_size
        dh, dw = dilation, dilation
        sh, sw = stride, stride

        H_out = (H_in + 2 * padding - dh * (kh - 1) - 1) // sh + 1
        W_out = (W_in + 2 * padding - dw * (kw - 1) - 1) // sw + 1

        x = x.contiguous()
        weight = weight.contiguous()

        M_total = N * H_out * W_out
        ICG = C_in // groups
        OGC = C_out // groups
        Khw = kh * kw
        KPG = ICG * Khw

        out = torch.empty((N, C_out, H_out, W_out), device=x.device,
                          dtype=x.dtype)

        if bias_param is None:
            bias_tensor = torch.zeros(C_out, device=x.device, dtype=x.dtype)
            has_bias = False
        else:
            bias_tensor = bias_param.contiguous()
            has_bias = True

        # ---- tile config ----
        BLOCK_OC = _next_pow2(OGC, 16, 128)
        tiles_per_group = (OGC + BLOCK_OC - 1) // BLOCK_OC
        num_oc_tiles = groups * tiles_per_group
        KPG_pad = triton.cdiv(KPG, 16) * 16
        if KPG_pad <= 32:
            BLOCK_K = 32
        elif KPG_pad <= 64:
            BLOCK_K = 64
        else:
            BLOCK_K = 64
        if M_total <= 1024:
            BLOCK_M = 32
        else:
            BLOCK_M = 64

        DIAG = True
        if DIAG:
            _probe1[(1,)](out)
            _probe2[(1, 2)](out)
            _probe3[(2, 1)](out)
            return out

        grid = (triton.cdiv(M_total, BLOCK_M), num_oc_tiles)
        _conv2d_implicit_gemm_kernel[grid](
            x, weight, bias_tensor, out,
            M_total, C_in, H_in, W_in, C_out, H_out, W_out,
            kh, kw, sh, sw, padding, padding, dh, dw,
            ICG, OGC, KPG, Khw, tiles_per_group,
            has_bias,
            BLOCK_M=BLOCK_M, BLOCK_OC=BLOCK_OC, BLOCK_K=BLOCK_K,
        )
        return out
