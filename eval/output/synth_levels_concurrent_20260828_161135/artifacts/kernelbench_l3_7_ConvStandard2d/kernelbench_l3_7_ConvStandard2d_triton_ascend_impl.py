import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:  # pragma: no cover
    torch_npu = None

try:
    import torch_npu.npu.npu_config as _npu_config
except Exception:  # pragma: no cover
    _npu_config = None


def _get_core_counts():
    try:
        lim = _npu_config.get_device_limit(0)
        cube = int(lim.get("cube_core_num", 24))
        vec = int(lim.get("vector_core_num", 48))
        if cube > 0:
            return cube, vec
    except Exception:
        pass
    return 24, 48


@triton.jit
def _im2col_kernel(
    x_ptr, a_ptr,
    N, Cin, H, W, Wo, S,
    K, KK, sh, sw, dh, dw, ph, pw,
    s_blocks,
    num_cores: tl.constexpr,
    BSA: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    hw_total = H * W
    items = N * Cin * KK * s_blocks
    for item in range(pid, items, num_cores):
        sb = item % s_blocks
        t1 = item // s_blocks
        row = t1 % (Cin * KK)
        n = t1 // (Cin * KK)
        ci = row // KK
        kk = row % KK
        kh = kk // K
        kw = kk % K
        s = sb * BSA + tl.arange(0, BSA).to(tl.int32)
        s_ok = s < S
        oh = s // Wo
        ow = s % Wo
        # input coords; separable row/col decomposition
        h_idx = (oh.to(tl.int32) * sh) + (kh * dh - ph)      # (BSA,)
        w_idx = (ow.to(tl.int32) * sw) + (kw * dw - pw)      # (BSA,)
        valid = (
            s_ok
            & (h_idx.to(tl.float32) >= 0.0)
            & (h_idx.to(tl.float32) < H.to(tl.float32))
            & (w_idx.to(tl.float32) >= 0.0)
            & (w_idx.to(tl.float32) < W.to(tl.float32))
        )
        x_off = (n * Cin + ci) * hw_total + h_idx * W + w_idx
        xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
        a_off = (n * Cin * KK + row) * S + s
        tl.store(a_ptr + a_off, xv, mask=s_ok)


@triton.jit
def _conv_gemm_kernel(
    a_ptr, w_ptr, b_ptr, out_ptr,
    N, Cin, Co, S, KK, Kdim,
    G, Cin_g, Co_g,
    num_cores: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BCO: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    co_blocks = tl.cdiv(Co_g, BCO)
    s_blocks = tl.cdiv(S, BS)
    items = N * G * co_blocks * s_blocks
    for item in range(pid, items, num_cores):
        sb = item % s_blocks
        t1 = item // s_blocks
        cb = t1 % co_blocks
        g = (t1 // co_blocks) % G
        n = t1 // (co_blocks * G)

        co_l = cb * BCO + tl.arange(0, BCO).to(tl.int32)
        s = sb * BS + tl.arange(0, BS).to(tl.int32)
        co_ok = co_l < Co_g
        s_ok = s < S

        acc = tl.zeros((BCO, BS), dtype=tl.float32)
        for k0 in range(0, Kdim, BK):
            k = k0 + tl.arange(0, BK).to(tl.int32)
            k_ok = k < Kdim
            # W tile: (BCO, BK) -> weight_flat[(g*Co_g + co_l)*Kdim + k]
            w_off = (g * Co_g + co_l)[:, None] * Kdim + k[None, :]
            w_tile = tl.load(
                w_ptr + w_off, mask=co_ok[:, None] & k_ok[None, :], other=0.0
            )
            # A tile: (BK, BS) -> a_flat[(n*Cin*KK + g*Kdim + k)*S + s]
            a_base = (n * Cin * KK + g * Kdim) * S
            a_off = a_base + k[:, None] * S + s[None, :]
            a_tile = tl.load(
                a_ptr + a_off, mask=k_ok[:, None] & s_ok[None, :], other=0.0
            )
            acc = tl.dot(w_tile, a_tile, acc)

        if HAS_BIAS:
            bv = tl.load(b_ptr + g * Co_g + co_l, mask=co_ok, other=0.0)
            acc = acc + bv[:, None]

        out_off = (n * Co + g * Co_g + co_l)[:, None] * S + s[None, :]
        tl.store(out_ptr + out_off, acc, mask=co_ok[:, None] & s_ok[None, :])


def _pow2_clamp(v, lo, hi):
    p = triton.next_power_of_2(v)
    if p < lo:
        return lo
    if p > hi:
        return hi
    return p


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._convs = {}
        self.CUBE_CORE_NUM, self.VEC_CORE_NUM = _get_core_counts()

    def forward(self, inputs) -> torch.Tensor:
        (
            x, in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias,
        ) = inputs

        x = x.contiguous()
        N, Cin, H, W = x.shape
        G = groups
        Cout = out_channels
        K = kernel_size
        KK = K * K
        sh = sw = stride
        ph = pw = padding
        dh = dw = dilation

        # ---- reproduce reference weight init exactly (seeded, CPU then move) ----
        key = (
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias,
        )
        entry = self._convs.get(key)
        if entry is None:
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
                bias=bias,
            )
            torch.set_rng_state(rng_state)
            conv = conv.to(x.device)
            entry = (conv.weight.contiguous(),
                     conv.bias.contiguous() if conv.bias is not None else None)
            self._convs[key] = entry
        w, b = entry

        Ho = (H + 2 * ph - dh * (K - 1) - 1) // sh + 1
        Wo = (W + 2 * pw - dw * (K - 1) - 1) // sw + 1
        S = Ho * Wo
        Cin_g = Cin // G
        Co_g = Cout // G
        Kdim = Cin_g * KK

        # im2col buffer: (N, Cin*KK, S) row-major, a[n, ci*KK+kk, s]
        # (for the 1x1, stride-1, no-pad case this is just x viewed as (N, Cin, S))
        if not (K == 1 and sh == 1 and ph == 0 and dh == 1 and H == Ho and W == Wo):
            a = torch.empty((N * Cin * KK, S), device=x.device, dtype=x.dtype)
            BSA = _pow2_clamp(S, 64, 512)
            s_blocks = triton.cdiv(S, BSA)
            total_items = N * Cin * KK * s_blocks
            grid = (min(total_items, self.VEC_CORE_NUM),)
            _im2col_kernel[grid](
                x, a,
                N, Cin, H, W, Wo, S,
                K, KK, sh, sw, dh, dw, ph, pw,
                s_blocks,
                grid[0], BSA,
            )
        else:
            a = x

        out = torch.empty((N, Cout, Ho, Wo), device=x.device, dtype=x.dtype)

        if b is None:
            b = w  # unused when HAS_BIAS=False
            has_bias = False
        else:
            has_bias = True

        # ---- GEMM block sizes ----
        BCO = _pow2_clamp(Co_g, 16, 64)
        BS = _pow2_clamp(S, 16, 128)
        BK = _pow2_clamp(Kdim, 16, 64)

        co_blocks = triton.cdiv(Co_g, BCO)
        s_blocks = triton.cdiv(S, BS)
        total_items = N * G * co_blocks * s_blocks
        grid = (min(total_items, self.CUBE_CORE_NUM),)
        _conv_gemm_kernel[grid](
            a, w, b, out,
            N, Cin, Cout, S, KK, Kdim,
            G, Cin_g, Co_g,
            grid[0], has_bias, BCO, BS, BK,
        )
        return out