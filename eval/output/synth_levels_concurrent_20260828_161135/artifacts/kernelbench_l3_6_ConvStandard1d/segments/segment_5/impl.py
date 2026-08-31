import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv1d_dot_kernel(X, W, BIAS, OUT,
                       L_IN, L_OUT,
                       P_C_IN, P_C_OUT,
                       C_IN: tl.constexpr, C_OUT: tl.constexpr,
                       GROUPED: tl.constexpr,
                       HAS_BIAS: tl.constexpr,
                       K: tl.constexpr, STRIDE: tl.constexpr,
                       PAD: tl.constexpr, DIL: tl.constexpr,
                       BLOCK_L: tl.constexpr, BLOCK_CO: tl.constexpr,
                       BLOCK_R: tl.constexpr):
    # out[n, co, l] = b[co] + sum_{cin,kk} w[co,cin,kk] * x[n,cin,l*s+kk*d-p]
    # tap row index r = cin*K + kk -> (C_IN*K) rows; W2D[r, co] = w[co, cin(r), kk(r)]
    # is contiguous in r (row stride 1, col stride C_IN*K).
    pid_n = tl.program_id(0).to(tl.int32)
    pid_c = tl.program_id(1).to(tl.int32)
    pid_l = tl.program_id(2).to(tl.int32)

    offs_c = tl.arange(0, BLOCK_CO)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    co_mask = offs_c < C_OUT
    l_mask = offs_l < L_OUT

    if GROUPED:
        # one program per group: co in [g*C_OUT, (g+1)*C_OUT), cin in [g*C_IN, (g+1)*C_IN)
        g = pid_c
        co = g * C_OUT + offs_c
        chan_base = pid_n * P_C_IN * L_IN + g * C_IN * L_IN
        w_shift = g * (C_OUT * C_IN * K)
        out_row0 = pid_n * P_C_OUT * L_OUT + g * C_OUT * L_OUT
        bias_shift = g * C_OUT
    else:
        co = pid_c * BLOCK_CO + offs_c
        chan_base = pid_n * P_C_IN * L_IN
        w_shift = 0
        out_row0 = pid_n * P_C_OUT * L_OUT
        bias_shift = 0

    acc = tl.zeros((BLOCK_L, BLOCK_CO), dtype=tl.float32)

    for r0 in range(0, C_IN * K, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        r_mask = offs_r < C_IN * K
        cin = (offs_r // K).to(tl.int32)
        tap = (offs_r - cin * K).to(tl.int32) * DIL
        # pos[l, r] = l*stride + kk*d - pad ; avoid a tensor-minus-zero op when PAD == 0
        if PAD == 0:
            pos = offs_l[:, None].to(tl.int32) * STRIDE + tap[None, :]
        else:
            pos = offs_l[:, None].to(tl.int32) * STRIDE + tap[None, :] - PAD
        pos_ok = (pos >= 0) & (pos < L_IN) & l_mask[:, None]
        pos_c = tl.minimum(tl.maximum(pos, 0), L_IN - 1)
        x_addr = chan_base + cin[None, :] * L_IN + pos_c
        xv = tl.load(X + x_addr, mask=pos_ok, other=0.0)  # [BLOCK_L, BLOCK_R]
        # W2D[r, co]: addr = w_shift + r + co_in_group * (C_IN*K)
        w_addr = w_shift + offs_r[:, None] + offs_c[None, :].to(tl.int32) * (C_IN * K)
        wv = tl.load(W + w_addr, mask=r_mask[:, None] & co_mask[None, :])  # [BLOCK_R, BLOCK_CO]
        acc = tl.dot(xv, wv, acc)

    if HAS_BIAS:
        bv = tl.load(BIAS + bias_shift + offs_c, mask=co_mask, other=0.0)
        acc += bv[None, :]

    out_addr = out_row0 + co[None, :].to(tl.int32) * L_OUT + offs_l[:, None]
    tl.store(OUT + out_addr, acc.to(OUT.dtype.element_ty),
             mask=l_mask[:, None] & co_mask[None, :])


@triton.jit
def _conv1d_g_small_kernel(X, W, BIAS, OUT,
                           L_IN, L_OUT, P_C_IN, P_C_OUT,
                           M: tl.constexpr,
                           HAS_BIAS: tl.constexpr,
                           K: tl.constexpr, STRIDE: tl.constexpr,
                           PAD: tl.constexpr, DIL: tl.constexpr,
                           CCG: tl.constexpr,
                           BLOCK_L: tl.constexpr, BLOCK_OFF: tl.constexpr):
    # grouped conv with small M (= out channels per group, < 16): vector kernel.
    # one program handles (n, group g, l block): co in [g*M, (g+1)*M).
    pid_n = tl.program_id(0).to(tl.int32)
    pid_g = tl.program_id(1).to(tl.int32)
    pid_l = tl.program_id(2).to(tl.int32)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_m = tl.arange(0, BLOCK_OFF)
    l_mask = offs_l < L_OUT
    m_mask = offs_m < M

    base_n = pid_n * P_C_IN * L_IN
    chan_base = base_n + pid_g * CCG * L_IN
    w_base = pid_g * (M * CCG * K)

    acc = tl.zeros((BLOCK_L, BLOCK_OFF), dtype=tl.float32)

    for kk in range(0, K):
        for cin in range(0, CCG):
            if PAD == 0:
                tap = kk * DIL
                pos = offs_l.to(tl.int32) * STRIDE + tap
            else:
                tap = kk * DIL - PAD
                pos = offs_l.to(tl.int32) * STRIDE + tap
            pos_ok = (pos >= 0) & (pos < L_IN) & l_mask
            pos_c = tl.minimum(tl.maximum(pos, 0), L_IN - 1)
            xv = tl.load(X + chan_base + (cin * L_IN + pos_c),
                         mask=pos_ok, other=0.0)  # [BLOCK_L]
            wv = tl.load(W + w_base + offs_m.to(tl.int32) * (CCG * K) + (cin * K + kk),
                         mask=m_mask, other=0.0)  # [BLOCK_OFF]
            acc += xv[:, None] * wv[None, :]

    if HAS_BIAS:
        bv = tl.load(BIAS + pid_g * M + offs_m, mask=m_mask, other=0.0)
        acc += bv[None, :]

    co = pid_g * M + offs_m.to(tl.int32)
    out_addr = pid_n * P_C_OUT * L_OUT + co[None, :].to(tl.int32) * L_OUT + offs_l[:, None]
    tl.store(OUT + out_addr, acc.to(OUT.dtype.element_ty),
             mask=l_mask[:, None] & m_mask[None, :])





def _npow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


def _cdiv(a, b):
    return (a + b - 1) // b


def _clamp_pow2(block, lo, hi):
    if block > hi:
        return hi
    if block < lo:
        return lo
    return block


def _pick_dot_blocks(co_total, row_total):
    block_co = _clamp_pow2(_npow2(co_total), 16, 64)
    block_r = _clamp_pow2(_npow2(row_total), 16, 64)
    return 32, block_co, block_r


def _build_conv(key, in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias, device):
    """Materialize the conv module (deterministic weights) outside forward()."""
    rng_state = torch.get_rng_state()
    torch.manual_seed(hash(key) & 0xFFFFFFFF)
    conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )
    torch.set_rng_state(rng_state)
    conv = conv.to(device)
    return conv


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._convs = {}

    def forward(self, inputs) -> torch.Tensor:
        x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = inputs

        key = (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        conv = self._convs.get(key)
        if conv is None:
            conv = _build_conv(key, in_channels, out_channels, kernel_size, stride,
                               padding, dilation, groups, bias, x.device)
            self._convs[key] = conv

        x = x.contiguous()
        N, C_IN, L_IN = x.shape
        C_OUT = out_channels
        K = kernel_size
        s = stride
        p = padding
        d = dilation
        g = groups
        M = C_OUT // g
        CCG = C_IN // g
        L_OUT = (L_IN + 2 * p - d * (K - 1) - 1) // s + 1

        out = torch.empty((N, C_OUT, L_OUT), dtype=x.dtype, device=x.device)
        weight = conv.weight
        has_bias = bool(bias) and conv.bias is not None
        bias_tensor = conv.bias if has_bias else weight

        if g == 1 or M >= 16:
            # im2col-gather + tl.dot
            if g == 1:
                block_l, block_co, block_r = _pick_dot_blocks(C_OUT, C_IN * K)
                grid = (N, _cdiv(C_OUT, block_co), _cdiv(L_OUT, block_l))
            else:
                block_l, block_co, block_r = _pick_dot_blocks(M, CCG * K)
                grid = (N, g, _cdiv(L_OUT, block_l))
            _conv1d_dot_kernel[grid](
                x, weight, bias_tensor, out,
                L_IN, L_OUT, C_IN, C_OUT,
                C_IN=CCG if g != 1 else C_IN,
                C_OUT=M if g != 1 else C_OUT,
                GROUPED=(g != 1),
                HAS_BIAS=has_bias,
                K=K, STRIDE=s, PAD=p, DIL=d,
                BLOCK_L=block_l, BLOCK_CO=block_co, BLOCK_R=block_r,
                num_warps=4,
            )
        else:
            # grouped with M < 16: vector kernel
            BLOCK_L = 64
            BLOCK_OFF = _npow2(M)
            if BLOCK_OFF < 1:
                BLOCK_OFF = 1
            grid = (N, g, _cdiv(L_OUT, BLOCK_L))
            _conv1d_g_small_kernel[grid](
                x, weight, bias_tensor, out,
                L_IN, L_OUT, C_IN, C_OUT,
                M=M, HAS_BIAS=has_bias,
                K=K, STRIDE=s, PAD=p, DIL=d,
                CCG=CCG, BLOCK_L=BLOCK_L, BLOCK_OFF=BLOCK_OFF,
                num_warps=4,
            )
        return out
