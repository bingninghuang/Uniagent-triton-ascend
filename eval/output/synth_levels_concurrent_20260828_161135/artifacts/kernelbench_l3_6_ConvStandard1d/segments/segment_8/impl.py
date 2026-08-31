import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv1d_dot_kernel(X, W, BIAS, OUT,
                       L_IN, L_OUT, C_IN, C_OUT,
                       K: tl.constexpr, S: tl.constexpr,
                       P: tl.constexpr, D: tl.constexpr,
                       CCG: tl.constexpr, M: tl.constexpr,
                       HAS_BIAS: tl.constexpr, GROUPED: tl.constexpr,
                       BLOCK_L: tl.constexpr, BLOCK_CO: tl.constexpr,
                       BLOCK_R: tl.constexpr):
    # out[n, co, l] = bias[co] + sum_{r} x[n, cin0 + r//K, l*S - P + (r%K)*D] * W[co, r//K, r%K]
    # r runs over [0, CCG*K); W rows within a group are contiguous: addr = group_base + co_local*(CCG*K) + r
    pid_n = tl.program_id(0).to(tl.int32)
    pid_c = tl.program_id(1).to(tl.int32)
    pid_l = tl.program_id(2).to(tl.int32)

    offs_c = tl.arange(0, BLOCK_CO).to(tl.int32)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L).to(tl.int32)
    l_mask = offs_l < L_OUT
    pos_base = offs_l * S - P                     # window start per output pos

    if GROUPED:
        co0 = pid_c * M
        cin0 = pid_c * CCG
        w_base = pid_c * (M * CCG * K)
        co_valid = offs_c < M
    else:
        co0 = pid_c * BLOCK_CO
        cin0 = 0
        w_base = 0
        co_valid = offs_c < (C_OUT - co0)

    co = co0 + offs_c
    x_base = pid_n * (C_IN * L_IN) + cin0 * L_IN
    acc = tl.zeros((BLOCK_L, BLOCK_CO), dtype=tl.float32)

    for r0 in range(0, CCG * K, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R).to(tl.int32)
        r_valid = offs_r < (CCG * K)
        ci = offs_r // K                          # local input channel
        tap = (offs_r - ci * K) * D               # tap displacement
        pos = pos_base[:, None] + tap[None, :]    # (BLOCK_L, BLOCK_R)
        ok = (pos >= 0) & (pos < L_IN)
        ok = ok & r_valid[None, :] & l_mask[:, None]
        pos_c = tl.minimum(tl.maximum(pos, 0), L_IN - 1)
        xv = tl.load(X + (x_base + ci[None, :] * L_IN + pos_c),
                     mask=ok, other=0.0)                      # (BLOCK_L, BLOCK_R)
        wv = tl.load(W + (w_base + offs_r[:, None] + co[None, :] * (CCG * K)),
                     mask=r_valid[:, None] & co_valid[None, :], other=0.0)  # (BLOCK_R, BLOCK_CO)
        acc = tl.dot(xv, wv, acc, out_dtype=tl.float32)

    if HAS_BIAS:
        bv = tl.load(BIAS + co, mask=co_valid, other=0.0)
        acc += bv[None, :]

    out_addr = pid_n * (C_OUT * L_OUT) + co[None, :] * L_OUT + offs_l[:, None]
    tl.store(OUT + out_addr, acc.to(OUT.dtype.element_ty),
             mask=l_mask[:, None] & co_valid[None, :])


@triton.jit
def _conv1d_vec_kernel(X, W, BIAS, OUT,
                       L_IN, L_OUT, C_IN, C_OUT,
                       K: tl.constexpr, S: tl.constexpr,
                       P: tl.constexpr, D: tl.constexpr,
                       M: tl.constexpr, CCG: tl.constexpr,
                       HAS_BIAS: tl.constexpr,
                       BLOCK_L: tl.constexpr, BLOCK_OFF: tl.constexpr):
    # Vector fallback for grouped conv where M (= out channels per group) < 16.
    # one program: (n, group, l block). co in [g*M, (g+1)*M).
    pid_n = tl.program_id(0).to(tl.int32)
    pid_g = tl.program_id(1).to(tl.int32)
    pid_l = tl.program_id(2).to(tl.int32)

    offs_m = tl.arange(0, BLOCK_OFF).to(tl.int32)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L).to(tl.int32)
    l_mask = offs_l < L_OUT
    m_mask = offs_m < M
    co = pid_g * M + offs_m

    x_base = pid_n * (C_IN * L_IN) + pid_g * CCG * L_IN
    w_base = pid_g * (M * CCG * K)

    acc = tl.zeros((BLOCK_L, BLOCK_OFF), dtype=tl.float32)
    for kk in range(0, K):
        for ci in range(0, CCG):
            pos = offs_l * S + kk * D - P
            ok = (pos >= 0) & (pos < L_IN) & l_mask
            pos_c = tl.minimum(tl.maximum(pos, 0), L_IN - 1)
            xv = tl.load(X + (x_base + ci * L_IN + pos_c), mask=ok, other=0.0)  # (BLOCK_L,)
            wv = tl.load(W + (w_base + offs_m * (CCG * K) + (ci * K + kk)),
                         mask=m_mask, other=0.0)                      # (BLOCK_OFF,)
            acc += xv[:, None] * wv[None, :]

    if HAS_BIAS:
        bv = tl.load(BIAS + co, mask=m_mask, other=0.0)
        acc += bv[None, :]

    out_addr = pid_n * (C_OUT * L_OUT) + co[None, :] * L_OUT + offs_l[:, None]
    tl.store(OUT + out_addr, acc.to(OUT.dtype.element_ty),
             mask=l_mask[:, None] & m_mask[None, :])


def _npow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


def _cdiv(a, b):
    return (a + b - 1) // b


def _pow2_clamp(v, lo, hi):
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return _npow2(v)


def _build_conv(key, in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias, device):
    """Materialize conv weights with the exact same RNG sequence as the reference."""
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
        has_bias = conv.bias is not None
        bias_tensor = conv.bias if has_bias else weight

        if M >= 16:
            # CUBE GEMM path: im2col-gather + tl.dot
            block_co = _pow2_clamp(M, 16, 64)
            block_r = _pow2_clamp(CCG * K, 16, 64)
            block_l = 32
            if g == 1:
                grid = (N, _cdiv(C_OUT, block_co), _cdiv(L_OUT, block_l))
            else:
                grid = (N, g, _cdiv(L_OUT, block_l))
            _conv1d_dot_kernel[grid](
                x, weight, bias_tensor, out,
                L_IN, L_OUT, C_IN, C_OUT,
                K=K, S=s, P=p, D=d,
                CCG=CCG, M=M,
                HAS_BIAS=has_bias, GROUPED=(g != 1),
                BLOCK_L=block_l, BLOCK_CO=block_co, BLOCK_R=block_r,
            )
        else:
            # VEC path: grouped conv with fewer than 16 out-channels per group
            block_l = 64
            grid = (N, g, _cdiv(L_OUT, block_l))
            _conv1d_vec_kernel[grid](
                x, weight, bias_tensor, out,
                L_IN, L_OUT, C_IN, C_OUT,
                K=K, S=s, P=p, D=d,
                M=M, CCG=CCG,
                HAS_BIAS=has_bias,
                BLOCK_L=block_l, BLOCK_OFF=_npow2(M),
            )
        return out
