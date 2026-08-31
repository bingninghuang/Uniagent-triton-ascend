import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv1d_vec_kernel(X, W, BIAS, OUT,
                       L_IN, L_OUT, C_IN, C_OUT,
                       S, P, D, K, CCG, M, G, COB_G,
                       HAS_BIAS: tl.constexpr,
                       BLOCK_L: tl.constexpr, BLOCK_CO: tl.constexpr):
    # out[n, co, l] = bias[co] + sum_{ci in [0,CCG), k in [0,K)}
    #                    x[n, cin0+ci, l*S - P + k*D] * W[co, ci, k]
    # (cross-correlation).  Grouped conv is handled generically: a group of
    # M output channels is produced from CCG input channels.  grid dim1 =
    # G*COB_G where COB_G = co-blocks per group.
    pid_n = tl.program_id(0).to(tl.int32)
    pid_cb = tl.program_id(1).to(tl.int32)
    pid_l = tl.program_id(2).to(tl.int32)

    offs_c = tl.arange(0, BLOCK_CO).to(tl.int32)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L).to(tl.int32)
    l_mask = offs_l < L_OUT
    pos_base = offs_l * S - P                      # window start per output pos (BLOCK_L,)

    gid = pid_cb // COB_G
    cob = pid_cb % COB_G
    co0 = gid * M + cob * BLOCK_CO                 # first global out-channel of this block
    cin0 = gid * CCG                               # first local input channel base for this group
    co_valid = offs_c < (M - cob * BLOCK_CO)

    co = co0 + offs_c
    x_base = pid_n * (C_IN * L_IN) + cin0 * L_IN
    w_base = co * (CCG * K)

    acc = tl.zeros((BLOCK_L, BLOCK_CO), dtype=tl.float32)

    for ci in range(0, CCG):
        x_ci_base = x_base + ci * L_IN
        w_ci_base = w_base + ci * K
        for k in range(0, K):
            x_pos = pos_base + k * D               # (BLOCK_L,)
            x_inb = (x_pos >= 0) & (x_pos < L_IN)
            x_idx = tl.minimum(tl.maximum(x_pos, 0), L_IN - 1)
            xv = tl.load(X + (x_ci_base + x_idx), mask=x_inb & l_mask, other=0.0)
            wv = tl.load(W + (w_ci_base + k), mask=co_valid, other=0.0)
            acc += xv[:, None] * wv[None, :]

    if HAS_BIAS:
        bv = tl.load(BIAS + co, mask=co_valid, other=0.0)
        acc += bv[None, :]

    out_addr = pid_n * (C_OUT * L_OUT) + co[None, :] * L_OUT + offs_l[:, None]
    tl.store(OUT + out_addr, acc.to(OUT.dtype.element_ty),
             mask=l_mask[:, None] & co_valid[None, :])


def _npow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


def _cdiv(a, b):
    return (a + b - 1) // b


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
    return conv.to(device)


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

        # Tiling: BLOCK_CO <= next_pow2(M); a co-block must stay within one group.
        block_co = min(_npow2(M), 64)
        block_l = min(256, 8192 // block_co)
        cob_g = _cdiv(M, block_co)

        grid = (N, g * cob_g, _cdiv(L_OUT, block_l))
        _conv1d_vec_kernel[grid](
            x, weight, bias_tensor, out,
            L_IN, L_OUT, C_IN, C_OUT,
            s, p, d, K, CCG, M, g, cob_g,
            HAS_BIAS=has_bias,
            BLOCK_L=block_l, BLOCK_CO=block_co,
        )
        return out
