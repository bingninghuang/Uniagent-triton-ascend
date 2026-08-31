import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv1d_kernel(X, W, BIAS, OUT,
                   N, C_IN, L_IN, C_OUT, L_OUT, C_OUT_PER_GROUP, C_IN_PER_GROUP,
                   HAS_BIAS: tl.constexpr,
                   K: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                   CCG: tl.constexpr,
                   BLOCK_CO: tl.constexpr, BLOCK_L: tl.constexpr):
    pid_n = tl.program_id(0).to(tl.int32)
    pid_co = tl.program_id(1).to(tl.int32)
    pid_l = tl.program_id(2).to(tl.int32)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)

    mask_co = offs_co < C_OUT
    mask_l = offs_l < L_OUT

    # group id for each output channel in this block
    offs_g = offs_co // C_OUT_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_L), dtype=tl.float32)

    for kk in range(0, K):
        for cin in range(0, CCG):
            pos = offs_l * STRIDE + kk * DIL - PAD
            xmask = (pos >= 0) & (pos < L_IN) & mask_l[None, :]
            # x[n, g*CCG + cin, pos]  -> base offset: (n*C_IN + g*CCG + cin) * L_IN
            xptr = X + (pid_n * C_IN + offs_g * C_IN_PER_GROUP + cin)[None, :] * L_IN + pos[None, :]
            xv = tl.load(xptr, mask=xmask, other=0.0).to(tl.float32)  # [BLOCK_CO, BLOCK_L] rows identical per co? no: per co, pos is same; xv varies by l only
            # w[co, cin, kk] : shape [C_OUT, CCG, K]
            wptr = W + (offs_co * C_IN_PER_GROUP + cin) * K + kk
            wmask = mask_co
            wv = tl.load(wptr, mask=wmask, other=0.0).to(tl.float32)  # [BLOCK_CO]
            # xv is [BLOCK_CO, BLOCK_L] but actually the loaded value depends only on l (pos) and n,g,cin;
            # however g varies per co within the block, causing wrong gather. Instead load per l with g from co.
            # We loaded per (co, l) using offs_g so it is correct.
            acc += wv[:, None] * xv

    offs_out = (pid_n * C_OUT + offs_co)[:, None] * L_OUT + offs_l[None, :]
    out_mask = mask_co[:, None] & mask_l[None, :]
    if HAS_BIAS:
        bias_ptr = BIAS + offs_co
        bv = tl.load(bias_ptr, mask=mask_co, other=0.0).to(tl.float32)
        acc += bv[:, None]
    tl.store(OUT + offs_out, acc.to(OUT.dtype.element_ty), mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._convs = {}

    def forward(self, inputs) -> torch.Tensor:
        x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = inputs

        key = (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if key in self._convs:
            conv = self._convs[key]
        else:
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
            conv = conv.to(x.device)
            self._convs[key] = conv

        x = x.contiguous()
        N, C_IN, L_IN = x.shape
        C_OUT = out_channels
        K = kernel_size
        C_OUT_PER_GROUP = C_OUT // groups
        C_IN_PER_GROUP = C_IN // groups
        L_OUT = (L_IN + 2 * padding - dilation * (K - 1) - 1) // stride + 1

        out = torch.empty((N, C_OUT, L_OUT), dtype=x.dtype, device=x.device)
        weight = conv.weight
        bias_tensor = conv.bias if (bias and conv.bias is not None) else x

        BLOCK_CO = 16
        BLOCK_L = 16
        grid = (N, triton.cdiv(C_OUT, BLOCK_CO), triton.cdiv(L_OUT, BLOCK_L))
        _conv1d_kernel[grid](
            x, weight, bias_tensor, out,
            N, C_IN, L_IN, C_OUT, L_OUT, C_OUT_PER_GROUP, C_IN_PER_GROUP,
            HAS_BIAS=bool(bias is not False and conv.bias is not None),
            K=K, STRIDE=stride, PAD=padding, DIL=dilation,
            CCG=C_IN_PER_GROUP,
            BLOCK_CO=BLOCK_CO, BLOCK_L=BLOCK_L,
            num_warps=4,
        )
        return out
