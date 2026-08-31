import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_vec_core_num():
    try:
        import torch_npu

        limit = torch_npu.npu.npu_config.get_device_limit(0)
        return int(limit.get("vector_core_num", 48))
    except Exception:
        return 48


def _build_conv(key, device):
    in_channels, kernel_size, stride, padding, bias = key
    rng_state = torch.get_rng_state()
    torch.manual_seed(hash(key) & 0xFFFFFFFF)
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=(kernel_size, kernel_size),
        stride=stride,
        padding=padding,
        groups=in_channels,
        bias=bias,
    )
    torch.set_rng_state(rng_state)
    conv = conv.to(device)
    w = conv.weight.reshape(in_channels, kernel_size, kernel_size).contiguous()
    if conv.bias is not None:
        b = conv.bias.reshape(in_channels).contiguous()
    else:
        b = torch.zeros(in_channels, device=device, dtype=conv.weight.dtype)
    return w, b


@triton.jit
def _dw_conv2d_kernel(
    x_ptr,
    wext_ptr,
    out_ptr,
    C,
    H,
    WS,
    H_OUT,
    W_OUT,
    W_OUTS,
    NW_TILES,
    NUM_BLOCKS,
    CORE_C,
    P,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    KEXT: tl.constexpr,
    BLOCK_W: tl.constexpr,
    S: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    # G6 contiguous partition: each core gets a contiguous run of blocks
    bpc = NUM_BLOCKS // CORE_C
    rem = NUM_BLOCKS - bpc * CORE_C
    if pid < rem:
        start = pid * (bpc + 1)
        nblk = bpc + 1
    else:
        start = rem * (bpc + 1) + (pid - rem) * bpc
        nblk = bpc

    kk = tl.arange(0, KEXT).to(tl.int32)
    off = tl.arange(0, BLOCK_W).to(tl.int32)

    for loc_b in range(0, nblk):
        bidx = start + loc_b
        # bidx -> (row, wo_tile); row = (n*C + c) * H_OUT + ho
        row = bidx // NW_TILES
        wt = bidx - row * NW_TILES
        ho = row % H_OUT
        nc = row - ho * H_OUT  # n * C + c
        c = nc % C
        wo0 = wt * BLOCK_W

        # valid output lanes of this tile
        w_ok = off < (W_OUT - wo0)

        # base offset of channel-plane block for this (n, c)
        chan_base = nc * S * (H * WS)

        # weights of channel c (row KH*KW holds the bias)
        wrow = tl.load(wext_ptr + c * KEXT + kk)

        ir0 = ho * S - P

        acc = tl.zeros([BLOCK_W], dtype=tl.float32)

        for kh in tl.static_range(KH):
            ir = ir0 + kh
            # clamp the row used for ADDRESS only; values are masked by row_ok
            rc = tl.minimum(tl.maximum(ir, 0), H - 1)
            row_ok = (ir >= 0) & (ir < H)
            for kw in tl.static_range(KW):
                sh = kw - P
                # in_col = S * wo + sh = S * (wo + sh2) + shq  (shq in [0, S))
                if S == 1:
                    sh2 = sh
                    shq = 0
                elif S == 2:
                    sh2 = sh >> 1
                    shq = sh & 1
                else:
                    if sh < 0:
                        sh2 = -(((-sh) + S - 1) // S)
                    else:
                        sh2 = sh // S
                    shq = sh - sh2 * S
                tap_base = chan_base + shq * (H * WS)
                # contiguous, clamped in-bounds address; values masked
                colc = tl.minimum(tl.maximum(wo0 + sh2 + off, 0), WS - 1)
                src = tl.load(
                    x_ptr + tap_base + rc * WS + colc,
                    mask=row_ok & w_ok,
                    other=0.0,
                )
                t = kh * KW + kw
                w_t = tl.sum(tl.where(kk == t, wrow, 0.0))
                acc += src * w_t

        if HAS_BIAS:
            bias_v = tl.sum(tl.where(kk == KH * KW, wrow, 0.0))
            acc += bias_v

        tl.store(out_ptr + row * W_OUTS + wo0 + off, acc, mask=w_ok)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._convs = {}
        self._vec_core_num = _get_vec_core_num()

    def forward(self, inputs) -> torch.Tensor:
        x, in_channels, kernel_size, stride, padding, bias = inputs

        key = (in_channels, kernel_size, stride, padding, bias)
        conv = self._convs.get(key)
        if conv is None:
            conv = _build_conv(key, x.device)
            self._convs[key] = conv

        N, C, H, W = x.shape
        if not x.is_contiguous():
            x = x.contiguous()
        k = kernel_size
        s = stride
        p = padding
        H_OUT = (H + 2 * p - (k - 1) - 1) // s + 1
        W_OUT = (W + 2 * p - (k - 1) - 1) // s + 1

        weight = conv.weight.reshape(C, k, k)  # (C, KH*KW) contiguous
        bias_t = conv.bias if conv.bias is not None else weight

        out = torch.empty((N, C, H_OUT, W_OUT), dtype=x.dtype, device=x.device)

        # Choose a W-tile width: power of 2, at least 16, capped at 128.
        block_w = triton.next_power_of_2(W_OUT)
        if block_w < 16:
            block_w = 16
        if block_w > 128:
            block_w = 128

        # flatten (n, c, ho) rows; each block covers one (n,c,ho) row-tile over W
        n_blocks_row = N * C * H_OUT
        nw_tiles = triton.cdiv(W_OUT, block_w)
        num_blocks = n_blocks_row * nw_tiles

        grid_size = num_blocks if num_blocks < self._vec_core_num else self._vec_core_num
        grid = (grid_size,)
        _depthwise_conv2d_kernel[grid](
            x,
            weight,
            bias_t,
            out,
            C,
            H,
            W,
            H_OUT,
            W_OUT,
            s,
            p,
            num_blocks,
            self._vec_core_num,
            conv.bias is not None,
            k,
            k,
            block_w,
        )
        return out
