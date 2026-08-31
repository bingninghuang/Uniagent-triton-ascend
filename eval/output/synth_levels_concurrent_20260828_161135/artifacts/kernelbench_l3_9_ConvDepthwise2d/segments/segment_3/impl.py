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
    return conv.to(device)


@triton.jit
def _depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    # input / output spatial dims
    C,
    H,
    W,
    H_OUT,
    W_OUT,
    # conv params
    STRIDE,
    PAD,
    NUM_BLOCKS,
    CORE_COUNT,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    nw_tiles = tl.cdiv(W_OUT, BLOCK_W)

    for bidx in range(pid, NUM_BLOCKS, CORE_COUNT):
        # bidx -> (row, wo_tile); row = (n * C + c), wo_tile covers [wo0, wo0+BLOCK_W)
        row = bidx // nw_tiles
        wt = bidx % nw_tiles
        ho = row % H_OUT
        nc = row // H_OUT  # n * C + c
        wo0 = wt * BLOCK_W

        off = tl.arange(0, BLOCK_W).to(tl.int32)
        wo = wo0 + off
        col_ok = wo < W_OUT

        in_row = ho * STRIDE - PAD
        in_col_base = wo0 * STRIDE - PAD

        x_base = x_ptr + nc.to(tl.int64) * (H * W)
        c_idx = nc % C
        w_base = w_ptr + c_idx * (KH * KW)

        acc = tl.zeros([BLOCK_W], dtype=tl.float32)

        for kh in range(0, KH):
            in_row_k = in_row + kh
            row_valid = (in_row_k >= 0) & (in_row_k < H)
            for kw in range(0, KW):
                in_col = in_col_base + kw + off * STRIDE
                mask = col_ok & (in_col >= 0) & (in_col < W) & row_valid
                wval = tl.load(w_base + kh * KW + kw)
                patch = tl.load(
                    x_base + in_row_k.to(tl.int64) * W + in_col.to(tl.int64),
                    mask=mask,
                    other=0.0,
                )
                acc += patch * wval
        if HAS_BIAS:
            bias_val = tl.load(b_ptr + c_idx)
            acc += bias_val

        out_base = out_ptr + row.to(tl.int64) * W_OUT + wo0
        tl.store(out_base + off, acc, mask=col_ok)


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
        H_OUT = (H + 2 * p - s * (k - 1) - 1) // s + 1
        W_OUT = (W + 2 * p - s * (k - 1) - 1) // s + 1

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
