import torch
import torch.nn as nn
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _dwconv2d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C, H, W, Ho, Wo,
    SH, SW, PH, PW,
    KH: tl.constexpr, KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # 2D-tiled depthwise conv2d (dilation = 1).
    # Each program handles a BLOCK_H x BLOCK_W tile of the output for one (b, c).
    pid = tl.program_id(0).to(tl.int32)
    tiles_h = tl.cdiv(Ho, BLOCK_H)
    tiles_w = tl.cdiv(Wo, BLOCK_W)
    tiles_per_plane = tiles_h * tiles_w
    total_tiles = B * C * tiles_per_plane

    for t in range(pid, total_tiles, NUM_CORES):
        plane = t // tiles_per_plane
        rem = t - plane * tiles_per_plane
        th = rem // tiles_w
        tw = rem - th * tiles_w
        b = plane // C
        c = plane - b * C

        offs_h = (th * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int32)
        offs_w = (tw * BLOCK_W + tl.arange(0, BLOCK_W)).to(tl.int32)
        mask_h = offs_h < Ho
        mask_w = offs_w < Wo
        mask_out = mask_h[:, None] & mask_w[None, :]

        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        base_plane = b * C + c

        for i in range(KH):
            irow = offs_h * SH - PH + i
            mrow = (irow >= 0) & (irow < H)
            irow2d = irow[:, None]
            mrow2d = mrow[:, None]
            for j in range(KW):
                jcol = offs_w * SW - PW + j
                mcol = (jcol >= 0) & (jcol < W)
                mask = mask_out & mrow2d & mcol[None, :]
                x_off = (base_plane * H + irow2d) * W + jcol[None, :]
                xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
                wv = tl.load(w_ptr + c * (KH * KW) + i * KW + j)
                acc += xv * wv

        if HAS_BIAS:
            bv = tl.load(b_ptr + c)
            acc = acc + bv

        y_off = (base_plane * Ho + offs_h[:, None]) * Wo + offs_w[None, :]
        tl.store(y_ptr + y_off, acc.to(y_ptr.dtype.element_ty), mask=mask_out)


_TILES = [(32, 32), (32, 16), (16, 32), (16, 16), (8, 16), (16, 8), (8, 8), (4, 8), (8, 4), (4, 4), (2, 4), (4, 2), (2, 2)]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._convs = {}
        try:
            limit = torch_npu.npu.npu_config.get_device_limit(0)
            self.VEC_CORE_NUM = int(limit.get('vector_core_num', 48))
            self.CUBE_CORE_NUM = int(limit.get('cube_core_num', 24))
        except Exception:
            self.VEC_CORE_NUM = 48
            self.CUBE_CORE_NUM = 24

    def _get_conv(self, key, device):
        """Build (and cache) the depthwise conv on the host side.

        The reference model seeds the RNG with ``hash(key) & 0xFFFFFFFF`` and
        builds an identical ``nn.Conv2d``; replicating that here yields the
        exact same weight / bias bits.
        """
        conv = self._convs.get(key)
        if conv is None:
            in_channels, kernel_size, stride, padding, has_bias = key
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=has_bias,
            )
            torch.set_rng_state(rng_state)
            conv = conv.to(device)
            self._convs[key] = conv
        return conv

    def _choose_tile(self, B, C, Ho, Wo):
        """Return (BLOCK_H, BLOCK_W, grid_size): grid uses ~all vector cores."""
        cores = self.VEC_CORE_NUM
        best = (8, 8)
        best_score = None
        for bh, bw in _TILES:
            if bh * bw > B * C * Ho * Wo:
                continue
            tiles = B * C * triton.cdiv(Ho, bh) * triton.cdiv(Wo, bw)
            util = 1.0 if tiles >= cores else float(cores) / float(tiles)
            score = (util, bh * bw, -tiles)
            if best_score is None or score > best_score:
                best_score = score
                best = (bh, bw)
        bh, bw = best
        total_tiles = B * C * triton.cdiv(Ho, bh) * triton.cdiv(Wo, bw)
        grid = cores if total_tiles > cores else total_tiles
        return bh, bw, grid

    def forward(self, inputs) -> torch.Tensor:
        x, in_channels, kernel_size, stride, padding, has_bias = inputs

        key = (in_channels, kernel_size, stride, padding, has_bias)
        conv = self._get_conv(key, x.device)

        x = x.contiguous()
        B, C, H, W = x.shape
        K = int(kernel_size)
        S = int(stride)
        P = int(padding)
        Ho = (H + 2 * P - K) // S + 1
        Wo = (W + 2 * P - K) // S + 1

        y = torch.empty((B, C, Ho, Wo), device=x.device, dtype=x.dtype)
        w = conv.weight
        b = conv.bias if has_bias else w

        BH, BW, grid0 = self._choose_tile(B, C, Ho, Wo)

        _dwconv2d_kernel[(grid0,)](
            x, w, b, y,
            B, C, H, W, Ho, Wo,
            S, S, P, P,
            KH=K, KW=K,
            HAS_BIAS=1 if has_bias else 0,
            BLOCK_H=BH, BLOCK_W=BW,
            NUM_CORES=grid0,
        )
        return y
