import torch
import torch.nn as nn
import triton
import triton.language as tl


def _to_tuple(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    r = int(v)
    return r, r


def _build_conv_entry(device, in_channels, out_channels,
                      kernel_size, stride, padding, has_bias, key):
    # Host-side weight generation (seeded identically to the reference so that
    # the convolution weights are bit-exact). Returns weight in
    # (Kh, Kw, Ci, Co) layout and bias (or None).
    rng_state = torch.get_rng_state()
    torch.manual_seed(hash(key) & 0xFFFFFFFF)
    conv = nn.ConvTranspose2d(
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=has_bias,
    )
    torch.set_rng_state(rng_state)
    w = conv.weight.to(device=device, dtype=torch.float32)
    b = conv.bias.to(device=device, dtype=torch.float32) if has_bias else None
    wt = w.permute(2, 3, 0, 1).contiguous()
    return (wt, b)


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, wt_ptr, bias_ptr, out_ptr,
    N, Ci, Hi, Wi, Co, Ho, Wo,
    sh, sw, ph, pw,
    NUM_BLOCKS,
    Kh: tl.constexpr, Kw: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SH: tl.constexpr, BLOCK_SW: tl.constexpr,
    BLOCK_CI: tl.constexpr, BLOCK_CO: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """
    Transposed 2D conv, groups=1, dilation=1:
      out[n, co, tu, tv] = bias[co] + sum_{ci, ju, jv}
          W[ci, co, ju, jv] * x[n, ci, (tu + ph - ju)//sh, (tv + pw - jv)//sw]
      where the tap contributes only when (tu+ph-ju) % sh == 0 and
      (tv+pw-jv) % sw == 0 and the derived input coords are in range.
    wt is preprocessed to layout (Kh, Kw, Ci, Co).
    """
    pid = tl.program_id(0).to(tl.int32)
    n_co = tl.cdiv(Co, BLOCK_CO).to(tl.int32)
    n_sh = tl.cdiv(Ho, BLOCK_SH).to(tl.int32)
    n_sw = tl.cdiv(Wo, BLOCK_SW).to(tl.int32)
    sp_cnt = n_sh * n_sw
    co_sp_cnt = n_co * sp_cnt

    SPAT: tl.constexpr = BLOCK_SH * BLOCK_SW
    sp = tl.arange(0, SPAT).to(tl.int32)
    ci_idx = tl.arange(0, BLOCK_CI).to(tl.int32)

    x_plane = Hi * Wi
    x_row = Ci * x_plane
    o_plane = Ho * Wo

    for blk in range(pid, NUM_BLOCKS, NUM_CORES):
        n = (blk // co_sp_cnt).to(tl.int32)
        rem = blk % co_sp_cnt
        co_b = (rem // sp_cnt).to(tl.int32)
        rem2 = rem % sp_cnt
        sh_b = (rem2 // n_sw).to(tl.int32)
        sw_b = (rem2 % n_sw).to(tl.int32)

        tu = (sh_b * BLOCK_SH + sp // BLOCK_SW).to(tl.int32)
        tv = (sw_b * BLOCK_SW + sp % BLOCK_SW).to(tl.int32)
        co_idx = (co_b * BLOCK_CO + tl.arange(0, BLOCK_CO).to(tl.int32))

        acc = tl.zeros((SPAT, BLOCK_CO), dtype=tl.float32)
        x_base = n * x_row

        for ci0 in range(0, Ci, BLOCK_CI):
            ci_c = ci0.to(tl.int32) + ci_idx
            ci_mask = ci_c < Ci
            for tid in range(0, Kh * Kw):
                ju = tid // Kw
                jv = tid % Kw
                nu = tu + ph - ju
                nv = tv + pw - jv
                iu = tl.where(nu >= 0, nu // sh, 0)
                iv = tl.where(nv >= 0, nv // sw, 0)
                vu = (nu >= 0) & ((nu % sh) == 0) & (iu < Hi)
                vv = (nv >= 0) & ((nv % sw) == 0) & (iv < Wi)
                valid = vu & vv
                a_off = x_base + ci_c[None, :] * x_plane + (iu * Wi + iv)[:, None]
                a = tl.load(x_ptr + a_off,
                            mask=valid[:, None] & ci_mask[None, :], other=0.0)
                b_off = tid * Ci * Co + ci_c[:, None] * Co + co_idx[None, :]
                b_mask = ci_mask[:, None] & (co_idx < Co)[None, :]
                b = tl.load(wt_ptr + b_off, mask=b_mask, other=0.0)
                acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        if HAS_BIAS:
            bv = tl.load(bias_ptr + co_idx, mask=co_idx < Co, other=0.0)
            acc = acc + bv[None, :]

        o_off = (n * Co * o_plane + co_idx[None, :] * o_plane
                 + (tu * Wo + tv)[:, None])
        o_mask = ((tu < Ho) & (tv < Wo))[:, None] & (co_idx < Co)[None, :]
        tl.store(out_ptr + o_off, acc.to(out_ptr.dtype.element_ty), mask=o_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._cache = {}
        try:
            import torch_npu
            lim = torch_npu.npu.npu_config.get_device_limit(0)
            core_num = int(lim.get("cube_core_num", 24))
        except Exception:
            core_num = 24
        if core_num <= 0:
            core_num = 24
        self.CUBE_CORE_NUM = core_num

    def forward(self, inputs) -> torch.Tensor:
        x, in_channels, out_channels, kernel_size, stride, padding, bias = inputs

        Kh, Kw = _to_tuple(kernel_size)
        sh, sw = _to_tuple(stride)
        ph, pw = _to_tuple(padding)
        has_bias = bool(bias)

        N, Ci, Hi, Wi = x.shape
        Co = int(out_channels)
        Ho = (Hi - 1) * sh - 2 * ph + (Kh - 1) + 1
        Wo = (Wi - 1) * sw - 2 * pw + (Kw - 1) + 1

        key = (in_channels, out_channels, kernel_size, stride, padding, has_bias)
        entry = self._cache.get(key)
        if entry is None:
            entry = _build_conv_entry(x.device, in_channels, Co,
                                      kernel_size, stride, padding,
                                      has_bias, key)
            self._cache[key] = entry
        wt, b = entry

        x_f32 = x if x.dtype == torch.float32 else x.to(torch.float32)
        if not x_f32.is_contiguous():
            x_f32 = x_f32.contiguous()

        out = torch.empty((N, Co, Ho, Wo), device=x.device, dtype=x.dtype)

        BLOCK_SH = 8
        BLOCK_SW = 8
        BLOCK_CI = 32
        BLOCK_CO = 32
        num_blocks = (N * triton.cdiv(Co, BLOCK_CO)
                      * triton.cdiv(Ho, BLOCK_SH)
                      * triton.cdiv(Wo, BLOCK_SW))
        grid_n = num_blocks if num_blocks < self.CUBE_CORE_NUM else self.CUBE_CORE_NUM
        grid = (grid_n,)
        dummy_bias = b if b is not None else wt
        conv_transpose2d_kernel[grid](
            x_f32, wt, dummy_bias, out,
            N, Ci, Hi, Wi, Co, Ho, Wo,
            sh, sw, ph, pw,
            num_blocks,
            Kh=Kh, Kw=Kw,
            HAS_BIAS=has_bias,
            BLOCK_SH=BLOCK_SH, BLOCK_SW=BLOCK_SW,
            BLOCK_CI=BLOCK_CI, BLOCK_CO=BLOCK_CO,
            NUM_CORES=self.CUBE_CORE_NUM,
        )
        return out
