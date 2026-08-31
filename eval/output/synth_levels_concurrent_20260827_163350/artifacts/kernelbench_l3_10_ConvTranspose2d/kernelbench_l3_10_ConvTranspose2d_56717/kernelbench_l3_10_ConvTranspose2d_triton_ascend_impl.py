import os
import sys
import traceback

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch_npu

_DIAG_DONE = False


def _diag(x):
    global _DIAG_DONE
    if _DIAG_DONE:
        return
    _DIAG_DONE = True
    lines = []
    try:
        lines.append("x.device=%s" % str(x.device))
        lines.append("npu.is_available=%s" % (torch_npu.npu.is_available()))
        try:
            lines.append("device_count=%s" % (torch_npu.npu.device_count()))
        except Exception as e:
            lines.append("device_count_err=%r" % (e,))
        try:
            lim = torch_npu.npu.npu_config.get_device_limit(0)
            lines.append("get_device_limit(0)=%r" % (lim,))
        except Exception as e:
            lines.append("get_device_limit_err=%r" % (e,))
        for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_SLOG_PRINT_TO_STDOUT", "TRITON_ASCEND_SOC_VERSION", "SOC_VERSION", "ASCEND_HOME", "CANN_PACKAGE_DIR"):
            lines.append("env %s=%r" % (k, os.environ.get(k)))
        try:
            t = torch.ones(4).to("npu:0")
            lines.append("probe npu:0 OK: %r device=%s" % ((t * 2), str(t.device)))
        except Exception as e:
            lines.append("probe_npu0_err=%r" % (e,))
            lines.append(traceback.format_exc(limit=5))
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(0)
            lines.append("triton_dev_props=%r" % (props,))
        except Exception as e:
            lines.append("triton_dev_props_err=%r" % (e,))
    except Exception as e:
        lines.append("diag_err=%r" % (e,))
        lines.append(traceback.format_exc(limit=5))
    try:
        with open("/opt/workspace_card0/agent_workdir/output/diag.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        sys.stderr.write("diag write failed: %r\n" % (e,))


@triton.jit
def _conv_t2d_dot_kernel(
    X, W, B, Y,
    # input shape: [N, Cin, Hin, Win]
    Cin, Hin, Win,
    # output shape: [N, Cout, Hout, Wout]
    Cout, Hout, Wout, Mw,
    # kernel / stride / padding (groups=1, dilation=1, output_padding=0)
    Kh, Kw, sh, sw, ph, pw,
    # number of tiles and total tiles
    num_oc_tiles, num_sp_tiles, n_tiles,
    # strides
    sxn, sxc, sxh, sxw,
    sy_n, sy_c, sy_h, sy_w,
    # fixed core count
    num_cores: tl.constexpr,
    BM: tl.constexpr,
    BOC: tl.constexpr,
    BC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    sp_base = tl.arange(0, BM).to(tl.int32)
    oc_base = tl.arange(0, BOC).to(tl.int32)
    k_base = tl.arange(0, BC).to(tl.int32)

    w_oc_stride = Kh * Kw
    w_ic_stride = Cout * Kh * Kw

    for tile_idx in range(pid, n_tiles, num_cores):
        n_b = tile_idx // (num_oc_tiles * num_sp_tiles)
        rem = tile_idx - n_b * num_oc_tiles * num_sp_tiles
        oc_tile = rem // num_sp_tiles
        sp_tile = rem - oc_tile * num_sp_tiles

        sp_offs = (sp_tile * BM + sp_base)
        sp_mask = sp_offs < Mw
        oh = sp_offs // Wout
        ow = sp_offs - oh * Wout

        oc_offs = oc_tile * BOC + oc_base
        oc_mask = oc_offs < Cout

        acc = tl.zeros((BM, BOC), dtype=tl.float32)

        for kh in range(0, Kh):
            for kw in range(0, Kw):
                th = oh + ph - kh
                tw = ow + pw - kw
                dh = th % sh
                dw = tw % sw
                div_ok = (dh == 0) & (dw == 0)
                ih = th // sh
                iw = tw // sw
                in_ok = sp_mask & div_ok & (ih >= 0) & (ih < Hin) & (iw >= 0) & (iw < Win)
                ih_c = tl.minimum(tl.maximum(ih, 0), Hin - 1)
                iw_c = tl.minimum(tl.maximum(iw, 0), Win - 1)

                x_row = n_b * sxn + ih_c * sxh + iw_c * sxw
                w_const_off = kh * Kw + kw

                for ic0 in range(0, Cin, BC):
                    x_ptrs = x_row[:, None] + k_base[None, :] * sxc
                    x_m = in_ok[:, None] & (k_base < Cin - ic0)[None, :]
                    x_t = tl.load(X + x_ptrs, mask=x_m, other=0.0).to(tl.float32)

                    w_ptrs = (ic0 + k_base[:, None]) * w_ic_stride + oc_offs[None, :] * w_oc_stride + w_const_off
                    w_m = (k_base < Cin - ic0)[:, None] & oc_mask[None, :]
                    w_t = tl.load(W + w_ptrs, mask=w_m, other=0.0).to(tl.float32)

                    acc = tl.dot(x_t, w_t, acc)

        if HAS_BIAS:
            b = tl.load(B + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
            acc = acc + b[None, :]

        out_ptrs = n_b * sy_n + oc_offs[None, :] * sy_c + oh[:, None] * sy_h + ow[:, None] * sy_w
        out_mask = sp_mask[:, None] & oc_mask[None, :]
        tl.store(Y + out_ptrs, acc.to(Y.dtype.element_ty), mask=out_mask)


def _get_core_num():
    # 910b1: 24 AI cores (each = 1 CUBE + 2 VEC) -> 48 VEC total.
    # Return the AI core count (== CUBE count).
    try:
        d = torch_npu.npu.npu_config.get_device_limit(0)
        if isinstance(d, dict):
            for k in ("cube_core_num", "ai_core_num", "core_num", "aicore_num"):
                v = d.get(k)
                if v is not None and int(v) > 0:
                    return int(v)
            vc = d.get("vector_core_num") or d.get("vector_core_cnt") or d.get("vec_core_num")
            if vc is not None and int(vc) > 0:
                return int(vc) // 2
    except Exception:
        pass
    return 24


def _create_conv(key, device):
    in_channels, out_channels, kernel_size, stride, padding, bias = key
    rng_state = torch.get_rng_state()
    torch.manual_seed(hash(key) & 0xFFFFFFFF)
    conv = nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    torch.set_rng_state(rng_state)
    return conv.to(device)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._convs = {}
        self._core_num = _get_core_num()

    def forward(self, inputs) -> torch.Tensor:
        x, in_channels, out_channels, kernel_size, stride, padding, bias = inputs
        _diag(x)

        key = (in_channels, out_channels, kernel_size, stride, padding, bias)
        conv = self._convs.get(key)
        if conv is None:
            conv = _create_conv(key, x.device)
            self._convs[key] = conv

        x = x.contiguous()
        w = conv.weight.contiguous()
        b = conv.bias

        N, Cin, Hin, Win = x.shape
        Cout = w.shape[1]
        Kh, Kw = int(w.shape[2]), int(w.shape[3])
        sh = int(conv.stride[0])
        sw = int(conv.stride[1])
        ph = int(conv.padding[0])
        pw = int(conv.padding[1])

        Hout = (Hin - 1) * sh + Kh - 2 * ph + 1
        Wout = (Win - 1) * sw + Kw - 2 * pw + 1
        Mw = Hout * Wout
        out = torch.empty((N, Cout, Hout, Wout), device=x.device, dtype=x.dtype)

        # ---- block size heuristics ----
        p2c = triton.next_power_of_2(Cout)
        if p2c > 64:
            BOC = 64
        else:
            BOC = p2c
        if BOC < 16:
            BOC = 16
        p2in = triton.next_power_of_2(Cin)
        if p2in > 128:
            BC = 128
        else:
            BC = p2in
        if BC < 16:
            BC = 16

        num_sp_tiles = triton.cdiv(Mw, 32)
        num_oc_tiles = triton.cdiv(Cout, BOC)
        n_tiles = N * num_oc_tiles * num_sp_tiles
        num_cores = self._core_num
        if n_tiles > num_cores:
            grid_n = num_cores
        else:
            grid_n = n_tiles
        if grid_n < 1:
            grid_n = 1
        grid = (grid_n,)

        b_arg = b if b is not None else x

        _conv_t2d_dot_kernel[grid](
            x, w, b_arg, out,
            Cin, Hin, Win,
            Cout, Hout, Wout, Mw,
            Kh, Kw, sh, sw, ph, pw,
            num_oc_tiles, num_sp_tiles, n_tiles,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            num_cores=num_cores,
            BM=32,
            BOC=BOC,
            BC=BC,
            HAS_BIAS=(b is not None),
        )
        return out
