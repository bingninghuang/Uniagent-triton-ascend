import torch
import torch.nn as nn
import torch_npu
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Implicit-GEMM kernel for "standard 3D conv" of KernelBench problem 8.
#
# The reference builds nn.Conv3d with kernel_size=(K, K, 1), so the kernel
# has spatial extent only along D and H; the W axis is a pointwise product.
# The conv is expressed as, for every (n, w_out) slice,
#     A[M, K] x B[K, N] -> C[M, N]
# with
#   M  = Do * Ho            (flattened output spatial positions)
#   K  = (C_in / G) * (KD*KH)
#   N  = C_out / G          (per group)
#   A[m, k] = x[n, ci, di*sd + kd*dil - pad, ho*sh + kh*dil - pad, wo*sw - pad]
#   B[k, n] = weight[co, ci, kd, kh, 0]
# All loads are masked by the padded boundaries (missing taps contribute 0),
# and the stored tile is bias + A @ B.
# ---------------------------------------------------------------------------


@triton.jit
def _conv3d_implicit_gemm(
    # pointers
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    # runtime shape / conv parameters
    n_im,
    c_in,
    c_out,
    d_in,
    h_in,
    w_in,
    do_out,
    ho_out,
    wo_out,
    kd,
    kh,
    sd,
    sh,
    sw,
    pad,
    dil,
    taps,
    cig,
    cog,
    grps,
    nbg,
    m_tot,
    k_tot,
    nnb,
    per_n,
    # compile-time tiling parameters
    NUM_CORES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    hw_in = h_in * w_in
    dhw_in = d_in * h_in * w_in
    m_ww = m_tot * wo_out

    offs_bm = tl.arange(0, BM).to(tl.int32)
    offs_bn = tl.arange(0, BN).to(tl.int32)
    offs_bk = tl.arange(0, BK).to(tl.int32)

    for item in range(pid, n_im * wo_out * tl.cdiv(m_tot, BM) * nnb, NUM_CORES):
        # decompose item id -> (n, wo, m-block, n-block)
        r = item
        n = r // per_n
        r = r - n * per_n
        wo = r // (tl.cdiv(m_tot, BM) * nnb)
        r = r - wo * (tl.cdiv(m_tot, BM) * nnb)
        mb = r // nnb
        nb = r - mb * nnb
        g = nb // nbg
        cblk = nb - g * nbg

        co0 = g * cog + cblk * BN
        m0 = mb * BM

        offs_m = (m0 + offs_bm).to(tl.int32)
        di = offs_m // ho_out
        ho = offs_m - di * ho_out
        d_base = di * sd - pad
        h_base = ho * sh - pad
        w_off = wo * sw - pad
        m_valid = offs_m < m_tot

        # A base: (n * C_in + g * CIG) * DHW + w_off
        a_base = ((n * c_in + g * cig) * dhw_in + w_off).to(tl.int32)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, k_tot, BK):
            k = (kb + offs_bk).to(tl.int32)
            ci = k // taps
            t = k - ci * taps
            td = t // kh
            th = t - td * kh

            d_pos = d_base[:, None] + (td * dil)[None, :]
            h_pos = h_base[:, None] + (th * dil)[None, :]
            k_valid = k < k_tot
            w_valid = (w_off >= 0) & (w_off < w_in)
            da = (d_pos >= 0) & (d_pos < d_in)
            hb = (h_pos >= 0) & (h_pos < h_in)
            ma = (da & hb) & k_valid[None, :]
            ma = ma & m_valid[:, None]
            ma = ma & w_valid

            a_off = (
                a_base
                + ci[None, :] * dhw_in
                + d_pos * hw_in
                + h_pos * w_in
            )
            a = tl.load(x_ptr + a_off, mask=ma, other=0.0).to(tl.float32)

            co = (co0 + offs_bn).to(tl.int32)
            co_valid = co < c_out
            tap_pos = td * kh + th
            b_off = (
                (co[None, :] * c_in + g * cig + ci[:, None]) * taps
                + tap_pos[:, None]
            )
            mbk = k_valid[:, None] & co_valid[None, :]
            b = tl.load(w_ptr + b_off, mask=mbk, other=0.0).to(tl.float32)

            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        if HAS_BIAS:
            co = (co0 + offs_bn).to(tl.int32)
            co_valid = co < c_out
            bias_v = tl.load(b_ptr + co, mask=co_valid, other=0.0).to(tl.float32)
            acc = acc + bias_v[None, :]

        # output offset: flat index of out[n, co, di, ho, wo]
        base_o = (((n * c_out) + co0) * m_tot) * wo_out + wo
        o_off = base_o + offs_m[:, None] * wo_out + offs_bn[None, :] * m_ww
        mo = m_valid[:, None]
        moc = (co0 + offs_bn) < c_out
        mo_out = mo & moc[None, :]
        tl.store(o_ptr + o_off, acc.to(o_ptr.dtype.element_ty), mask=mo_out)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        try:
            dev = 0
            try:
                dev = torch_npu.npu.current_device()
            except Exception:
                dev = 0
            limit = torch_npu.npu.npu_config.get_device_limit(dev)
            self.num_cores = int(limit.get("cube_core_num", limit.get("vector_core_num", 24)))
        except Exception:
            self.num_cores = 24
        self._convs = {}

    def forward(self, inputs) -> torch.Tensor:
        x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = inputs

        key = (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        conv = self._convs.get(key)
        if conv is None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size, 1),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
            torch.set_rng_state(rng_state)
            conv = conv.to(x.device, x.dtype)
            self._convs[key] = conv

        weight = conv.weight
        bias_vec = conv.bias

        # x may arrive non-contiguous
        xc = x if x.is_contiguous() else x.contiguous()

        n_b, c_in, d_in, h_in, w_in = xc.shape
        c_out = out_channels
        kd, kh = kernel_size, kernel_size
        sd = sh = sw = stride
        pad = padding
        dil = dilation

        do_out = (d_in + 2 * pad - dil * (kd - 1) - 1) // sd + 1
        ho_out = (h_in + 2 * pad - dil * (kh - 1) - 1) // sh + 1
        wo_out = (w_in + 2 * pad - dil * (1 - 1) - 1) // sw + 1

        out = torch.empty((n_b, c_out, do_out, ho_out, wo_out), device=xc.device, dtype=xc.dtype)

        grps = groups
        cig = c_in // grps
        cog = c_out // grps
        taps = kd * kh
        k_tot = cig * taps
        m_tot = do_out * ho_out

        # ------------------------- tiling heuristic -------------------------
        BN = 16
        while BN < cog and BN < 128:
            BN *= 2
        nbg = triton.cdiv(cog, BN)
        base_items = n_b * wo_out * grps * nbg
        nmb = triton.cdiv(48, base_items)
        nmb = max(1, min(nmb, triton.cdiv(m_tot, 16)))
        BM = triton.next_power_of_2(triton.cdiv(m_tot, nmb))
        if BM < 16:
            BM = 16
        if BM > 128:
            BM = 128
        BK = 64 if k_tot >= 256 else 32

        num_programs = self.num_cores
        grid = (num_programs,)

        if bias_vec is None:
            bias_arg = weight  # unused placeholder when HAS_BIAS is 0
            has_bias = 0
        else:
            bias_arg = bias_vec
            has_bias = 1

        _conv3d_implicit_gemm[grid](
            xc,
            weight,
            bias_arg,
            out,
            n_b, c_in, c_out, d_in, h_in, w_in, do_out, ho_out, wo_out,
            kd, kh, sd, sh, sw, pad, dil, taps, cig, cog, grps, nbg,
            m_tot, k_tot,
            grps * nbg,
            wo_out * triton.cdiv(m_tot, BM) * (grps * nbg),
            num_programs, BM, BN, BK, has_bias,
        )
        return out
