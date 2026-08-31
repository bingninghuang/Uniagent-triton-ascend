import torch
import triton
import triton.language as tl


@triton.jit
def _lstm_fwd_kernel(
    x_ptr, w_ih_ptr, w_hh_ptr, bias_ih_ptr, bias_hh_ptr,
    h0_ptr, c0_ptr, out_ptr, hn_ptr, cn_ptr,
    S, B, I,
    x_sb, x_st,
    o_sb, o_st,
    num_cores: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    H: tl.constexpr,
    H4: tl.constexpr,
    HC: tl.constexpr,
    HCHUNK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_blocks = tl.cdiv(B, BLOCK_B)
    offs_h = tl.arange(0, H).to(tl.int32)
    offs_h4 = tl.arange(0, H4).to(tl.int32)
    offs_hc = tl.arange(0, HC).to(tl.int32)

    for bidx in range(pid, num_blocks, num_cores):
        offs_b = (bidx * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int32)
        mask_b = offs_b < B
        mask_b2 = mask_b[:, None]

        # initial states (fp32)
        h = tl.load(h0_ptr + offs_b[:, None] * H + offs_h[None, :],
                    mask=mask_b2, other=0.0).to(tl.float32)
        c = tl.load(c0_ptr + offs_b[:, None] * H + offs_h[None, :],
                    mask=mask_b2, other=0.0).to(tl.float32)

        for t in range(S):
            acc = tl.zeros((BLOCK_B, H4), dtype=tl.float32)
            # ---- x_t @ W_ih^T part (native dtype dot, fp32 accumulate) ----
            for i0 in range(0, I, BLOCK_I):
                offs_i = (i0 + tl.arange(0, BLOCK_I)).to(tl.int32)
                x_tile = tl.load(x_ptr + offs_b[:, None] * x_sb + t * x_st + offs_i[None, :],
                                 mask=mask_b2 & (offs_i[None, :] < I), other=0.0)
                w_tile = tl.load(w_ih_ptr + offs_i[:, None] + offs_h4[None, :] * I,
                                 mask=(offs_i[:, None] < I), other=0.0)
                acc = tl.dot(x_tile, w_tile, acc)
            # ---- h @ W_hh^T part (fp32 dot) ----
            if HCHUNK == 1:
                wh = tl.load(w_hh_ptr + offs_h[:, None] * H4 + offs_h4[None, :]).to(tl.float32)
                acc = tl.dot(h, wh, acc)
            elif HCHUNK == 2:
                hc = H // 2
                offs_hc2 = tl.arange(0, hc).to(tl.int32)
                hl, hr = tl.split(h)
                whl = tl.load(w_hh_ptr + offs_hc2[:, None] * H4 + offs_h4[None, :]).to(tl.float32)
                whr = tl.load(w_hh_ptr + (hc + offs_hc2)[:, None] * H4 + offs_h4[None, :]).to(tl.float32)
                acc = tl.dot(hl, whl, acc)
                acc = tl.dot(hr, whr, acc)
            else:
                hc = H // 4
                offs_hc4 = tl.arange(0, hc).to(tl.int32)
                hl, hr = tl.split(h)
                h00, h01 = tl.split(hl)
                h10, h11 = tl.split(hr)
                wh0 = tl.load(w_hh_ptr + offs_hc4[:, None] * H4 + offs_h4[None, :]).to(tl.float32)
                wh1 = tl.load(w_hh_ptr + (hc + offs_hc4)[:, None] * H4 + offs_h4[None, :]).to(tl.float32)
                wh2 = tl.load(w_hh_ptr + (2 * hc + offs_hc4)[:, None] * H4 + offs_h4[None, :]).to(tl.float32)
                wh3 = tl.load(w_hh_ptr + (3 * hc + offs_hc4)[:, None] * H4 + offs_h4[None, :]).to(tl.float32)
                acc = tl.dot(h00, wh0, acc)
                acc = tl.dot(h01, wh1, acc)
                acc = tl.dot(h10, wh2, acc)
                acc = tl.dot(h11, wh3, acc)
            # ---- bias ----
            if HAS_BIAS:
                bi = tl.load(bias_ih_ptr + offs_h4).to(tl.float32)
                bh = tl.load(bias_hh_ptr + offs_h4).to(tl.float32)
                acc = acc + (bi + bh)[None, :]
            # ---- gate split: [i, f, g, o] each (BLOCK_B, H) ----
            gl, gr = tl.split(acc)          # gl: cols [0:2H) = i,f ; gr: cols [2H:4H) = g,o
            g_i, g_f = tl.split(gl)
            g_g, g_o = tl.split(gr)
            # ---- activations (tanh(x) = 2*sigmoid(2x) - 1) ----
            sig_i = tl.sigmoid(g_i)
            sig_f = tl.sigmoid(g_f)
            tanh_g = 2.0 * tl.sigmoid(2.0 * g_g) - 1.0
            sig_o = tl.sigmoid(g_o)
            c_new = sig_f * c + sig_i * tanh_g
            tanh_c = 2.0 * tl.sigmoid(2.0 * c_new) - 1.0
            h_new = sig_o * tanh_c
            # ---- store h_t (cast to output dtype), update states ----
            out_off = offs_b[:, None] * o_sb + t * o_st + offs_h[None, :]
            tl.store(out_ptr + out_off, h_new.to(out_ptr.dtype.element_ty), mask=mask_b2)
            h = h_new
            c = c_new

        # ---- final h_n, c_n ----
        hn_off = offs_b[:, None] * H + offs_h[None, :]
        tl.store(hn_ptr + hn_off, h.to(hn_ptr.dtype.element_ty), mask=mask_b2)
        tl.store(cn_ptr + hn_off, c.to(cn_ptr.dtype.element_ty), mask=mask_b2)


def _pick_block_b(B, hidden, num_cores):
    # keep accumulator (BLOCK_B, 4*H) fp32 <= ~32768 elements (UB safe)
    acc_cap = 32768 // (4 * hidden)
    acc_cap = max(16, acc_cap)
    # want grid (num batch blocks) ~= num_cores
    want = 16
    if B > num_cores * 16:
        want = triton.next_power_of_2(triton.cdiv(B, num_cores))
    bb = min(want, triton.next_power_of_2(B))
    # clamp to power of two <= acc_cap, at least 16
    while bb > acc_cap:
        bb //= 2
    while bb < 16:
        bb = 16
    return bb


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        try:
            import torch_npu
            info = torch_npu.npu.npu_config.get_device_limit(0)
            self.CUBE_CORE_NUM = max(1, int(info.get("cube_core_num", 24)))
        except Exception:
            self.CUBE_CORE_NUM = 24

    def forward(self, x: torch.Tensor, weight_ih_l0: torch.Tensor, weight_hh_l0: torch.Tensor,
                bias_ih_l0: torch.Tensor, bias_hh_l0: torch.Tensor,
                h_0: torch.Tensor, c_0: torch.Tensor,
                batch_first: bool = False, dropout: float = 0.0):
        if batch_first:
            B, S, I = x.shape
            x_sb, x_st = S * I, I
            out = torch.empty((B, S, weight_hh_l0.shape[1]), device=x.device, dtype=x.dtype)
            o_sb, o_st = S * weight_hh_l0.shape[1], weight_hh_l0.shape[1]
        else:
            S, B, I = x.shape
            x_sb, x_st = I, B * I
            out = torch.empty((S, B, weight_hh_l0.shape[1]), device=x.device, dtype=x.dtype)
            o_sb, o_st = weight_hh_l0.shape[1], B * weight_hh_l0.shape[1]
        H = weight_hh_l0.shape[1]
        H4 = weight_ih_l0.shape[0]

        x = x.contiguous()
        w_ih = weight_ih_l0.contiguous()
        w_hh = weight_hh_l0.contiguous()
        h0 = h_0.contiguous()
        c0 = c_0.contiguous()
        has_bias = bias_ih_l0 is not None and bias_hh_l0 is not None
        if has_bias:
            bi = bias_ih_l0.contiguous()
            bh = bias_hh_l0.contiguous()
        else:
            bi = x  # dummy, not read
            bh = x

        hn = torch.empty((1, B, H), device=x.device, dtype=x.dtype)
        cn = torch.empty((1, B, H), device=x.device, dtype=x.dtype)

        if H <= 128:
            hchunk = 1
            hc = H
        elif H <= 256:
            hchunk = 2
            hc = H // 2
        else:
            hchunk = 4
            hc = H // 4

        BLOCK_B = _pick_block_b(B, H, self.CUBE_CORE_NUM)
        BLOCK_I = 128
        num_blocks = triton.cdiv(B, BLOCK_B)
        grid = (min(num_blocks, self.CUBE_CORE_NUM),)

        _lstm_fwd_kernel[grid](
            x, w_ih, w_hh, bi, bh,
            h0, c0, out, hn, cn,
            S, B, I,
            x_sb, x_st,
            o_sb, o_st,
            num_cores=self.CUBE_CORE_NUM,
            HAS_BIAS=has_bias,
            H=H,
            H4=H4,
            HC=hc,
            HCHUNK=hchunk,
            BLOCK_B=BLOCK_B,
            BLOCK_I=BLOCK_I,
        )
        return out, (hn, cn)
