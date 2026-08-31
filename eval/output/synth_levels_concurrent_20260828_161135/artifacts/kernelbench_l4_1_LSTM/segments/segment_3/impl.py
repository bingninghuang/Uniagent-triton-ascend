import torch
import triton
import triton.language as tl


@triton.jit
def _lstm_fwd_kernel(
    x_ptr, w_ih_ptr, w_hh_ptr, bias_ih_ptr, bias_hh_ptr,
    h0_ptr, c0_ptr, out_ptr, hn_ptr, cn_ptr, hstate_ptr,
    S, B, I,
    x_sb, x_st,
    o_sb, o_st,
    num_cores: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    H: tl.constexpr,
    H4: tl.constexpr,
    BKH: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
    O_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_blocks = tl.cdiv(B, BLOCK_B)
    offs_h = tl.arange(0, H).to(tl.int32)
    offs_i0 = tl.arange(0, BLOCK_I).to(tl.int32)

    for bidx in range(pid, num_blocks, num_cores):
        offs_b = (bidx * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int32)
        mask_b = offs_b < B
        mask_b2 = mask_b[:, None]
        bbase = offs_b * H

        # states: c kept in registers (fp32); h kept in a scratch buffer
        # (fp32) so it can be re-read in chunks for the h-part GEMM.
        h0v = tl.load(h0_ptr + bbase[:, None] + offs_h[None, :],
                      mask=mask_b2, other=0.0).to(tl.float32)
        tl.store(hstate_ptr + bbase[:, None] + offs_h[None, :], h0v, mask=mask_b2)
        c = tl.load(c0_ptr + bbase[:, None] + offs_h[None, :],
                    mask=mask_b2, other=0.0).to(tl.float32)

        # pre-computed per-gate bias sums (fp32), constant across timesteps
        if HAS_BIAS:
            b0 = tl.load(bias_ih_ptr + offs_h).to(tl.float32) \
                + tl.load(bias_hh_ptr + offs_h).to(tl.float32)
            b1 = tl.load(bias_ih_ptr + (H + offs_h)).to(tl.float32) \
                + tl.load(bias_hh_ptr + (H + offs_h)).to(tl.float32)
            b2 = tl.load(bias_ih_ptr + (2 * H + offs_h)).to(tl.float32) \
                + tl.load(bias_hh_ptr + (2 * H + offs_h)).to(tl.float32)
            b3 = tl.load(bias_ih_ptr + (3 * H + offs_h)).to(tl.float32) \
                + tl.load(bias_hh_ptr + (3 * H + offs_h)).to(tl.float32)

        h_new = tl.zeros((BLOCK_B, H), dtype=tl.float32)
        for t in range(S):
            acc0 = tl.zeros((BLOCK_B, H), dtype=tl.float32)
            acc1 = tl.zeros((BLOCK_B, H), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_B, H), dtype=tl.float32)
            acc3 = tl.zeros((BLOCK_B, H), dtype=tl.float32)

            # ---- x_t @ W_ih^T part (native-dtype dot, fp32 accumulate) ----
            for i0 in range(0, I, BLOCK_I):
                offs_i = i0 + offs_i0
                mi = (offs_i[None, :] < I)
                xt = tl.load(x_ptr + offs_b[:, None] * x_sb + t * x_st + offs_i[None, :],
                             mask=mask_b2 & mi, other=0.0)
                w0 = tl.load(w_ih_ptr + offs_h[:, None] * I + offs_i[None, :], mask=mi, other=0.0)
                w1 = tl.load(w_ih_ptr + (H + offs_h)[:, None] * I + offs_i[None, :], mask=mi, other=0.0)
                w2 = tl.load(w_ih_ptr + (2 * H + offs_h)[:, None] * I + offs_i[None, :], mask=mi, other=0.0)
                w3 = tl.load(w_ih_ptr + (3 * H + offs_h)[:, None] * I + offs_i[None, :], mask=mi, other=0.0)
                acc0 = tl.dot(xt, tl.trans(w0), acc0)
                acc1 = tl.dot(xt, tl.trans(w1), acc1)
                acc2 = tl.dot(xt, tl.trans(w2), acc2)
                acc3 = tl.dot(xt, tl.trans(w3), acc3)

            # ---- h @ W_hh^T part (h re-read from scratch in K-chunks, fp32 dot) ----
            for k0 in range(0, H, BKH):
                offs_k = (k0 + tl.arange(0, BKH)).to(tl.int32)
                hk = tl.load(hstate_ptr + bbase[:, None] + offs_k[None, :],
                             mask=mask_b2, other=0.0)
                w0 = tl.load(w_hh_ptr + offs_k[:, None] * H + offs_h[None, :]).to(tl.float32)
                w1 = tl.load(w_hh_ptr + offs_k[:, None] * H + (H + offs_h)[None, :]).to(tl.float32)
                w2 = tl.load(w_hh_ptr + offs_k[:, None] * H + (2 * H + offs_h)[None, :]).to(tl.float32)
                w3 = tl.load(w_hh_ptr + offs_k[:, None] * H + (3 * H + offs_h)[None, :]).to(tl.float32)
                acc0 = tl.dot(hk, w0, acc0)
                acc1 = tl.dot(hk, w1, acc1)
                acc2 = tl.dot(hk, w2, acc2)
                acc3 = tl.dot(hk, w3, acc3)

            # ---- bias ----
            if HAS_BIAS:
                acc0 = acc0 + b0[None, :]
                acc1 = acc1 + b1[None, :]
                acc2 = acc2 + b2[None, :]
                acc3 = acc3 + b3[None, :]

            # ---- gates: sigmoid via 1/(1+e^-x); tanh(x) = 2*sigmoid(2x) - 1 ----
            gi = 1.0 / (1.0 + tl.exp(-acc0))
            gf = 1.0 / (1.0 + tl.exp(-acc1))
            gg = 2.0 / (1.0 + tl.exp(-2.0 * acc2)) - 1.0
            go = 1.0 / (1.0 + tl.exp(-acc3))
            c_new = gf * c + gi * gg
            h_new = go * (2.0 / (1.0 + tl.exp(-2.0 * c_new)) - 1.0)

            # ---- store h_t (output dtype) and update states ----
            out_off = offs_b[:, None] * o_sb + t * o_st + offs_h[None, :]
            tl.store(out_ptr + out_off, h_new.to(O_DTYPE), mask=mask_b2)
            tl.store(hstate_ptr + bbase[:, None] + offs_h[None, :], h_new, mask=mask_b2)
            c = c_new

        # ---- final h_n, c_n ----
        tl.store(hn_ptr + bbase[:, None] + offs_h[None, :], h_new.to(O_DTYPE), mask=mask_b2)
        tl.store(cn_ptr + bbase[:, None] + offs_h[None, :], c.to(O_DTYPE), mask=mask_b2)


def _next_pow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


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
        H = weight_hh_l0.shape[1]
        H4 = weight_ih_l0.shape[0]

        if batch_first:
            B, S, I = x.shape
            x_sb, x_st = S * I, I
            out = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
            o_sb, o_st = S * H, H
        else:
            S, B, I = x.shape
            x_sb, x_st = I, B * I
            out = torch.empty((S, B, H), device=x.device, dtype=x.dtype)
            o_sb, o_st = H, B * H

        has_bias = bias_ih_l0 is not None and bias_hh_l0 is not None
        if has_bias:
            bi = bias_ih_l0
            bh = bias_hh_l0
        else:
            bi = x  # dummy, not read
            bh = x

        hn = torch.empty((1, B, H), device=x.device, dtype=x.dtype)
        cn = torch.empty((1, B, H), device=x.device, dtype=x.dtype)
        hstate = torch.empty((B, H), device=x.device, dtype=torch.float32)

        # K-chunk size for the h-part GEMM (rows of h / cols of W_hh)
        if H <= 128:
            bkh = H
        elif H <= 256:
            bkh = 64 if H % 64 == 0 else 32
        else:
            bkh = 32

        block_b = 16
        block_i = max(16, min(_next_pow2(I), 128))
        num_blocks = (B + block_b - 1) // block_b
        grid = (num_blocks if num_blocks < self.CUBE_CORE_NUM else self.CUBE_CORE_NUM,)
        if x.dtype == torch.float16:
            o_dtype = tl.float16
        elif x.dtype == torch.bfloat16:
            o_dtype = tl.bfloat16
        else:
            o_dtype = tl.float32

        _lstm_fwd_kernel[grid](
            x, weight_ih_l0, weight_hh_l0, bi, bh,
            h_0, c_0, out, hn, cn, hstate,
            S, B, I,
            x_sb, x_st,
            o_sb, o_st,
            num_cores=self.CUBE_CORE_NUM,
            HAS_BIAS=has_bias,
            H=H,
            H4=H4,
            BKH=bkh,
            BLOCK_B=block_b,
            BLOCK_I=block_i,
            O_DTYPE=o_dtype,
        )
        out_ret = out.transpose(0, 1) if batch_first else out
        return out_ret, (hn, cn)
