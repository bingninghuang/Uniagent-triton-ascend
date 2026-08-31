import torch
import triton
import triton.language as tl


@triton.jit
def _lstm_fwd_kernel(
    x_ptr, w_ih_ptr, w_hh_ptr, bias_ih_ptr, bias_hh_ptr,
    h0_ptr, c0_ptr, out_ptr, hn_ptr, cn_ptr,
    hstate_ptr, cstate_ptr,
    S, B, I,
    x_sb, x_st,
    o_sb, o_st,
    num_cores: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    H: tl.constexpr,
    BKH: tl.constexpr,
    BO: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_I: tl.constexpr,
    O_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    num_blocks = tl.cdiv(B, BLOCK_B)
    offs_bo = tl.arange(0, BO).to(tl.int32)
    offs_bk = tl.arange(0, BKH).to(tl.int32)
    offs_i0 = tl.arange(0, BLOCK_I).to(tl.int32)

    for bidx in range(pid, num_blocks, num_cores):
        offs_b = (bidx * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int32)
        mask_b = offs_b < B
        mask_b2 = mask_b[:, None]
        bbase = offs_b * H

        # fp32 states: c in one buffer; h double-buffered (ping-pong per t)
        for hi in range(0, H, BO):
            oj = hi + offs_bo
            maskbh = mask_b2 & (oj < H)[None, :]
            h0v = tl.load(h0_ptr + bbase[:, None] + oj[None, :],
                          mask=maskbh, other=0.0).to(tl.float32)
            tl.store(hstate_ptr + bbase[:, None] + oj[None, :], h0v, mask=maskbh)
            c0v = tl.load(c0_ptr + bbase[:, None] + oj[None, :],
                          mask=maskbh, other=0.0).to(tl.float32)
            tl.store(cstate_ptr + bbase[:, None] + oj[None, :], c0v, mask=maskbh)

        for t in range(S):
            rd = (t & 1) * B * H
            wr = (1 - (t & 1)) * B * H
            for hi in range(0, H, BO):
                oj = hi + offs_bo
                maskbh = mask_b2 & (oj < H)[None, :]
                mwr = (oj < H)[:, None]
                acc0 = tl.zeros((BLOCK_B, BO), dtype=tl.float32)
                acc1 = tl.zeros((BLOCK_B, BO), dtype=tl.float32)
                acc2 = tl.zeros((BLOCK_B, BO), dtype=tl.float32)
                acc3 = tl.zeros((BLOCK_B, BO), dtype=tl.float32)

                # ---- x_t @ W_ih^T (native dot, fp32 acc) ----
                for i0 in range(0, I, BLOCK_I):
                    offs_i = i0 + offs_i0
                    mi = (offs_i[None, :] < I)
                    xt = tl.load(x_ptr + offs_b[:, None] * x_sb + t * x_st + offs_i[None, :],
                                 mask=mask_b2 & mi, other=0.0)
                    w0 = tl.load(w_ih_ptr + oj[:, None] * I + offs_i[None, :], mask=mwr & mi, other=0.0)
                    w1 = tl.load(w_ih_ptr + (H + oj)[:, None] * I + offs_i[None, :], mask=mwr & mi, other=0.0)
                    w2 = tl.load(w_ih_ptr + (2 * H + oj)[:, None] * I + offs_i[None, :], mask=mwr & mi, other=0.0)
                    w3 = tl.load(w_ih_ptr + (3 * H + oj)[:, None] * I + offs_i[None, :], mask=mwr & mi, other=0.0)
                    acc0 = tl.dot(xt, tl.trans(w0), acc0)
                    acc1 = tl.dot(xt, tl.trans(w1), acc1)
                    acc2 = tl.dot(xt, tl.trans(w2), acc2)
                    acc3 = tl.dot(xt, tl.trans(w3), acc3)

                # ---- h @ W_hh^T (fp32 dot). W_hh tile: rows = oj, cols = offs_k ----
                for k0 in range(0, H, BKH):
                    offs_k = k0 + offs_bk
                    mk = (offs_k < H)[None, :]
                    hk = tl.load(hstate_ptr + rd + bbase[:, None] + offs_k[None, :],
                                 mask=mask_b2 & mk, other=0.0)
                    w0 = tl.load(w_hh_ptr + oj[:, None] * H + offs_k[None, :], mask=mwr & mk, other=0.0).to(tl.float32)
                    w1 = tl.load(w_hh_ptr + (H + oj)[:, None] * H + offs_k[None, :], mask=mwr & mk, other=0.0).to(tl.float32)
                    w2 = tl.load(w_hh_ptr + (2 * H + oj)[:, None] * H + offs_k[None, :], mask=mwr & mk, other=0.0).to(tl.float32)
                    w3 = tl.load(w_hh_ptr + (3 * H + oj)[:, None] * H + offs_k[None, :], mask=mwr & mk, other=0.0).to(tl.float32)
                    acc0 = tl.dot(hk, tl.trans(w0), acc0)
                    acc1 = tl.dot(hk, tl.trans(w1), acc1)
                    acc2 = tl.dot(hk, tl.trans(w2), acc2)
                    acc3 = tl.dot(hk, tl.trans(w3), acc3)

                # ---- bias ----
                if HAS_BIAS:
                    b0 = tl.load(bias_ih_ptr + oj, mask=oj < H, other=0.0).to(tl.float32) \
                        + tl.load(bias_hh_ptr + oj, mask=oj < H, other=0.0).to(tl.float32)
                    b1 = tl.load(bias_ih_ptr + (H + oj), mask=oj < H, other=0.0).to(tl.float32) \
                        + tl.load(bias_hh_ptr + (H + oj), mask=oj < H, other=0.0).to(tl.float32)
                    b2 = tl.load(bias_ih_ptr + (2 * H + oj), mask=oj < H, other=0.0).to(tl.float32) \
                        + tl.load(bias_hh_ptr + (2 * H + oj), mask=oj < H, other=0.0).to(tl.float32)
                    b3 = tl.load(bias_ih_ptr + (3 * H + oj), mask=oj < H, other=0.0).to(tl.float32) \
                        + tl.load(bias_hh_ptr + (3 * H + oj), mask=oj < H, other=0.0).to(tl.float32)
                    acc0 = acc0 + b0[None, :]
                    acc1 = acc1 + b1[None, :]
                    acc2 = acc2 + b2[None, :]
                    acc3 = acc3 + b3[None, :]

                # ---- gates: sigmoid 1/(1+e^-x); tanh 2/(1+e^-2x)-1 ----
                gi = 1.0 / (1.0 + tl.exp(-acc0))
                gf = 1.0 / (1.0 + tl.exp(-acc1))
                gg = 2.0 / (1.0 + tl.exp(-2.0 * acc2)) - 1.0
                go = 1.0 / (1.0 + tl.exp(-acc3))
                c_old = tl.load(cstate_ptr + bbase[:, None] + oj[None, :],
                                mask=maskbh, other=0.0)
                c_new = gf * c_old + gi * gg
                tl.store(cstate_ptr + bbase[:, None] + oj[None, :], c_new, mask=maskbh)
                h_new = go * (2.0 / (1.0 + tl.exp(-2.0 * c_new)) - 1.0)

                # ---- store h_t (out dtype) and ping-pong h state ----
                ooff = offs_b[:, None] * o_sb + t * o_st + oj[None, :]
                tl.store(out_ptr + ooff, h_new.to(O_DTYPE), mask=maskbh)
                tl.store(hstate_ptr + wr + bbase[:, None] + oj[None, :], h_new, mask=maskbh)

        # ---- final h_n, c_n ----
        fin = (S & 1) * B * H
        for hi in range(0, H, BO):
            oj = hi + offs_bo
            maskbh = mask_b2 & (oj < H)[None, :]
            hv = tl.load(hstate_ptr + fin + bbase[:, None] + oj[None, :],
                         mask=maskbh, other=0.0)
            cv = tl.load(cstate_ptr + bbase[:, None] + oj[None, :],
                         mask=maskbh, other=0.0)
            tl.store(hn_ptr + bbase[:, None] + oj[None, :], hv.to(O_DTYPE), mask=maskbh)
            tl.store(cn_ptr + bbase[:, None] + oj[None, :], cv.to(O_DTYPE), mask=maskbh)


def _pick_block_i(I):
    p = 1
    while p < I:
        p *= 2
    if p > 128:
        p = 128
    if p < 16:
        p = 16
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
        hstate = torch.empty((2, B, H), device=x.device, dtype=torch.float32)
        cstate = torch.empty((B, H), device=x.device, dtype=torch.float32)

        # hidden-chunk (BO) and K-chunk (BKH) sizes for the inner loops
        if H <= 64:
            bo = 16
            while bo < H:
                bo *= 2
            bkh = bo
        else:
            bo = 64
            bkh = 64 if (H % 64) == 0 else 32

        block_b = 16
        block_i = _pick_block_i(I)
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
            h_0, c_0, out, hn, cn, hstate, cstate,
            S, B, I,
            x_sb, x_st,
            o_sb, o_st,
            num_cores=self.CUBE_CORE_NUM,
            HAS_BIAS=has_bias,
            H=H,
            BKH=bkh,
            BO=bo,
            BLOCK_B=block_b,
            BLOCK_I=block_i,
            O_DTYPE=o_dtype,
        )
        out_ret = out.transpose(0, 1) if batch_first else out
        return out_ret, (hn, cn)
