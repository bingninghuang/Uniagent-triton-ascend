import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


_TRT_MAP = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


# ---------------------------------------------------------------------------
# Kernel 1: attn[b,n] = sum_c w1[c]*x[b,c,n] + b1
#           ctx_partial(b,nc,c) = sum_{n in tile} x[b,c,n]*attn[b,n]
# ---------------------------------------------------------------------------

@triton.jit
def _k1_attn_ctx(x_ptr, w1_ptr, b1_ptr, ctx_ws_ptr,
                 B, C, N, NC1,
                 CP: tl.constexpr, BN: tl.constexpr, NP: tl.constexpr,
                 X_T: tl.constexpr, RET_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    total = B * NC1
    c_off = tl.arange(0, CP).to(tl.int32)
    c_mask = c_off < C
    n_vec = tl.arange(0, BN).to(tl.int32)
    w = tl.load(w1_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + tl.arange(0, 1), mask=(tl.arange(0, 1) == 0),
                 other=0.0).to(tl.float32)
    for blk in range(pid, total, NP):
        b = (blk // NC1).to(tl.int32)
        nc = (blk % NC1).to(tl.int32)
        n0 = nc * BN
        n_off = n0 + n_vec
        n_mask = n_off < N
        x = tl.load(x_ptr + (b * C + c_off[:, None]) * N + n_off[None, :],
                    mask=c_mask[:, None] & n_mask[None, :], other=0.0) \
            .to(tl.float32)
        attn = tl.sum(w[:, None] * x, axis=0) + b1
        attn = attn.to(RET_T).to(tl.float32)
        ctxp = tl.sum(x * attn[None, :], axis=1)
        tl.store(ctx_ws_ptr + (b * NC1 + nc) * C + c_off, ctxp,
                 mask=c_mask)


# ---------------------------------------------------------------------------
# Kernel 2a: ctx(b,c) = sum_nc ws1[b,nc,c]
#            z(b,k) = sum_c tw1[k,c]*ctx[b,c] + tb1[k]      (K, )
# ---------------------------------------------------------------------------

@triton.jit
def _k2a_ctx_conv(ctx_ws_ptr, tw1_ptr, tb1_ptr, z_ws_ptr,
                  B, C, K, NC1,
                  CP: tl.constexpr, NCC: tl.constexpr,
                  BK: tl.constexpr,
                  X_T: tl.constexpr, RET_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    KT = tl.cdiv(K, BK)
    b = (pid // KT).to(tl.int32)
    kt = (pid % KT).to(tl.int32)

    c_off = tl.arange(0, CP).to(tl.int32)
    c_mask = c_off < C

    # ctx(b, CP) fp32 = sum over NC1 partial rows, then round to out dtype
    ctx = tl.zeros([CP], dtype=tl.float32)
    for nc0 in range(0, NC1, NCC):
        nc_off = nc0 + tl.arange(0, NCC).to(tl.int32)
        nc_mask = nc_off < NC1
        sl = tl.load(ctx_ws_ptr + (b * NC1 + nc_off[:, None]) * C + c_off[None, :],
                     mask=nc_mask[:, None] & c_mask[None, :], other=0.0) \
            .to(tl.float32)
        ctx += tl.sum(sl, axis=0)
    ctx = ctx.to(RET_T).to(tl.float32)

    k_off = kt * BK + tl.arange(0, BK).to(tl.int32)
    k_mask = k_off < K
    tw = tl.load(tw1_ptr + k_off[:, None] * C + c_off[None, :],
                 mask=k_mask[:, None] & c_mask[None, :], other=0.0) \
        .to(tl.float32)
    tb1 = tl.load(tb1_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)
    z = tl.sum(tw * ctx[None, :], axis=1) + tb1
    tl.store(z_ws_ptr + b * K + k_off, z.to(RET_T), mask=k_mask)


# ---------------------------------------------------------------------------
# Kernel 3: u = LN(z) (eps=1e-5) -> h = max(u, 0)
#           y(b, c) = sum_k tw2[c, k] * h(b, k) + tb2[c]
# ---------------------------------------------------------------------------
@triton.jit
def _k3_ln_conv(z_ws_ptr, tw2_ptr, tb2_ptr, lnw_ptr, lnb_ptr, y_ws_ptr,
                B, C, K,
                KCP: tl.constexpr, BC: tl.constexpr,
                X_T: tl.constexpr, RET_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    CT = tl.cdiv(C, BC)
    b = (pid // CT).to(tl.int32)
    ct = (pid % CT).to(tl.int32)

    k_off = tl.arange(0, KCP).to(tl.int32)
    k_mask = k_off < K
    z = tl.load(z_ws_ptr + b * K + k_off, mask=k_mask, other=0.0).to(tl.float32)
    lnw = tl.load(lnw_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)
    lnb = tl.load(lnb_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)

    mu = tl.sum(tl.where(k_mask, z, 0.0)) / K
    d = tl.where(k_mask, z - mu, 0.0)
    var = tl.sum(d * d) / K
    rstd = 1.0 / tl.sqrt(var + 1e-05)
    u = (d * rstd * lnw + lnb)
    u = u.to(RET_T).to(tl.float32)
    h = tl.maximum(u, 0.0)

    c_off = ct * BC + tl.arange(0, BC).to(tl.int32)
    c_mask = c_off < C
    tw2 = tl.load(tw2_ptr + c_off[:, None] * K + k_off[None, :],
                  mask=c_mask[:, None] & k_mask[None, :], other=0.0) \
        .to(tl.float32)
    tb2 = tl.load(tb2_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)
    y = tl.sum(tw2 * h[None, :], axis=1) + tb2
    tl.store(y_ws_ptr + b * C + c_off, y.to(RET_T), mask=c_mask)


# ---------------------------------------------------------------------------
# Kernel 4: out[b, c, n] = x[b, c, n] + y[b, c]
# ---------------------------------------------------------------------------
@triton.jit
def _k4_res(x_ptr, y_ws_ptr, out_ptr,
            B, C, N, NC1,
            CP: tl.constexpr, BN: tl.constexpr, NP: tl.constexpr,
            X_T: tl.constexpr, RET_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    total = B * NC1
    c_off = tl.arange(0, CP).to(tl.int32)
    c_mask = c_off < C
    n_vec = tl.arange(0, BN).to(tl.int32)
    for blk in range(pid, total, NP):
        b = (blk // NC1).to(tl.int32)
        nc = (blk % NC1).to(tl.int32)
        n_off = nc * BN + n_vec
        n_mask = n_off < N
        yv = tl.load(y_ws_ptr + b * C + c_off,
                     mask=c_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + (b * C + c_off[:, None]) * N + n_off[None, :],
                    mask=c_mask[:, None] & n_mask[None, :], other=0.0) \
            .to(tl.float32)
        o = (x + yv[:, None]).to(RET_T)
        tl.store(out_ptr + (b * C + c_off[:, None]) * N + n_off[None, :], o,
                 mask=c_mask[:, None] & n_mask[None, :])


def _pow2_le(v):
    p = 1
    while p <= v:
        p *= 2
    return p // 2


def _pow2_lt(v):
    p = 1
    while p < v:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._bufs = {}
        try:
            limit = torch_npu.npu.npu_config.get_device_limit(0)
            self.VEC_CORE_NUM = int(limit.get('vector_core_num', 48))
        except Exception:
            self.VEC_CORE_NUM = 48

    def forward(self, x, channel, reduction, context,
                conv1_w, conv1_b,
                t_w1, t_b1, ln_w, ln_b, t_w2, t_b2):
        b, c, h, w = x.shape
        n = h * w
        k = c // reduction
        ret_t = _TRT_MAP[x.dtype]
        x = x.contiguous()

        w1 = conv1_w.contiguous().view(c)
        b1 = conv1_b.contiguous()
        tw1 = t_w1.contiguous().view(k, c)
        tb1 = t_b1.contiguous()
        lnw = ln_w.contiguous().view(k)
        lnb = ln_b.contiguous().view(k)
        tw2 = t_w2.contiguous().view(c, k)
        tb2 = t_b2.contiguous()

        cp = 1 << (c - 1).bit_length()
        kcp = 1 << (k - 1).bit_length()

        cap = 8192 // cp
        bn = cap
        if bn < 16:
            bn = 16
        nc1 = (n + bn - 1) // bn

        bk = _pow2_le(k)
        if bk > cap:
            bk = _pow2_lt(cap)
        kt = (k + bk - 1) // bk

        ncc = _pow2_le(nc1)
        if ncc > cap:
            ncc = _pow2_lt(cap)

        bc = _pow2_le(c)
        cap3 = 8192 // kcp
        if bc > cap3:
            bc = _pow2_lt(cap3)
        ct = (c + bc - 1) // bc

        key = (b, c, n, k, x.dtype)
        bufs = self._bufs.get(key)
        if bufs is None:
            bufs = (
                torch.empty(b * nc1 * c, dtype=torch.float32, device=x.device),
                torch.empty(b * k, dtype=torch.float32, device=x.device),
                torch.empty(b * c, dtype=torch.float32, device=x.device),
            )
            self._bufs[key] = bufs
        ctx_ws, z_ws, y_ws = bufs

        out = torch.empty_like(x)
        out2 = out.view(b, c, n)

        np_ = b * nc1
        if np_ > self.VEC_CORE_NUM:
            np_ = self.VEC_CORE_NUM

        _k1_attn_ctx[(np_,)](
            x, w1, b1, ctx_ws,
            b, c, n, nc1,
            CP=cp, BN=bn, NP=np_,
            X_T=ret_t, RET_T=ret_t,
        )
        _k2a_ctx_conv[(b * kt,)](
            ctx_ws, tw1, tb1, z_ws,
            b, c, k, nc1,
            CP=cp, NCC=ncc, BK=bk,
            X_T=ret_t, RET_T=ret_t,
        )
        _k3_ln_conv[(b * ct,)](
            z_ws, tw2, tb2, lnw, lnb, y_ws,
            b, c, k,
            KCP=kcp, BC=bc,
            X_T=ret_t, RET_T=ret_t,
        )
        _k4_res[(np_,)](
            x, y_ws, out2,
            b, c, n, nc1,
            CP=cp, BN=bn, NP=np_,
            X_T=ret_t, RET_T=ret_t,
        )
        return out
   