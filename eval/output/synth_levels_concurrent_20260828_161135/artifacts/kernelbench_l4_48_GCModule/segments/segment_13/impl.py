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
                 C: tl.constexpr, N: tl.constexpr, NC1: tl.constexpr,
                 total14: tl.constexpr, grid14: tl.constexpr,
                 CP: tl.constexpr, BN: tl.constexpr,
                 RET_T: tl.constexpr):
    pid = tl.program_id(0)
    c_off = tl.arange(0, CP)
    c_mask = c_off < C
    n_vec = tl.arange(0, BN)
    w = tl.load(w1_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + tl.arange(0, 1))
    for blk in range(pid, total14, grid14):
        b = blk // NC1
        nc = blk - b * NC1
        n_off = nc * BN + n_vec
        n_mask = n_off < N
        x = tl.load(x_ptr + (b * C + c_off[:, None]) * N + n_off[None, :],
                    mask=c_mask[:, None] & n_mask[None, :], other=0.0) \
            .to(tl.float32)
        attn = tl.sum(w[:, None] * x, axis=0) + b1
        attn = attn.to(RET_T).to(tl.float32)
        ctxp = tl.sum(x * attn[None, :], axis=1)
        tl.store(ctx_ws_ptr + blk * C + c_off, ctxp, mask=c_mask)


# ---------------------------------------------------------------------------
# Kernel 2b (merged ctx-conv + LN + conv): one program per batch element.
#   ctx(b,c) = sum_nc ws1[b,nc,c]
#   z(b,k)   = sum_c tw1[k,c]*ctx[b,c] + tb1[k]       (K, )
#   u = LN(z) (eps=1e-5) -> h = max(u, 0)
#   y(b, c)  = sum_k tw2[c, k] * h(b, k) + tb2[c]
#   LN / h live in UB; z is spilled to the small z_ws workspace (one B*K
#   f32 buffer) because a (KCP,) loop-carried accumulator cannot be
#   re-assigned a (BK,) tile in BiSheng Triton (CompilationError: shape
#   mismatch of loop-carried variable).
# ---------------------------------------------------------------------------
@triton.jit
def _k2b_ctx_transform(ctx_ws_ptr, z_ws_ptr, tw1_ptr, tb1_ptr, lnw_ptr, lnb_ptr,
                       tw2_ptr, tb2_ptr, y_ws_ptr,
                       C: tl.constexpr, K: tl.constexpr, NC1: tl.constexpr,
                       CP: tl.constexpr, NCC: tl.constexpr,
                       BK: tl.constexpr, KT: tl.constexpr,
                       KCP: tl.constexpr, BC: tl.constexpr,
                       RET_T: tl.constexpr):
    b = tl.program_id(0)
    c_off = tl.arange(0, CP)
    c_mask = c_off < C
    ctx = tl.zeros((CP,), dtype=tl.float32)
    for nc0 in range(0, NC1, NCC):
        nc_off = nc0 + tl.arange(0, NCC)
        nc_mask = nc_off < NC1
        sl = tl.load(ctx_ws_ptr + (b * NC1 + nc_off[:, None]) * C + c_off[None, :],
                     mask=nc_mask[:, None] & c_mask[None, :], other=0.0)
        ctx += tl.sum(sl, axis=0)
    ctx = ctx.to(RET_T).to(tl.float32)
    k_off = tl.arange(0, KCP)
    k_mask = k_off < K
    for kt in range(0, KT):
        kt_off = kt * BK + tl.arange(0, BK)
        kt_mask = kt_off < K
        tw = tl.load(tw1_ptr + kt_off[:, None] * C + c_off[None, :],
                     mask=kt_mask[:, None] & c_mask[None, :], other=0.0) \
            .to(tl.float32)
        tb1 = tl.load(tb1_ptr + kt_off, mask=kt_mask, other=0.0).to(tl.float32)
        z = tl.sum(tw * ctx[None, :], axis=1) + tb1
        tl.store(z_ws_ptr + b * K + kt_off, z.to(RET_T), mask=kt_mask)
    z_all = tl.load(z_ws_ptr + b * K + k_off, mask=k_mask, other=0.0) \
        .to(tl.float32)
    lnw = tl.load(lnw_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)
    lnb = tl.load(lnb_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)
    mu = tl.sum(z_all, axis=0) / K
    d = tl.where(k_mask, z_all - mu, 0.0)
    var = tl.sum(d * d, axis=0) / K
    rstd = 1.0 / tl.sqrt(var + 1e-05)
    u = d * rstd * lnw + lnb
    u = u.to(RET_T).to(tl.float32)
    h = tl.maximum(u, 0.0)
    c_base = tl.arange(0, BC)
    for ct0 in range(0, C, BC):
        c_off2 = ct0 + c_base
        c_mask2 = c_off2 < C
        tw2 = tl.load(tw2_ptr + c_off2[:, None] * K + k_off[None, :],
                      mask=c_mask2[:, None] & k_mask[None, :], other=0.0) \
            .to(tl.float32)
        tb2 = tl.load(tb2_ptr + c_off2, mask=c_mask2, other=0.0).to(tl.float32)
        y = tl.sum(tw2 * h[None, :], axis=1) + tb2
        tl.store(y_ws_ptr + b * C + c_off2, y.to(RET_T), mask=c_mask2)


# ---------------------------------------------------------------------------
# Kernel 4: out[b, c, n] = x[b, c, n] + y[b, c]
# ---------------------------------------------------------------------------
@triton.jit
def _k4_res(x_ptr, y_ws_ptr, out_ptr,
            C: tl.constexpr, N: tl.constexpr, NC1: tl.constexpr,
            total14: tl.constexpr, grid14: tl.constexpr,
            CP: tl.constexpr, BN: tl.constexpr,
            RET_T: tl.constexpr):
    pid = tl.program_id(0)
    c_off = tl.arange(0, CP)
    c_mask = c_off < C
    n_vec = tl.arange(0, BN)
    for blk in range(pid, total14, grid14):
        b = blk // NC1
        nc = blk - b * NC1
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

        # BiSheng per-core UB = 192KB; with multi-buffering the f32 compute
        # tile (cp*bn elts @4B) must stay <= 32KB, i.e. cp*bn <= 8192.
        # 16384 caused: "ub overflow, requires ... bits while 1572864 bits
        # available" in the BiShengHIR pipeline for every case.
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

        total14 = b * nc1
        grid14 = total14 if total14 < self.VEC_CORE_NUM else self.VEC_CORE_NUM

        _k1_attn_ctx[(grid14,)](
            x, w1, b1, ctx_ws,
            c, n, nc1, total14, grid14,
            CP=cp, BN=bn,
            RET_T=ret_t,
        )
        _k2b_ctx_transform[(b,)](
            ctx_ws, z_ws, tw1, tb1, lnw, lnb, tw2, tb2, y_ws,
            c, k, nc1,
            CP=cp, NCC=ncc, BK=bk, KT=kt,
            KCP=kcp, BC=bc,
            RET_T=ret_t,
        )
        _k4_res[(grid14,)](
            x, y_ws, out2,
            c, n, nc1, total14, grid14,
            CP=cp, BN=bn,
            RET_T=ret_t,
        )
        return out
   