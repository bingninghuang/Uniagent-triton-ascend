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
                 C, N, NC1,
                 CP: tl.constexpr, BN: tl.constexpr, NP: tl.constexpr,
                 X_T: tl.constexpr, RET_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    total = NC1 * (ctx_ws_ptr.shape[0] // C) if False else NC1 * 0 + (CP // CP) * 0  # placeholder removed below
    total2 = tl.load(x_ptr - x_ptr + 0, mask=(tl.arange(0, 1) == 0), other=1)  # unused
    pass


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
    b1 = tl.load(b1_ptr, mask=(tl.arange(0, 1) == 0), other=0.0).to(tl.float32)
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
        tl.store(ctx_ws_ptr + (b * NC1 + nc) * C + c_off, ctxp.to(RET_T),
                 mask=c_mask)


# ---------------------------------------------------------------------------
# Kernel 2a: ctx(b,c) = sum_nc ws1[b,nc,c]
#            z(b,k) = sum_c tw1[k,c]*ctx[b,c] + tb1[k]      (K, )
# ---------------------------------------------------------------------------
@triton.jit
def _k2a_ctx_conv(ctx_ws_ptr, tw1_ptr, tb1_ptr, z_ws_ptr,
                  B, C, K, NC1,
                  CP: tl.constexpr, KC: tl.constexpr,
                  BC: tl.constexpr, BK: tl.constexpr,
                  X_T: tl.constexpr, RET_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    KT = tl.cdiv(K, BK)
    b = (pid // KT).to(tl.int32)
    kt = (pid % KT).to(tl.int32)

    c_off = tl.arange(0, CP).to(tl.int32)
    c_mask = c_off < C
    ncv = tl.arange(0, CP // CP * CP).to(tl.int32)  # placeholder (avoid)
    pass


@triton.jit
def _k2a_ctx_conv(ctx_ws_ptr, tw1_ptr, tb1_ptr, z_ws_ptr,
                  B, C, K, NC1,
                  CP: tl.constexpr, KCP: tl.constexpr,
                  BC: tl.constexpr, BK: tl.constexpr,
                  X_T: tl.constexpr, RET_T: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    KT = tl.cdiv(K, BK)
    b = (pid // KT).to(tl.int32)
    kt = (pid % KT).to(tl.int32)

    c_off = tl.arange(0, CP).to(tl.int32)
    c_mask = c_off < C

    # ctx(b, CP) = sum over NC1 partial rows
    nc_off = tl.arange(0, KCP).to(tl.int32)
    nc_mask = nc_off < NC1
    sl = tl.load(ctx_ws_ptr + (b * NC1 + nc_off[:, None]) * C + c_off[None, :],
                 mask=nc_mask[:, None] & c_mask[None, :], other=0.0) \
        .to(tl.float32)
    ctx = tl.sum(sl, axis=0)

    k_lo = kt * BK
   