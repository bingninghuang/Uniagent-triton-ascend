import torch
import triton
import triton.language as tl


@triton.jit
def _kda_fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, a_ptr, bias_ptr,
    init_ptr, final_ptr, o_ptr, cu_ptr,
    T, H, K, V, scale,
    HAS_BIAS: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_FINAL: tl.constexpr,
    HAS_VARLEN: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NVSP: tl.constexpr,
):
    pid = tl.program_id(0)
    vs = pid % NVSP
    pid = pid // NVSP
    h = pid % H
    seq = pid // H

    if HAS_VARLEN:
        bos = tl.load(cu_ptr + seq).to(tl.int32)
        eos = tl.load(cu_ptr + seq + 1).to(tl.int32)
        tb_base = 0
    else:
        bos = 0
        eos = T
        tb_base = seq * T

    kk = tl.arange(0, BK)
    vv = tl.arange(0, BV)
    vfull = vs * BV + vv
    kmask = kk < K
    vmask = vfull < V
    kkm = kmask[:, None] & vmask[None, :]

    st_off = seq * (H * K * V) + h * (K * V) + kk[:, None] * V + vfull[None, :]
    if HAS_INIT:
        h_state = tl.load(init_ptr + st_off, mask=kkm, other=0.0).to(tl.float32)
    else:
        h_state = tl.zeros((BK, BV), dtype=tl.float32)

    a_log = tl.load(a_ptr + h)
    neg_exp_a = -tl.exp(a_log)

    for t in range(bos, eos):
        tt = tb_base + t

        kt = tl.load(k_ptr + tt * (H * K) + h * K + kk, mask=kmask, other=0.0).to(tl.float32)
        qt = tl.load(q_ptr + tt * (H * K) + h * K + kk, mask=kmask, other=0.0).to(tl.float32) * scale
        graw = tl.load(g_ptr + tt * (H * K) + h * K + kk, mask=kmask, other=0.0)
        if HAS_BIAS:
            graw = graw + tl.load(bias_ptr + h * K + kk, mask=kmask, other=0.0)
        bt = tl.load(beta_ptr + tt * H + h).to(tl.float32)
        vt = tl.load(v_ptr + tt * (H * V) + h * V + vfull, mask=vmask, other=0.0).to(tl.float32)

        sp = tl.where(graw > 20.0, graw, tl.log(1.0 + tl.exp(tl.minimum(graw, 20.0))))
        g = neg_exp_a * sp
        alpha = tl.exp(g)

        h_state = h_state * alpha[:, None]
        htk = tl.sum(h_state * kt[:, None], axis=0)
        delta = (vt - htk) * bt
        h_state = h_state + kt[:, None] * delta[None, :]
        o_t = tl.sum(h_state * qt[:, None], axis=0)
        tl.store(o_ptr + tt * (H * V) + h * V + vfull, o_t, mask=vmask)

    if HAS_FINAL:
        tl.store(final_ptr + st_off, h_state, mask=kkm)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, raw_g, beta, A_log, g_bias=None, scale=None,
                initial_state=None, output_final_state=False, cu_seqlens=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        rg = raw_g.contiguous()
        beta = beta.contiguous()
        A_log = A_log.contiguous()

        B, T, H, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K ** -0.5
        if rg.dim() == 3:
            rg = rg.reshape(B, T, H, K).contiguous()

        has_varlen = cu_seqlens is not None
        if has_varlen:
            nseq = cu_seqlens.shape[0] - 1
        else:
            nseq = B

        o = torch.empty(B, T, H, V, device=q.device, dtype=v.dtype)
        if output_final_state:
            final_state = torch.empty(nseq, H, K, V, device=q.device, dtype=torch.float32)
        else:
            final_state = o

        has_init = initial_state is not None
        init = initial_state.contiguous() if has_init else o
        bias = g_bias.contiguous() if g_bias is not None else o

        BK = triton.next_power_of_2(K)
        BV = 64
        if V > BV:
            NVSP = triton.cdiv(V, BV)
        else:
            NVSP = 1
        grid = (nseq * H * NVSP,)
        _kda_fwd_kernel[grid](
            q, k, v, rg, beta, A_log, bias, init, final_state, o, cu_seqlens,
            T, H, K, V, scale,
            g_bias is not None, has_init, output_final_state, has_varlen,
            BK=BK, BV=BV, NVSP=NVSP,
        )
        if output_final_state:
            return o, final_state
        return o, torch.empty(0, device=q.device, dtype=torch.float32)
