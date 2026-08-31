import torch
import torch.nn as nn
import triton
import triton.language as tl


# tanh(x) = erf(x / sqrt(2)).  tl.tanh is not available on Ascend.
_INV_SQRT2 = tl.constexpr(0.70710678118654752440084436210485)
_BIG = tl.constexpr(10000000)


@triton.jit
def _fa_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, dy_ptr, o_ptr, smax_ptr, ssum_ptr, dq_ptr,
    S_q, S_kv, H_q, group, D,
    scale, softcap, inv_softcap, delta, wl, wr, causal_f,
    q_sb, q_ss, q_sh,
    k_sb, k_ss, k_sh,
    v_sb, v_ss, v_sh,
    dy_sb, dy_ss, dy_sh,
    o_sb, o_ss, o_sh,
    st_sb, st_sh, st_ss,
    ut_sb, ut_sh, ut_ss,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H_q
    h = pid_bh % H_q
    h_kv = h // group

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < S_q
    offs_d = tl.arange(0, BD)
    mask_d = offs_d < D
    m_offs = offs_m[:, None]
    d_offs = offs_d[None, :]
    tile_mask = mask_m[:, None] & mask_d[None, :]

    q = tl.load(
        q_ptr + b * q_sb + h * q_sh + m_offs * q_ss + d_offs,
        mask=tile_mask, other=0.0).to(tl.float32)
    dy_ = tl.load(
        dy_ptr + b * dy_sb + h * dy_sh + m_offs * dy_ss + d_offs,
        mask=tile_mask, other=0.0).to(tl.float32)
    o = tl.load(
        o_ptr + b * o_sb + h * o_sh + m_offs * o_ss + d_offs,
        mask=tile_mask, other=0.0).to(tl.float32)

    # NOTE: other=1.0 keeps out-of-range rows finite (x_sum must never be 0).
    x_max = tl.load(
        smax_ptr + b * st_sb + h * st_sh + offs_m * st_ss,
        mask=mask_m, other=1.0)
    x_sum = tl.load(
        ssum_ptr + b * ut_sb + h * ut_sh + offs_m * ut_ss,
        mask=mask_m, other=1.0)

    d_row = tl.sum(dy_ * o, axis=1)

    # Effective per-row valid column range:  row + l_eff <= col <= row + r_eff
    r_eff = tl.minimum(tl.where(causal_f != 0, delta, _BIG),
                       tl.where(wr >= 0, wr, _BIG))
    l_eff = tl.where(wl >= 0, -wl, -_BIG)

    row_min = pid_m * BM
    row_max = tl.minimum(row_min + BM - 1, S_q - 1)

    # First kv tile that is not fully masked by the left window boundary.
    x_l = tl.maximum(0, row_min + l_eff + 1)
    num_n = (S_kv + BN - 1) // BN
    n_start = tl.minimum((x_l + BN - 1) // BN, num_n)

    acc = tl.zeros((BM, BD), dtype=tl.float32)

    for n in range(n_start, num_n):
        n0 = n * BN
        if n0 > row_max + r_eff:
            break

        offs_n = n0 + tl.arange(0, BN)
        mask_n = offs_n < S_kv
        n_mask = mask_n[None, :]

        k_j = tl.load(
            k_ptr + b * k_sb + h_kv * k_sh + offs_n[:, None] * k_ss + d_offs,
            mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v_j = tl.load(
            v_ptr + b * v_sb + h_kv * v_sh + offs_n[:, None] * v_ss + d_offs,
            mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k_j), out_dtype=tl.float32)
        s = s * scale

        # softcap: t = tanh(s / softcap);  s_cap = softcap * t
        t = tl.erf(s * (inv_softcap * _INV_SQRT2))
        s = tl.where(softcap > 0.0, softcap * t, s)

        # relative position mask
        rel = offs_n[None, :].to(tl.int32) - offs_m[:, None].to(tl.int32) - delta
        masked = (tl.where((causal_f != 0) & (rel > 0), 1, 0)
                  | tl.where((wl >= 0) & (rel < -wl), 1, 0)
                  | tl.where((wr >= 0) & (rel > wr), 1, 0))
        s = tl.where(masked == 0, s, -40000.0)

        p = tl.exp(s - x_max[:, None]) / x_sum[:, None]

        dp = tl.dot(dy_, tl.trans(v_j), out_dtype=tl.float32)
        ds = p * (dp - d_row[:, None])
        ds = ds * (tl.where(softcap > 0.0, 1.0 - t * t, 1.0))
        ds = ds * scale

        acc = tl.dot(ds, k_j, acc, out_dtype=tl.float32)

    out = acc.to(dq_ptr.dtype.element_ty)
    tl.store(
        dq_ptr + b * q_sb + h * q_sh + m_offs * q_ss + d_offs,
        out, mask=tile_mask)


@triton.jit
def _fa_bwd_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, dy_ptr, o_ptr, smax_ptr, ssum_ptr, dk_ptr, dv_ptr,
    S_q, S_kv, H_kv, group, D,
    scale, softcap, inv_softcap, delta, wl, wr, causal_f,
    q_sb, q_ss, q_sh,
    k_sb, k_ss, k_sh,
    v_sb, v_ss, v_sh,
    dy_sb, dy_ss, dy_sh,
    o_sb, o_ss, o_sh,
    st_sb, st_sh, st_ss,
    ut_sb, ut_sh, ut_ss,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bhk = tl.program_id(1)

    b = pid_bhk // H_kv
    h_kv = pid_bhk % H_kv
    h_first = h_kv * group

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < S_kv
    offs_d = tl.arange(0, BD)
    mask_d = offs_d < D
    n_offs = offs_n[:, None]
    d_offs = offs_d[None, :]
    tile_mask = mask_n[:, None] & mask_d[None, :]

    k_j = tl.load(
        k_ptr + b * k_sb + h_kv * k_sh + n_offs * k_ss + d_offs,
        mask=tile_mask, other=0.0).to(tl.float32)
    v_j = tl.load(
        v_ptr + b * v_sb + h_kv * v_sh + n_offs * v_ss + d_offs,
        mask=tile_mask, other=0.0).to(tl.float32)

    n0 = pid_n * BN
    n1 = n0 + BN - 1

    r_eff = tl.minimum(tl.where(causal_f != 0, delta, _BIG),
                       tl.where(wr >= 0, wr, _BIG))
    l_eff = tl.where(wl >= 0, -wl, -_BIG)

    # First q tile that is not fully masked by the causal / right boundary.
    x_r = tl.maximum(0, n0 - r_eff - BM + 1)
    num_m = (S_q + BM - 1) // BM
    m_start = tl.minimum((x_r + BM - 1) // BM, num_m)

    acc_dk = tl.zeros((BN, BD), dtype=tl.float32)
    acc_dv = tl.zeros((BN, BD), dtype=tl.float32)

    for m in range(m_start, num_m):
        m0 = m * BM
        if n1 < m0 + l_eff:
            break

        offs_m = m0 + tl.arange(0, BM)
        mask_m = offs_m < S_q
        m_mask = mask_m[None, :]

        # relative position mask (independent of the group head g)
        rel = offs_n[:, None].to(tl.int32) - offs_m[None, :].to(tl.int32) - delta
        masked = (tl.where((causal_f != 0) & (rel > 0), 1, 0)
                  | tl.where((wl >= 0) & (rel < -wl), 1, 0)
                  | tl.where((wr >= 0) & (rel > wr), 1, 0))
        valid = (masked == 0) & m_mask  # (BN, BM)

        for g in range(group):
            h = h_first + g

            q_i = tl.load(
                q_ptr + b * q_sb + h * q_sh + offs_m[:, None] * q_ss + d_offs,
                mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            dy_i = tl.load(
                dy_ptr + b * dy_sb + h * dy_sh + offs_m[:, None] * dy_ss + d_offs,
                mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            o_i = tl.load(
                o_ptr + b * o_sb + h * o_sh + offs_m[:, None] * o_ss + d_offs,
                mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            # other=1.0 keeps out-of-range rows finite (x_sum must never be 0)
            x_max = tl.load(
                smax_ptr + b * st_sb + h * st_sh + offs_m * st_ss,
                mask=mask_m, other=1.0)
            x_sum = tl.load(
                ssum_ptr + b * ut_sb + h * ut_sh + offs_m * ut_ss,
                mask=mask_m, other=1.0)

            d_row = tl.sum(dy_i * o_i, axis=1)  # (BM,)

            s = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32)  # (BM, BN)
            s = s * scale

            t = tl.erf(s * (inv_softcap * _INV_SQRT2))
            s = tl.where(softcap > 0.0, softcap * t, s)
            s = tl.where(tl.trans(valid), s, -40000.0)

            p = tl.exp(s - x_max[:, None]) / x_sum[:, None]

            dp = tl.dot(dy_i, tl.trans(v_j), out_dtype=tl.float32)
            ds = p * (dp - d_row[:, None])
            ds = ds * (tl.where(softcap > 0.0, 1.0 - t * t, 1.0))
            ds = ds * scale

            acc_dk = tl.dot(tl.trans(ds), q_i, acc_dk, out_dtype=tl.float32)
            acc_dv = tl.dot(tl.trans(p), dy_i, acc_dv, out_dtype=tl.float32)

    dk_out = acc_dk.to(dk_ptr.dtype.element_ty)
    dv_out = acc_dv.to(dv_ptr.dtype.element_ty)
    tl.store(
        dk_ptr + b * k_sb + h_kv * k_sh + n_offs * k_ss + d_offs,
        dk_out, mask=tile_mask)
    tl.store(
        dv_ptr + b * v_sb + h_kv * v_sh + n_offs * v_ss + d_offs,
        dv_out, mask=tile_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        dy,
        softmax_max,
        softmax_sum,
        attention_in,
        causal,
        window_left,
        window_right,
        softcap,
    ):
        B, S_q, H_q, D = q.shape
        S_kv = k.shape[1]
        H_kv = k.shape[2]

        scale = D ** -0.5
        group = H_q // H_kv
        delta = S_kv - S_q
        softcap = float(softcap)
        inv_softcap = 1.0 / softcap if softcap > 0.0 else 0.0
        causal_f = 1 if causal else 0

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BM = 32
        BN = 32
        if D <= 16:
            BD = 16
        elif D <= 32:
            BD = 32
        elif D <= 64:
            BD = 64
        elif D <= 128:
            BD = 128
        else:
            BD = 256

        num_m = triton.cdiv(S_q, BM)
        num_n = triton.cdiv(S_kv, BN)

        common_scalars = (
            scale, softcap, inv_softcap, delta,
            int(window_left), int(window_right), causal_f,
        )

        _fa_bwd_dq_kernel[(num_m, B * H_q)](
            q, k, v, dy, attention_in, softmax_max, softmax_sum, dq,
            S_q, S_kv, H_q, group, D,
            *common_scalars,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
            attention_in.stride(0), attention_in.stride(1), attention_in.stride(2),
            softmax_max.stride(0), softmax_max.stride(1), softmax_max.stride(2),
            softmax_sum.stride(0), softmax_sum.stride(1), softmax_sum.stride(2),
            BM=BM, BN=BN, BD=BD,
        )

        _fa_bwd_dkdv_kernel[(num_n, B * H_kv)](
            q, k, v, dy, attention_in, softmax_max, softmax_sum, dk, dv,
            S_q, S_kv, H_kv, group, D,
            *common_scalars,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            dy.stride(0), dy.stride(1), dy.stride(2),
            attention_in.stride(0), attention_in.stride(1), attention_in.stride(2),
            softmax_max.stride(0), softmax_max.stride(1), softmax_max.stride(2),
            softmax_sum.stride(0), softmax_sum.stride(1), softmax_sum.stride(2),
            BM=BM, BN=BN, BD=BD,
        )

        return dq, dk, dv
