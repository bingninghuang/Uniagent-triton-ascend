import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_cube_core_num():
    # checklist: core count must be read dynamically, not hardcoded
    try:
        import torch_npu
        limit = torch_npu.npu.npu_config.get_device_limit(0)
        return int(limit.get("cube_core_num", 24))
    except Exception:
        return 24  # ascend910b1: 24 AI cores (mixed cube+vec kernel -> cube count)


CUBE_CORE_NUM = _get_cube_core_num()
LOG2E: tl.constexpr = tl.constexpr(1.4426950408889634)


@triton.jit
def sdpa_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    scale,
    B: tl.constexpr,
    num_heads: tl.constexpr,
    sq_len: tl.constexpr,
    sk_len: tl.constexpr,
    dk: tl.constexpr,
    dv: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    num_blocks_total: tl.constexpr,
    num_m_tiles: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DQ: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    QK_F32: tl.constexpr,
):
    # D-split: any dim > 128 (i.e. next_pow2 would be 256) is processed in a
    # second 128-wide chunk. The backend rejects 256-wide tiles here.
    HAS_DQ2: tl.constexpr = dk > BLOCK_DQ
    HAS_V2: tl.constexpr = dv > BLOCK_DV

    pid = tl.program_id(0)
    in_dtype = q_ptr.dtype.element_ty
    off_m_idx = tl.arange(0, BLOCK_M)
    offs_dq0 = tl.arange(0, BLOCK_DQ)
    offs_dv0 = tl.arange(0, BLOCK_DV)

    f_dk: tl.constexpr = float(dk)
    f_dv: tl.constexpr = float(dv)
    f_sq: tl.constexpr = float(sq_len)
    f_sk: tl.constexpr = float(sk_len)
    # loop-invariant masks, computed once
    mask_dq0 = offs_dq0[None, :].to(tl.float32) < f_dk    # [1, BLOCK_DQ]
    mask_dv0 = offs_dv0[None, :].to(tl.float32) < f_dv    # [1, BLOCK_DV]
    if HAS_DQ2:
        offs_dq1 = 128 + tl.arange(0, 128)
        mask_dq1 = offs_dq1[None, :].to(tl.float32) < f_dk
    if HAS_V2:
        offs_dv1 = 128 + tl.arange(0, 128)
        mask_dv1 = offs_dv1[None, :].to(tl.float32) < f_dv

    # checklist: contiguous (non-interleaved) task partition; each program
    # handles a contiguous block range -> K/V reuse across m-tiles of one head.
    # Host guarantees num_cores <= num_blocks_total, so CHUNK >= 1.
    CHUNK: tl.constexpr = num_blocks_total // num_cores
    EXTRA: tl.constexpr = num_blocks_total - CHUNK * num_cores
    extra = (pid.to(tl.float32) < float(EXTRA)).to(tl.int32)
    start_off = pid * CHUNK + extra
    for i in range(CHUNK + extra):
        block_idx = start_off + i
        # plain scalar int ops (constexpr scalars have no .to() in this DSL)
        tmp = block_idx // num_m_tiles
        m_tile = block_idx - tmp * num_m_tiles
        b = tmp // num_heads
        h = tmp - b * num_heads

        off_m = m_tile * BLOCK_M + off_m_idx
        mask_m = off_m.to(tl.float32) < f_sq
        mask_mrow = mask_m[:, None]

        off_q_m = off_m * dk
        off_o_m = off_m * dv

        q_base = q_ptr + b * stride_qb + h * stride_qh
        k_base = k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        o_base = o_ptr + b * stride_ob + h * stride_oh

        # load Q once per block (in the needed dtype)
        q0 = tl.load(
            q_base + off_q_m[:, None] + offs_dq0[None, :],
            mask=mask_mrow & mask_dq0, other=0.0,
        )
        if QK_F32:
            q0 = q0.to(tl.float32)
        if HAS_DQ2:
            q1 = tl.load(
                q_base + off_q_m[:, None] + offs_dq1[None, :],
                mask=mask_mrow & mask_dq1, other=0.0,
            )
            if QK_F32:
                q1 = q1.to(tl.float32)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc0 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        if HAS_V2:
            acc1 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)

        # single loop over all K/V tiles; the tail is handled by column
        # masking (masked K cols load 0 -> their scores become 0.0, which is
        # then clamped to -inf so they never disturb the online max/rescale)
        offs_n_base = tl.arange(0, BLOCK_N)
        num_n_tiles: tl.constexpr = (sk_len + BLOCK_N - 1) // BLOCK_N
        for n_tile in range(0, num_n_tiles):
            off_n = n_tile * BLOCK_N + offs_n_base
            mask_n = off_n.to(tl.float32) < f_sk
            mask_nrow = mask_n[:, None]
            off_k_n = off_n * dk
            off_v_n = off_n * dv

            k0 = tl.load(
                k_base + off_k_n[:, None] + offs_dq0[None, :],
                mask=mask_nrow & mask_dq0, other=0.0,
            )
            if QK_F32:
                k0 = k0.to(tl.float32)
            scores = tl.dot(q0, tl.trans(k0))
            if HAS_DQ2:
                k1 = tl.load(
                    k_base + off_k_n[:, None] + offs_dq1[None, :],
                    mask=mask_nrow & mask_dq1, other=0.0,
                )
                if QK_F32:
                    k1 = k1.to(tl.float32)
                scores = tl.dot(q1, tl.trans(k1), scores)

            # CRITICAL: masked K columns load as 0 -> their scores are 0.0.
            # Without -inf masking, m_new would pick up 0.0 whenever every
            # real score is negative, breaking the online rescaling
            # (exp underflow -> l_i = 0 -> 0/0 = NaN).
            scores = scores * scale
            scores = tl.where(mask_n[None, :], scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp2((scores - m_new[:, None]) * LOG2E)
            alpha = tl.exp2((m_i - m_new) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            v0 = tl.load(
                v_base + off_v_n[:, None] + offs_dv0[None, :],
                mask=mask_nrow & mask_dv0, other=0.0,
            )
            acc0 = tl.dot(p, v0.to(tl.float32), acc0 * alpha[:, None])
            if HAS_V2:
                v1 = tl.load(
                    v_base + off_v_n[:, None] + offs_dv1[None, :],
                    mask=mask_nrow & mask_dv1, other=0.0,
                )
                acc1 = tl.dot(p, v1.to(tl.float32), acc1 * alpha[:, None])

        inv_l = 1.0 / l_i
        tl.store(
            o_base + off_o_m[:, None] + offs_dv0[None, :],
            (acc0 * inv_l[:, None]).to(in_dtype),
            mask=mask_mrow & mask_dv0,
        )
        if HAS_V2:
            tl.store(
                o_base + off_o_m[:, None] + offs_dv1[None, :],
                (acc1 * inv_l[:, None]).to(in_dtype),
                mask=mask_mrow & mask_dv1,
            )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, scale):
        B, H, Sq, Dk = Q.shape
        Sk = K.shape[2]
        Dv = V.shape[3]
        dtype = Q.dtype

        Q = Q.contiguous() if not Q.is_contiguous() else Q
        K = K.contiguous() if not K.is_contiguous() else K
        V = V.contiguous() if not V.is_contiguous() else V
        out = torch.empty((B, H, Sq, Dv), device=Q.device, dtype=dtype)

        is_f32 = (dtype == torch.float32)
        BLOCK_M = 32 if is_f32 else 64
        BLOCK_N = 64
        # cap chunk widths at 128; bigger dims use a second 128-wide chunk
        BLOCK_DQ = min(triton.next_power_of_2(Dk), 128)
        BLOCK_DV = min(triton.next_power_of_2(Dv), 128)

        num_m_tiles = triton.cdiv(Sq, BLOCK_M)
        total_blocks = B * H * num_m_tiles
        # never launch more programs than there are blocks (keeps CHUNK >= 1)
        num_cores = min(CUBE_CORE_NUM, total_blocks)
        grid = (num_cores,)
        sdpa_fwd_kernel[grid](
            Q, K, V, out,
            float(scale),
            B, H, Sq, Sk, Dk, Dv,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            V.stride(0), V.stride(1),
            out.stride(0), out.stride(1),
            total_blocks,
            num_m_tiles,
            num_cores,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DQ=BLOCK_DQ,
            BLOCK_DV=BLOCK_DV,
            QK_F32=is_f32,
        )
        return out
