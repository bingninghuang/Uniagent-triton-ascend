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
    pid = tl.program_id(0)
    in_dtype = q_ptr.dtype.element_ty
    off_m_idx = tl.arange(0, BLOCK_M)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dq = tl.arange(0, BLOCK_DQ)
    # fp32 comparisons (checklist: numeric compares in fp32 for vector path)
    f_dk: tl.constexpr = float(dk)
    f_dv: tl.constexpr = float(dv)
    f_sq: tl.constexpr = float(sq_len)
    f_sk: tl.constexpr = float(sk_len)
    # loop-invariant masks, computed once
    mask_d = offs_dq[None, :].to(tl.float32) < f_dk      # [1, BLOCK_DQ]
    mask_dv = offs_dv[None, :].to(tl.float32) < f_dv     # [1, BLOCK_DV]

    # checklist: contiguous (non-interleaved) task partition; each program
    # handles a contiguous block range -> K/V L2 reuse across m-tiles of head
    CHUNK: tl.constexpr = num_blocks_total // num_cores
    EXTRA: tl.constexpr = num_blocks_total - CHUNK * num_cores
    extra = (pid.to(tl.float32) < float(EXTRA)).to(tl.int32)
    start_off = pid * CHUNK + extra
    for i in range(CHUNK + extra):
        block_idx = start_off + i
        # scalar int div -> float32 div (exact for small ints)
        f_idx = block_idx.to(tl.float32)
        f_m = num_m_tiles.to(tl.float32)
        f_h = num_heads.to(tl.float32)
        tmp = (f_idx // f_m).to(tl.int32)
        m_tile = block_idx - tmp * num_m_tiles
        b = (tmp.to(tl.float32) // f_h).to(tl.int32)
        h = tmp - b * num_heads

        off_m = m_tile * BLOCK_M + off_m_idx
        mask_m = off_m.to(tl.float32) < f_sq
        mask_mrow = mask_m[:, None]

        q_base = q_ptr + b * stride_qb + h * stride_qh
        k_base = k_ptr + b * stride_kb + h * stride_kh
        v_base = v_ptr + b * stride_vb + h * stride_vh
        o_base = o_ptr + b * stride_ob + h * stride_oh

        # load Q once per block
        q_load = tl.load(
            q_base + off_m[:, None] * dk + offs_dq[None, :],
            mask=mask_mrow & mask_d, other=0.0,
        )
        if QK_F32:
            q_tile = q_load.to(tl.float32)
        else:
            q_tile = q_load

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

        N_FULL: tl.constexpr = sk_len // BLOCK_N
        for n_tile in range(0, N_FULL):
            off_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
            k_load = tl.load(
                k_base + off_n[:, None] * dk + offs_dq[None, :],
                mask=mask_d, other=0.0,
            )
            if QK_F32:
                k_tile = k_load.to(tl.float32)
            else:
                k_tile = k_load
            scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp2((scores - m_new[:, None]) * LOG2E)
            alpha = tl.exp2((m_i - m_new) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
            v_load = tl.load(
                v_base + off_n[:, None] * dv + offs_dv[None, :],
                mask=mask_dv, other=0.0,
            )
            acc = tl.dot(p, v_load.to(tl.float32), acc * alpha[:, None])

        N_TAIL: tl.constexpr = sk_len - N_FULL * BLOCK_N
        if N_TAIL > 0:
            off_n = N_FULL * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = off_n.to(tl.float32) < f_sk          # [BLOCK_N]
            k_load = tl.load(
                k_base + off_n[:, None] * dk + offs_dq[None, :],
                mask=mask_n[:, None] & mask_d, other=0.0,
            )
            if QK_F32:
                k_tile = k_load.to(tl.float32)
            else:
                k_tile = k_load
            scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
            # CRITICAL: masked K rows load as 0 -> their scores are 0.0.
            # Without -inf masking, m_new picks up 0.0 whenever every real
            # score is negative, which breaks the online-softmax rescaling
            # (exp underflow -> l_i = 0 -> 0/0 = NaN).
            scores = tl.where(mask_n[None, :], scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp2((scores - m_new[:, None]) * LOG2E)
            alpha = tl.exp2((m_i - m_new) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
            v_load = tl.load(
                v_base + off_n[:, None] * dv + offs_dv[None, :],
                mask=mask_n[:, None] & mask_dv, other=0.0,
            )
            acc = tl.dot(p, v_load.to(tl.float32), acc * alpha[:, None])

        out = acc * (1.0 / l_i)[:, None]
        tl.store(
            o_base + off_m[:, None] * dv + offs_dv[None, :],
            out.to(in_dtype), mask=mask_mrow & mask_dv,
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
        BLOCK_DQ = triton.next_power_of_2(Dk)
        BLOCK_DV = triton.next_power_of_2(Dv)

        num_m_tiles = triton.cdiv(Sq, BLOCK_M)
        total_blocks = B * H * num_m_tiles

        grid = (CUBE_CORE_NUM,)
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
            CUBE_CORE_NUM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DQ=BLOCK_DQ,
            BLOCK_DV=BLOCK_DV,
            QK_F32=is_f32,
        )
        return out
