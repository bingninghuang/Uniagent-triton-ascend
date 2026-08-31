import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _flex_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sm_scale,
    B, H, S_q, S_k,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    num_m_blocks,
    num_cores,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    total_blocks = B * H * num_m_blocks

    for block_idx in range(pid, total_blocks, num_cores):
        # decode block_idx -> (b, h, m_block); block ordering: ((b*H + h) * num_m_blocks + m_block)
        bh = block_idx // num_m_blocks
        m_block = block_idx - bh * num_m_blocks
        b = bh // H
        h = bh - b * H
        h_kv = h // G

        m_start = m_block * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < S_q

        q_ptrs = q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        hi = S_k
        if IS_CAUSAL:
            # key j is valid for row i iff j <= i + (S_k - S_q);
            # rows in this block are [m_start, m_start+BLOCK_M)
            hi = tl.minimum(S_k, m_start + BLOCK_M + (S_k - S_q))

        for n_start in range(0, hi, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S_k

            k_ptrs = k_ptr + b * stride_kb + h_kv * stride_kh + offs_d[:, None] + offs_n[None, :] * stride_kn
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)  # [BLOCK_D, BLOCK_N]

            qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N], fp32 acc
            qk = qk * sm_scale

            if IS_CAUSAL:
                causal_ok = (offs_m[:, None] + (S_k - S_q)) >= offs_n[None, :]
                ok = causal_ok & mask_n[None, :]
            else:
                ok = mask_n[None, :]

            s = tl.where(ok, qk, float('-inf'))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            p = tl.where(ok, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            v_ptrs = v_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :]
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # [BLOCK_N, BLOCK_D]

            acc = acc * alpha[:, None]
            acc = tl.dot(p, v.to(tl.float32), acc)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = o_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._cache = {}
        try:
            import torch_npu
            limit = torch_npu.npu.npu_config.get_device_limit(0)
            self.CUBE_CORE_NUM = int(limit.get('cube_core_num', 24))
        except Exception:
            self.CUBE_CORE_NUM = 24

    def forward(self, query, key, value, is_causal=False, enable_gqa=False):
        B, H, S_q, D = query.shape
        H_kv = key.shape[1]
        S_k = key.shape[2]

        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        o = torch.empty_like(query)

        G = (H // H_kv) if (enable_gqa and H != H_kv) else 1
        sm_scale = 1.0 / math.sqrt(D)

        BLOCK_M = 64
        BLOCK_N = 64
        num_m_blocks = triton.cdiv(S_q, BLOCK_M)
        total_blocks = B * H * num_m_blocks
        grid_size = min(total_blocks, self.CUBE_CORE_NUM)

        _flex_attn_fwd_kernel[(grid_size,)](
            q, k, v, o,
            sm_scale,
            B, H, S_q, S_k,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            num_m_blocks,
            grid_size,
            G=G,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=D,
            IS_CAUSAL=bool(is_causal),
        )
        return o
