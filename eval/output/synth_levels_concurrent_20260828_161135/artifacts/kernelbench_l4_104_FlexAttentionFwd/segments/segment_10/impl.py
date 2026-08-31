import torch
import torch.nn as nn
import torch_npu
import triton
import triton.language as tl


@triton.jit
def flex_attention_fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    B, H, S_q, S_k, M_BLOCKS,
    NUM_CORES,
    IS_CAUSAL: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    task_total = B * H * M_BLOCKS
    chunk = (task_total - 1) // NUM_CORES + 1
    t_start = pid * chunk
    t_end = t_start + tl.minimum(chunk, task_total - t_start)

    for t in range(t_start, t_end):
        bh = t // M_BLOCKS
        m_idx = t - bh * M_BLOCKS
        h_idx = bh - (bh // H) * H
        b_idx = bh // H
        if G == 1:
            h_kv = h_idx
        else:
            h_kv = h_idx // G

        m_start = m_idx * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_m_f = offs_m.to(tl.float32)

        q_ptrs = (
            Q
            + b_idx * stride_qb
            + h_idx * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=(offs_m_f[:, None] < S_q), other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        sm_scale = 1.0 / tl.sqrt(tl.full([1], BLOCK_D, dtype=tl.float32))

        if IS_CAUSAL:
            hi = tl.minimum(S_k, m_start + BLOCK_M + (S_k - S_q))
        else:
            hi = S_k

        q_shift = S_k - S_q
        for n_start in range(0, hi, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            offs_n_f = offs_n.to(tl.float32)

            k_ptrs = (
                K
                + b_idx * stride_kb
                + h_kv * stride_kh
                + offs_d[:, None]
                + offs_n[None, :] * stride_kn
            )
            k = tl.load(k_ptrs, mask=(offs_n_f[None, :] < S_k), other=0.0)
            s = tl.dot(q, k)
            s = s * sm_scale

            if IS_CAUSAL:
                causal_ok = (offs_m_f[:, None] + q_shift) >= offs_n_f[None, :]
                s = tl.where(causal_ok, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)

            v_ptrs = (
                V
                + b_idx * stride_vb
                + h_kv * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_d[None, :]
            )
            v = tl.load(v_ptrs, mask=(offs_n_f[:, None] < S_k), other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p, v.to(tl.float32))
            m_i = m_new

        acc = acc / l_i[:, None]

        o_ptrs = (
            O
            + b_idx * stride_ob
            + h_idx * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :]
        )
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=(offs_m_f[:, None] < S_q))


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        try:
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get(
                "cube_core_num", 20
            )
        except Exception:
            self.CUBE_CORE_NUM = 20

    def forward(self, query, key, value, is_causal=False, enable_gqa=False):
        B, H, S_q, D = query.shape
        H_kv = key.shape[1]
        S_k = key.shape[2]

        out = torch.empty_like(query)
        G = H // H_kv if (enable_gqa and H != H_kv) else 1

        block_m = 64
        m_blocks = triton.cdiv(S_q, block_m)
        task_total = B * H * m_blocks
        if task_total < self.CUBE_CORE_NUM:
            grid_size = task_total
        else:
            grid_size = self.CUBE_CORE_NUM

        flex_attention_fwd_kernel[(grid_size,)](
            query, key, value, out,
            query.stride(0), query.stride(1), query.stride(2),
            key.stride(0), key.stride(1), key.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            B, H, S_q, S_k, m_blocks,
            self.CUBE_CORE_NUM,
            IS_CAUSAL=bool(is_causal),
            G=G,
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_D=D,
        )
        return out
