import math

import torch
import triton
import triton.language as tl


# Strided attention core (KernelBench 25):
#   q = k = v = x  (per-head slices)
#   query i attends only to keys j with j % stride == i % stride
#   scale = 1/sqrt(head_dim); softmax in fp32; output in input dtype.
#
# Design:
#   For a fixed residue r in [0, stride), queries at positions {r, r+R, r+2R, ...}
#   and keys at the same positions form a DENSE full-attention subproblem over the
#   subsequence s(t) = r + t*R (t in [0, L_r)).  The whole operation is therefore
#   R independent dense flash-attention subproblems (per batch, per head) over
#   strided rows of x, which exploits the mask sparsity (FLOPs ~= dense / R).
#
#   Single fused Triton kernel:
#     - grid is clamped to the CUBE core count (T1/G4); each program loops over
#       its share of (b, h, r, m_block) tiles;
#     - online softmax in fp32 over key blocks of the subsequence;
#     - fp32 dot products (matches reference which upcasts x to float32);
#     - results cast back to the input dtype on store.


@triton.jit
def strided_attention_kernel(
    x_ptr,
    out_ptr,
    S,            # seq_len (int32)
    D,            # d_model (int32)
    H,            # n_heads (int32)
    R,            # stride (int32)
    total_blocks, # total (b, h, r, m) tiles
    stride_b,     # S * D (elements)
    stride_s,     # D (elements)
    scale,        # 1 / sqrt(head_dim)
    log2e,
    num_pids: tl.constexpr,
    M_BLOCKS: tl.constexpr,   # cdiv(cdiv(S, R), BLOCK_M)
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,    # next_power_of_2(HEAD_DIM)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    offs_d = tl.arange(0, BLOCK_D).to(tl.int32)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int32)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int32)
    d_ok = offs_d < HEAD_DIM

    for block_idx in range(pid, total_blocks, num_pids):
        # decode block_idx -> (b, h, r, m_block)
        mb = block_idx % M_BLOCKS
        tmp = block_idx // M_BLOCKS
        r = tmp % R
        tmp2 = tmp // R
        h = tmp2 % H
        b = tmp2 // H

        # length of the residue-r subsequence
        L_r = tl.where(r < S, (S - 1 - r) // R + 1, 0)

        # ---- load query tile (subsequence rows, strided in GM) ----
        m0 = mb * BLOCK_M
        t_q = m0 + offs_m                 # subsequence indices
        s_q = r + t_q * R                 # global sequence positions
        row_q = b * stride_b + s_q * stride_s + h * HEAD_DIM
        q_mask = (t_q < L_r)[:, None] & d_ok[None, :]
        q = tl.load(
            x_ptr + row_q[:, None] + offs_d[None, :], mask=q_mask, other=0.0
        ).to(tl.float32)

        # ---- online softmax over key blocks ----
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for n0 in range(0, L_r, BLOCK_N):
            t_k = n0 + offs_n
            s_k = r + t_k * R
            row_k = b * stride_b + s_k * stride_s + h * HEAD_DIM
            key_mask = (t_k < L_r)[:, None] & d_ok[None, :]
            k = tl.load(
                x_ptr + row_k[:, None] + offs_d[None, :], mask=key_mask, other=0.0
            ).to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * scale
            qk = tl.where((t_k < L_r)[None, :], qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2((m_i - m_new) * log2e)
            p = tl.math.exp2((qk - m_new[:, None]) * log2e)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, k)
            m_i = m_new

        acc = acc / l_i[:, None]

        # ---- store output tile (same layout as x) ----
        out_mask = (t_q < L_r)[:, None] & d_ok[None, :]
        tl.store(
            out_ptr + row_q[:, None] + offs_d[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=out_mask,
        )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch_npu

            limit = torch_npu.npu.npu_config.get_device_limit(0)
            self.VEC_CORE_NUM = limit.get("vector_core_num", 48)
            self.CUBE_CORE_NUM = limit.get("cube_core_num", 24)
        except Exception:
            self.VEC_CORE_NUM = 48
            self.CUBE_CORE_NUM = 24

    def forward(self, x, n_heads, stride):
        if not x.is_contiguous():
            x = x.contiguous()

        batch, seq_len, d_model = x.shape
        head_dim = d_model // n_heads

        out = torch.empty_like(x)

        # block sizes
        BLOCK_D = triton.next_power_of_2(head_dim)
        BLOCK_M = 32
        BLOCK_N = 64

        # max residue subsequence length (residue 0)
        L_max = triton.cdiv(seq_len, stride)
        m_blocks = triton.cdiv(L_max, BLOCK_M)
        total_blocks = batch * n_heads * stride * m_blocks

        grid_size = min(total_blocks, self.CUBE_CORE_NUM)

        scale = 1.0 / math.sqrt(head_dim)
        log2e = 1.4426950408889634

        strided_attention_kernel[(grid_size,)](
            x,
            out,
            seq_len,
            d_model,
            n_heads,
            stride,
            total_blocks,
            seq_len * d_model,
            d_model,
            scale,
            log2e,
            num_pids=grid_size,
            M_BLOCKS=m_blocks,
            HEAD_DIM=head_dim,
            BLOCK_D=BLOCK_D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return out
