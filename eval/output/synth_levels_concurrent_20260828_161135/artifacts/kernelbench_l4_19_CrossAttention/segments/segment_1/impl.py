import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _vec_core_num():
    try:
        import torch_npu

        return int(
            torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
        )
    except Exception:
        return 48


VEC_CORE_NUM = _vec_core_num()


@triton.jit
def gemm_kernel(
    a_ptr, w_ptr, c_ptr,
    M, N, K, total_blocks, num_cores,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ W[N, K]^T   (W is the row-major [N, K] linear weight)
    pid = tl.program_id(0).to(tl.int32)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for block_id in range(pid, total_blocks, num_cores):
        block_m = block_id // num_pid_n
        block_n = block_id % num_pid_n
        row_m = block_m * BLOCK_M + offs_m
        row_n = block_n * BLOCK_N + offs_n
        m_mask = row_m[:, None] < M
        n_mask = row_n[None, :] < N
        a_ptrs = a_ptr + row_m[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + row_n[:, None] * K + offs_k[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_mask = (k_start + offs_k)[None, :] < K
            a_blk = tl.load(a_ptrs, mask=m_mask & k_mask, other=0.0)
            w_blk = tl.load(w_ptrs, mask=n_mask & k_mask, other=0.0)
            acc = tl.dot(a_blk, tl.trans(w_blk), acc)
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        c_ptrs = c_ptr + row_m[:, None] * N + row_n[None, :]
        c_mask = (row_m[:, None] < M) & (row_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    Lq, Lk, H, D, scale, total_blocks, num_cores,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, DH: tl.constexpr,
    LOW_PREC: tl.constexpr,
):
    # q/k/v/o are [B*L, D] buffers laid out so that head h, dim d of token l
    # lives at offset (b*L + l)*D + h*DH + d.  (i.e. [B, L, H, DH] contiguous)
    pid = tl.program_id(0).to(tl.int32)
    num_q_blocks = tl.cdiv(Lq, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < DH
    for block_id in range(pid, total_blocks, num_cores):
        bh = block_id // num_q_blocks
        q_blk_id = block_id % num_q_blocks
        b = bh // H
        h = bh % H
        offs_m = q_blk_id * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < Lq
        q_base = b * (Lq * D) + h * DH
        q_offs = q_base + offs_m[:, None] * D + offs_d[None, :]
        q = tl.load(
            q_ptr + q_offs, mask=m_mask[:, None] & d_mask[None, :], other=0.0
        )
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        kv_base = b * (Lk * D) + h * DH
        for start_n in range(0, Lk, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < Lk
            kv_offs = kv_base + offs_n[:, None] * D + offs_d[None, :]
            k = tl.load(
                k_ptr + kv_offs, mask=n_mask[:, None] & d_mask[None, :], other=0.0
            )
            s = tl.dot(q, tl.trans(k))
            s = s * scale
            s = tl.where(n_mask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            v = tl.load(
                v_ptr + kv_offs, mask=n_mask[:, None] & d_mask[None, :], other=0.0
            )
            if LOW_PREC:
                acc = tl.dot(p.to(v.dtype), v, acc)
            else:
                acc = tl.dot(p, v, acc)
            m_i = m_new
        acc = acc / l_i[:, None]
        o_offs = q_base + offs_m[:, None] * D + offs_d[None, :]
        tl.store(
            o_ptr + o_offs,
            acc.to(o_ptr.dtype.element_ty),
            mask=m_mask[:, None] & d_mask[None, :],
        )


def _launch_gemm(a, w, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    total_blocks = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid_size = min(total_blocks, VEC_CORE_NUM)
    gemm_kernel[(grid_size,)](
        a, w, c, M, N, K, total_blocks, grid_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def _layers(self, x, n_heads):
        d_model = x.shape[-1]
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        key = (d_model, n_heads, x.device, x.dtype)
        if key not in self._cache:
            self._cache.clear()
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            self._cache[key] = tuple(
                nn.Linear(d_model, d_model, bias=False).to(
                    device=x.device, dtype=x.dtype
                )
                for _ in range(4)
            )
            torch.set_rng_state(rng_state)
        return self._cache[key]

    def forward(self, query, context, n_heads):
        torch.manual_seed(42)
        query, context = query.contiguous(), context.contiguous()
        q_proj, k_proj, v_proj, out_proj = self._layers(query, n_heads)
        batch, query_length, d_model = query.shape
        key_length = context.shape[1]
        head_dim = d_model // n_heads

        q2d = query.view(batch * query_length, d_model)
        c2d = context.view(batch * key_length, d_model)

        qbuf = torch.empty(
            (batch * query_length, d_model), device=query.device, dtype=query.dtype
        )
        kbuf = torch.empty(
            (batch * key_length, d_model), device=query.device, dtype=query.dtype
        )
        vbuf = torch.empty(
            (batch * key_length, d_model), device=query.device, dtype=query.dtype
        )
        obuf = torch.empty(
            (batch * query_length, d_model), device=query.device, dtype=query.dtype
        )
        out2d = torch.empty(
            (batch * query_length, d_model), device=query.device, dtype=query.dtype
        )

        _launch_gemm(q2d, q_proj.weight, qbuf, batch * query_length, d_model,
                     d_model, 64, 64, 64)
        _launch_gemm(c2d, k_proj.weight, kbuf, batch * key_length, d_model,
                     d_model, 64, 64, 64)
        _launch_gemm(c2d, v_proj.weight, vbuf, batch * key_length, d_model,
                     d_model, 64, 64, 64)

        BLOCK_M, BLOCK_N = 32, 64
        BLOCK_D = triton.next_power_of_2(head_dim)
        n_q_blocks = triton.cdiv(query_length, BLOCK_M)
        total_blocks = n_q_blocks * batch * n_heads
        grid_size = min(total_blocks, VEC_CORE_NUM)
        attention_kernel[(grid_size,)](
            qbuf, kbuf, vbuf, obuf,
            query_length, key_length, n_heads, d_model,
            1.0 / math.sqrt(head_dim), total_blocks, grid_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, DH=head_dim,
            LOW_PREC=query.dtype != torch.float32,
        )

        _launch_gemm(obuf, out_proj.weight, out2d, batch * query_length, d_model,
                     d_model, 64, 64, 64)
        return out2d.view(batch, query_length, d_model)
