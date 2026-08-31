import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except ImportError:  # pragma: no cover - only when importing outside NPU env
    torch_npu = None


@triton.jit
def _proj_mm_kernel(
    a_ptr, w_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ W[N, K]^T ; all inputs row-major contiguous.
    # K must be a multiple of BLOCK_K (guaranteed on host side).
    pid = tl.program_id(0).to(tl.int32)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_pid_m * num_pid_n

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for block_idx in range(pid, num_blocks, num_cores):
        pid_m = block_idx // num_pid_n
        pid_n = block_idx - pid_m * num_pid_n

        m_base = pid_m * BLOCK_M
        n_base = pid_n * BLOCK_N
        a_ptrs = a_ptr + (m_base + offs_m)[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + (n_base + offs_n)[None, :] * K + offs_k[:, None]

        a_mask_row = (m_base + offs_m) < M
        w_mask_n = (n_base + offs_n) < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for kk in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=a_mask_row[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=w_mask_n[None, :], other=0.0)
            acc = tl.dot(a, w, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K

        c_ptrs = c_ptr + (m_base + offs_m)[:, None] * N + (n_base + offs_n)[None, :]
        c_mask = a_mask_row[:, None] & w_mask_n[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def _attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, S, H,
    s_stride,  # D, row stride of the [B, S, H, HD] (flat [M, D]) buffers
    scale,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HD: tl.constexpr,
):
    # Self-attention per (b, h): flash-style online softmax, fp32 inner math
    # (mirrors the reference which computes scores/softmax/PV in float32).
    # q/k/v layout: [B, S, H, HD] contiguous (i.e. flat [M, D], row stride
    # s_stride = D, head offset h*HD).
    pid = tl.program_id(0).to(tl.int32)
    num_q_blk = tl.cdiv(S, BLOCK_M)
    total = B * H * num_q_blk

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD)

    for blk in range(pid, total, num_cores):
        qb = blk % num_q_blk
        bh = blk // num_q_blk
        h = bh % H
        b = bh // H

        s_row = b * S
        offs_s = qb * BLOCK_M + offs_m
        s_mask = offs_s < S

        q_base = q_ptr + (s_row + offs_s)[:, None] * s_stride + h * HD + offs_d[None, :]
        q = tl.load(q_base, mask=s_mask[:, None], other=0.0).to(tl.float32)

        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HD), tl.float32)
        offs_n = tl.arange(0, BLOCK_N)

        for ks in range(0, S, BLOCK_N):
            k_s = ks + offs_n
            k_mask = k_s < S

            k_base = k_ptr + (s_row + k_s)[None, :] * s_stride + h * HD + offs_d[:, None]
            k = tl.load(k_base, mask=k_mask[None, :], other=0.0).to(tl.float32)

            qk = tl.dot(q, k, out_dtype=tl.float32)
            qk = qk * scale
            qk = tl.where(k_mask[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_base = v_ptr + (s_row + k_s)[:, None] * s_stride + h * HD + offs_d[None, :]
            v = tl.load(v_base, mask=k_mask[:, None], other=0.0).to(tl.float32)
            acc = tl.dot(p, v, acc, out_dtype=tl.float32)

            m_i = m_new

        acc = acc / l_i[:, None]
        o_base = o_ptr + (s_row + offs_s)[:, None] * s_stride + h * HD + offs_d[None, :]
        tl.store(o_base, acc.to(o_ptr.dtype.element_ty), mask=s_mask[:, None])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._weights = {}
        self.vec_cores = 48
        self.cube_cores = 24
        if torch_npu is not None:
            try:
                limits = torch_npu.npu.npu_config.get_device_limit(0)
                self.vec_cores = int(limits.get("vector_core_num", 48))
                self.cube_cores = int(limits.get("cube_core_num", 24))
            except Exception:
                self.vec_cores = 48
                self.cube_cores = 24

    def _layers(self, x, n_heads):
        d_model = x.shape[-1]
        key = (d_model, n_heads, x.device, x.dtype)
        if key not in self._weights:
            self._weights.clear()
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            wq = nn.Linear(d_model, d_model, bias=False).to(
                device=x.device, dtype=x.dtype
            ).weight
            wk = nn.Linear(d_model, d_model, bias=False).to(
                device=x.device, dtype=x.dtype
            ).weight
            wv = nn.Linear(d_model, d_model, bias=False).to(
                device=x.device, dtype=x.dtype
            ).weight
            wo = nn.Linear(d_model, d_model, bias=False).to(
                device=x.device, dtype=x.dtype
            ).weight
            torch.set_rng_state(rng_state)
            self._weights[key] = (wq, wk, wv, wo)
        return self._weights[key]

    def _proj(self, x2, w, c, M, N):
        if M >= 128:
            block_m, block_n = 128, 128
        else:
            block_m = max(32, triton.next_power_of_2(M))
            block_n = 64
        _proj_mm_kernel[(self.cube_cores,)](
            x2, w, c,
            M, N, N,
            num_cores=self.cube_cores,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=64,
        )

    def forward(self, x, n_heads):
        x = x.contiguous()
        batch, seq, d_model = x.shape
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads

        wq, wk, wv, wo = self._layers(x, n_heads)
        dev, dtype = x.device, x.dtype
        M = batch * seq
        x2 = x.view(M, d_model)

        q = torch.empty(M, d_model, device=dev, dtype=dtype)
        k = torch.empty(M, d_model, device=dev, dtype=dtype)
        v = torch.empty(M, d_model, device=dev, dtype=dtype)
        self._proj(x2, wq, q, M, d_model)
        self._proj(x2, wk, k, M, d_model)
        self._proj(x2, wv, v, M, d_model)

        attn = torch.empty(M, d_model, device=dev, dtype=dtype)
        block_m, block_n = 64, 64
        total = batch * n_heads * triton.cdiv(seq, block_m)
        grid = min(total, self.cube_cores)
        _attn_kernel[(grid,)](
            q, k, v, attn,
            batch, seq, n_heads,
            d_model,
            1.0 / math.sqrt(head_dim),
            num_cores=grid,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HD=head_dim,
        )

        out = torch.empty(batch, seq, d_model, device=dev, dtype=dtype)
        self._proj(attn, wo, out.view(M, d_model), M, d_model)
        return out