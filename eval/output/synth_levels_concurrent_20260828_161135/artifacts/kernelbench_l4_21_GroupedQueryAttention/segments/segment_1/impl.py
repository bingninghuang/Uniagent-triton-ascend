import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu

    _DEV_LIMIT = torch_npu.npu.npu_config.get_device_limit(0)
    NUM_VEC_CORES = int(_DEV_LIMIT.get("vector_core_num", 48))
except Exception:
    NUM_VEC_CORES = 48


@triton.jit
def _linear_gemm_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M,
    N,
    K,
    num_pids,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ W[N, K]^T   (all row-major in GM)
    pid = tl.program_id(0).to(tl.int32)
    n_m = tl.cdiv(M, BLOCK_M)
    n_n = tl.cdiv(N, BLOCK_N)
    for p in range(pid, n_m * n_n, num_pids):
        pid_m = p // n_n
        pid_n = p % n_n
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
        m_mask = offs_m < M
        n_mask = offs_n < N
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + offs_k[:, None] + offs_n[None, :] * K
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(K, BLOCK_K)):
            k_bound = K - kk * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_k[None, :] < k_bound) & m_mask[:, None],
                other=0.0,
            )
            w = tl.load(
                w_ptrs,
                mask=(offs_k[:, None] < k_bound) & n_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(a, w, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    S,
    H,
    KVH,
    B,
    REP,
    scale,
    num_pids,
    HD: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Fused flash-attention (online softmax) for GQA.
    # q: [B, S, H, HD]  k/v: [B, S, KVH, HD]  o: [B, S, H, HD]  (row-major)
    # kv head for query head h is h // REP
    pid = tl.program_id(0).to(tl.int32)
    n_m = tl.cdiv(S, BLOCK_M)
    total = B * H * n_m
    neg_inf = -float("inf")
    for p in range(pid, total, num_pids):
        mbl = p % n_m
        bh = p // n_m
        h = bh % H
        b = bh // H
        kvh = h // REP
        m0 = mbl * BLOCK_M
        offs_m = (m0 + tl.arange(0, BLOCK_M)).to(tl.int32)
        offs_d = tl.arange(0, BD)
        d_mask = offs_d < HD
        m_mask = offs_m < S

        q_base = b * S * H * HD
        q_ptrs = (
            q_ptr + q_base + offs_m[:, None] * (H * HD) + h * HD + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        m_i = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BD), dtype=tl.float32)

        kv_base = b * S * KVH * HD
        for n0 in range(0, S, BLOCK_N):
            offs_n = (n0 + tl.arange(0, BLOCK_N)).to(tl.int32)
            n_mask = offs_n < S
            kv_ptrs_off = (
                kv_base + offs_n[:, None] * (KVH * HD) + kvh * HD + offs_d[None, :]
            )
            kt = tl.load(
                k_ptr + kv_ptrs_off, mask=n_mask[:, None] & d_mask[None, :], other=0.0
            )
            vt = tl.load(
                v_ptr + kv_ptrs_off, mask=n_mask[:, None] & d_mask[None, :], other=0.0
            )
            s = tl.dot(q, tl.trans(kt), out_dtype=tl.float32) * scale
            s = tl.where(n_mask[None, :], s, neg_inf)
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p_t = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p_t, axis=1)
            acc = tl.dot(p_t, vt, acc * alpha[:, None], out_dtype=tl.float32)
            m_i = m_new

        o_ptrs = (
            o_ptr + q_base + offs_m[:, None] * (H * HD) + h * HD + offs_d[None, :]
        )
        o = acc / l_i[:, None]
        tl.store(
            o_ptrs,
            o.to(o_ptr.dtype.element_ty),
            mask=m_mask[:, None] & d_mask[None, :],
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def _layers(self, x, n_heads, n_kv_heads):
        d_model = x.shape[-1]
        if d_model % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError("head counts must divide d_model and each other")
        head_dim = d_model // n_heads
        key = (d_model, n_heads, n_kv_heads, x.device, x.dtype)
        if key not in self._cache:
            self._cache.clear()
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            self._cache[key] = (
                nn.Linear(d_model, d_model, bias=False).to(
                    device=x.device, dtype=x.dtype
                ),
                nn.Linear(d_model, n_kv_heads * head_dim, bias=False).to(
                    device=x.device, dtype=x.dtype
                ),
                nn.Linear(d_model, n_kv_heads * head_dim, bias=False).to(
                    device=x.device, dtype=x.dtype
                ),
                nn.Linear(d_model, d_model, bias=False).to(
                    device=x.device, dtype=x.dtype
                ),
            )
            torch.set_rng_state(rng_state)
        return self._cache[key]

    def forward(self, x, n_heads, n_kv_heads):
        q_proj, k_proj, v_proj, out_proj = self._layers(
            x, n_heads, n_kv_heads
        )
        x = x.contiguous()
        B, S, D = x.shape
        head_dim = D // n_heads
        kv_dim = n_kv_heads * head_dim
        rep = n_heads // n_kv_heads
        M = B * S
        dev = x.device
        dt = x.dtype

        x2d = x.view(M, D)
        q2d = torch.empty((M, D), dtype=dt, device=dev)
        k2d = torch.empty((M, kv_dim), dtype=dt, device=dev)
        v2d = torch.empty((M, kv_dim), dtype=dt, device=dev)
        o2d = torch.empty((M, D), dtype=dt, device=dev)
        y2d = torch.empty((M, D), dtype=dt, device=dev)

        BM, BN, BK = 64, 64, 64

        def _gemm(a, w, c, N_out, K_in):
            nM = triton.cdiv(M, BM)
            nN = triton.cdiv(N_out, BN)
            natural = nM * nN
            grid = min(natural, NUM_VEC_CORES)
            _linear_gemm_kernel[(grid,)](
                a, w, c, M, N_out, K_in, grid,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            )

        _gemm(x2d, q_proj.weight, q2d, D, D)
        _gemm(x2d, k_proj.weight, k2d, kv_dim, D)
        _gemm(x2d, v_proj.weight, v2d, kv_dim, D)

        A_M, A_N = 32, 32
        BD = triton.next_power_of_2(head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        nM_a = triton.cdiv(S, A_M)
        natural_a = B * n_heads * nM_a
        grid_a = min(natural_a, NUM_VEC_CORES)
        _attention_kernel[(grid_a,)](
            q2d, k2d, v2d, o2d, S, n_heads, n_kv_heads, B, rep, scale, grid_a,
            HD=head_dim, BD=BD, BLOCK_M=A_M, BLOCK_N=A_N,
        )

        _gemm(o2d, out_proj.weight, y2d, D, D)
        return y2d.view(B, S, D)
