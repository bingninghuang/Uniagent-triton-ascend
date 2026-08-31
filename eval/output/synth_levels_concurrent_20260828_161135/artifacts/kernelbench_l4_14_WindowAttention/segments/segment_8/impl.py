import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: generic GEMM  C[M, N] = A[M, K] @ W[N, K]^T + bias[N]
# All math in fp32 (inputs may be fp16/bf16, upcast on load).
# A rows may be 2-level strided (e.g. x of shape (B, Nx, C)):
#   row m = b * Nx + i  ->  offset b*sxa + i*sxb (+ k*sxk)
# ---------------------------------------------------------------------------
@triton.jit
def _gemm_kernel(a_ptr, w_ptr, bias_ptr, c_ptr,
                 M, N: tl.constexpr, K, Nx: tl.constexpr,
                 sxa: tl.constexpr, sxb: tl.constexpr, sxk: tl.constexpr,
                 sow: tl.constexpr, sok: tl.constexpr,
                 num_pids: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    total_blocks = tl.cdiv(M, BLOCK_M) * num_n
    per = total_blocks // num_pids
    rem = total_blocks - per * num_pids
    start = pid * per + tl.minimum(pid, rem)
    end = start + per + tl.where(pid < rem, 1, 0)
    for bidx in range(start, end):
        bm = bidx // num_n
        bn = bidx - (bidx // num_n) * num_n
        offs_m = bm * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = bn * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mmask = offs_m < M
        nmask = offs_n < N
        # map flat row m -> (b, i)
        m_div = offs_m // Nx
        a_row = m_div * sxa + (offs_m - m_div * Nx) * sxb
        a_ptrs = a_ptr + a_row[:, None] + offs_k[None, :] * sxk
        w_ptrs = w_ptr + offs_n[:, None] * sow + offs_k[None, :] * sok
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs + k0 * BLOCK_K * sxk, mask=mmask[:, None], other=0.0).to(tl.float32)
            w = tl.load(w_ptrs + k0 * BLOCK_K * sok, mask=nmask[:, None], other=0.0).to(tl.float32)
            acc = tl.dot(a, tl.trans(w), acc)
        bias = tl.load(bias_ptr + offs_n, mask=nmask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mmask[:, None] & nmask[None, :])


# ---------------------------------------------------------------------------
# Kernel 2: precompute relative-position bias
#   bias[h, i, j] = rel_table[idx(i,j), h]
#   idx(i,j) = (hi - hj + Wh - 1) * (2 Ww - 1) + (wi - wj + Ww - 1)
#            = row_term(i) - col_term(j)
# bias buffer layout: (H, N, NP) fp32, NP = N padded to a multiple of 64.
# ---------------------------------------------------------------------------
@triton.jit
def _bias_kernel(tbl_ptr, bias_ptr,
                 N: tl.constexpr, H: tl.constexpr, Wh: tl.constexpr, Ww: tl.constexpr, NP: tl.constexpr,
                 t_s0: tl.constexpr, t_s1: tl.constexpr,
                 num_pids: tl.constexpr,
                 BI: tl.constexpr, BJ: tl.constexpr):
    pid = tl.program_id(0)
    nbi = tl.cdiv(N, BI)
    nbj = tl.cdiv(N, BJ)
    nb = nbi * nbj
    total = H * nb
    per = total // num_pids
    rem = total - per * num_pids
    start = pid * per + tl.minimum(pid, rem)
    end = start + per + tl.where(pid < rem, 1, 0)
    for bidx in range(start, end):
        h = bidx // nb
        t = bidx - (bidx // nb) * nb
        bi = t // nbj
        bj = t - (t // nbj) * nbj
        ii = bi * BI + tl.arange(0, BI)
        jj = bj * BJ + tl.arange(0, BJ)
        im = ii < N
        jm = jj < N
        hi = ii // Ww
        hj = jj // Ww
        row_term = (hi + Wh - 1) * (2 * Ww - 1) + (ii - hi * Ww) + (Ww - 1)
        col_term = hj * (2 * Ww - 1) + (jj - hj * Ww)
        idx = (row_term[:, None] - col_term[None, :]).to(tl.int32)
        val = tl.load(tbl_ptr + idx * t_s0 + h * t_s1,
                      mask=im[:, None] & jm[None, :], other=0.0).to(tl.float32)
        o_ptrs = bias_ptr + (h * N + ii[:, None]) * NP + jj[None, :]
        tl.store(o_ptrs, val, mask=im[:, None] & jm[None, :])


# ---------------------------------------------------------------------------
# Kernel 3: flash (online-softmax) window attention, per (b, h)
# q/k/v are strided views into qkv buffer (B, N, 3C):
#   q[b, h, i, d] -> qkv[(b*N + i), h*D + d]
#   k[b, h, i, d] -> qkv[(b*N + i), C + h*D + d]
#   v[b, h, i, d] -> qkv[(b*N + i), 2*C + h*D + d]
# output written directly head-merged: out[(b*N + i), h*D + d]  (B, N, C)
# ---------------------------------------------------------------------------
@triton.jit
def _attn_kernel(qkv_ptr, bias_ptr, o_ptr,
                 B: tl.constexpr, N: tl.constexpr, C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, NP: tl.constexpr,
                 scale: tl.constexpr,
                 num_pids: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    n_mb = tl.cdiv(N, BM)
    total = B * H * n_mb
    per = total // num_pids
    rem = total - per * num_pids
    start = pid * per + tl.minimum(pid, rem)
    end = start + per + tl.where(pid < rem, 1, 0)
    for bidx in range(start, end):
        b = bidx // (n_mb * H)
        t = bidx - b * (n_mb * H)
        h = t // n_mb
        mb = t - (t // n_mb) * n_mb
        i0 = mb * BM
        offs_i = i0 + tl.arange(0, BM)
        offs_d = tl.arange(0, BD)
        im = offs_i < N
        dm = offs_d < D
        row_base = qkv_ptr + b * N * (3 * C)
        qbase = h * D
        kbase = C + h * D
        vbase = 2 * C + h * D
        q = tl.load(row_base + offs_i[:, None] * (3 * C) + qbase + offs_d[None, :],
                    mask=im[:, None] & dm[None, :], other=0.0).to(tl.float32) * scale
        m_i = tl.full((BM,), float('-inf'), dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, BD), dtype=tl.float32)
        for j0 in range(0, N, BN):
            offs_jv = j0 + tl.arange(0, BN)
            jm = offs_jv < N
            k = tl.load(row_base + offs_jv[:, None] * (3 * C) + kbase + offs_d[None, :],
                        mask=jm[:, None] & dm[None, :], other=0.0).to(tl.float32)
            qk = tl.dot(q, tl.trans(k))
            bias = tl.load(bias_ptr + h * N * NP + offs_i[:, None] * NP + offs_jv[None, :],
                           mask=im[:, None] & jm[None, :], other=0.0)
            qk = qk + bias
            qk = tl.where(jm[None, :], qk, float('-inf'))
            row_max = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, row_max)
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(row_base + offs_jv[:, None] * (3 * C) + vbase + offs_d[None, :],
                        mask=jm[:, None] & dm[None, :], other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(p, v)
            m_i = m_new
        out = acc / l_i[:, None]
        o_ptrs = o_ptr + (b * N + offs_i[:, None]) * C + h * D + offs_d[None, :]
        tl.store(o_ptrs, out, mask=im[:, None] & dm[None, :])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.VEC_CORE_NUM = 48
        self.CUBE_CORE_NUM = 24
        try:
            import torch_npu
            dev_limit = torch_npu.npu.npu_config.get_device_limit(0)
            if isinstance(dev_limit, dict):
                self.VEC_CORE_NUM = dev_limit.get('vector_core_num', 48)
                self.CUBE_CORE_NUM = dev_limit.get('cube_core_num', 24)
        except Exception:
            pass

    def forward(self, x, window_size, num_heads, scale,
                qkv_w, qkv_b, proj_w, proj_b, rel_table):
        B, N, C = x.shape
        if isinstance(window_size, int):
            Wh, Ww = window_size, window_size
        else:
            Wh, Ww = tuple(window_size)
        H = num_heads
        D = C // H
        dev, dt = x.device, x.dtype

        qkv_buf = torch.empty((B, N, 3 * C), device=dev, dtype=torch.float32)
        NP = (N + 63) // 64 * 64
        bias_buf = torch.empty((H, N, NP), device=dev, dtype=torch.float32)
        attn_buf = torch.empty((B, N, C), device=dev, dtype=torch.float32)
        y = torch.empty((B, N, C), device=dev, dtype=dt)

        # 1) relative bias (H, N, NP): grid = min(actual blocks, vec cores)
        nbi = (N + 63) // 64
        _bias_blocks = H * nbi * nbi
        _bias_grid = self.VEC_CORE_NUM if _bias_blocks > self.VEC_CORE_NUM else _bias_blocks
        _bias_kernel[(_bias_grid,)](
            rel_table, bias_buf, N, H, Wh, Ww, NP,
            rel_table.stride(0), rel_table.stride(1), _bias_grid,
            BI=64, BJ=64)

        # 2) QKV projection (B*N, C) @ (3C, C)^T -> (B*N, 3C) fp32
        _qkv_blocks = ((B * N + 63) // 64) * ((3 * C + 127) // 128)
        qkv_grid = self.CUBE_CORE_NUM if _qkv_blocks > self.CUBE_CORE_NUM else _qkv_blocks
        _gemm_kernel[(qkv_grid,)](
            x, qkv_w, qkv_b, qkv_buf,
            B * N, 3 * C, C, N,
            x.stride(0), x.stride(1), x.stride(2),
            qkv_w.stride(0), qkv_w.stride(1), qkv_grid,
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64)

        # 3) window attention (flash) -> head-merged (B*N, C) fp32
        if D <= 16:
            BM, BN, BD = 64, 64, 16
        elif D <= 32:
            BM, BN, BD = 64, 64, 32
        elif D <= 64:
            BM, BN, BD = 64, 32, 64
        elif D <= 128:
            BM, BN, BD = 32, 32, 128
        elif D <= 256:
            BM, BN, BD = 16, 32, 256
        else:
            BM, BN, BD = 16, 16, D
        _attn_blocks = B * H * ((N + BM - 1) // BM)
        attn_grid = self.CUBE_CORE_NUM if _attn_blocks > self.CUBE_CORE_NUM else _attn_blocks
        _attn_kernel[(attn_grid,)](
            qkv_buf, bias_buf, attn_buf,
            B, N, C, D, H, NP, scale, attn_grid,
            BM=BM, BN=BN, BD=BD)

        # 4) output projection (B*N, C) @ (C, C)^T -> (B*N, C) x.dtype
        _proj_blocks = ((B * N + 63) // 64) * ((C + 127) // 128)
        proj_grid = self.CUBE_CORE_NUM if _proj_blocks > self.CUBE_CORE_NUM else _proj_blocks
        _gemm_kernel[(proj_grid,)](
            attn_buf, proj_w, proj_b, y,
            B * N, C, C, N,
            attn_buf.stride(0), attn_buf.stride(1), attn_buf.stride(2),
            proj_w.stride(0), proj_w.stride(1), proj_grid,
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64)

        return y
                