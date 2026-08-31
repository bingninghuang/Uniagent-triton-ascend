import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _build_layers(x, n_heads, projection_dim, cache):
    sequence, d_model = x.shape[1:]
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if not 0 < projection_dim <= sequence:
        raise ValueError("projection_dim must be in [1, sequence]")
    key = (sequence, d_model, n_heads, projection_dim, x.device, x.dtype)
    if key not in cache:
        cache.clear()
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        projections = tuple(
            nn.Linear(d_model, d_model, bias=False).to(
                device=x.device, dtype=x.dtype
            )
            for _ in range(4)
        )
        sequence_projections = (
            nn.Linear(sequence, projection_dim, bias=False).to(
                device=x.device, dtype=x.dtype
            ),
            nn.Linear(sequence, projection_dim, bias=False).to(
                device=x.device, dtype=x.dtype
            ),
        )
        cache[key] = projections + sequence_projections
        torch.set_rng_state(rng_state)
    return cache[key]


def _build_prep(q_proj, k_proj, v_proj, out_proj, key_reduce, value_reduce, cache, pkey):
    if pkey not in cache:
        wqkv = torch.cat(
            [q_proj.weight, k_proj.weight, v_proj.weight], dim=0
        )
        cache[pkey] = (
            wqkv,
            key_reduce.weight,
            value_reduce.weight,
            out_proj.weight,
        )
    return cache[pkey]


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ B[N, K]^T; A and B are both row-major, K contiguous.
    pid = tl.program_id(0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N
        m_row = block_m * BLOCK_M + offs_m
        n_row = block_n * BLOCK_N + offs_n
        m_mask = m_row < M
        n_mask = n_row < N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_off = k + offs_k
            k_mask = k_off < K
            a = tl.load(
                a_ptr + m_row[:, None] * K + k_off[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptr + n_row[None, :] * K + k_off[:, None],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            accumulator += tl.dot(a, b)
        c_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            c_ptr + m_row[:, None] * N + n_row[None, :],
            accumulator.to(c_ptr.dtype.element_ty),
            mask=c_mask,
        )
@triton.jit
def _kv_reduce_kernel(
    qkv_ptr,
    wk_ptr,
    wv_ptr,
    kred_ptr,
    vred_ptr,
    B,
    S,
    D,
    H,
    P,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # For each (b, h) pair:
    #   kred[bh, m, p] = sum_s k[b, s, h*Dh+m] * wk[p, s]
    #   vred[bh, p, m] = sum_s v[b, s, h*Dh+m] * wv[p, s]
    # qkv is (B*S, 3*D) row-major, row stride 3*D: q at col h*Dh,
    # k at col D + h*Dh, v at col 2*D + h*Dh.
    pid = tl.program_id(0)
    Dh = D // H
    D3 = 3 * D
    TOT = B * H
    offs_m = tl.arange(0, BLOCK_M)
    offs_p = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < Dh
    p_mask = offs_p < P
    store_mask = m_mask[:, None] & p_mask[None, :]
    for bh in range(pid, TOT, num_cores):
        b = bh // H
        h = bh % H
        k_base = qkv_ptr + b * S * D3 + D + h * Dh + offs_m[:, None]
        v_base = k_base + D
        acc_k = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s0 in range(0, S, BLOCK_K):
            s_off = s0 + offs_k
            s_mask = s_off < S
            a_mask = m_mask[:, None] & s_mask[None, :]
            ak = tl.load(
                k_base + s_off[None, :] * D3,
                mask=a_mask,
                other=0.0,
            )
            av = tl.load(
                v_base + s_off[None, :] * D3,
                mask=a_mask,
                other=0.0,
            )
            b_mask = s_mask[:, None] & p_mask[None, :]
            bk = tl.load(
                wk_ptr + offs_p[None, :] * S + s_off[:, None],
                mask=b_mask,
                other=0.0,
            )
            bv = tl.load(
                wv_ptr + offs_p[None, :] * S + s_off[:, None],
                mask=b_mask,
                other=0.0,
            )
            acc_k += tl.dot(ak, bk)
            acc_v += tl.dot(av, bv)
        tl.store(
            kred_ptr + bh * Dh * P + offs_m[:, None] * P + offs_p[None, :],
            acc_k,
            mask=store_mask,
        )
        tl.store(
            vred_ptr + bh * P * Dh + offs_p[None, :] * Dh + offs_m[:, None],
            acc_v,
            mask=store_mask,
        )

               
@triton.jit
def _attn_kernel(
    qkv_ptr,
    kred_ptr,
    vred_ptr,
    out_ptr,
    B,
    S,
    D,
    H,
    P,
    num_cores: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Per (b, h, s-tile): out[s, d] = softmax(q[s, :] @ kred^T * scale) @ vred
    pid = tl.program_id(0)
    Dh = D // H
    scale = 1.0 / tl.sqrt(tl.full((1,), 1, tl.float32) * Dh)
    S_TILES = tl.cdiv(S, BLOCK_S)
    TOT = B * H * S_TILES
    offs_s = tl.arange(0, BLOCK_S)
    offs_k = tl.arange(0, BLOCK_K)
    offs_p = tl.arange(0, BLOCK_P)
    offs_d = tl.arange(0, BLOCK_D)
    k_mask = offs_k < Dh
    p_mask = offs_p < P
    d_mask = offs_d < Dh
    for tid in range(pid, TOT, num_cores):
        st = tid % S_TILES
        bh = tid // S_TILES
        b = bh // H
        h = bh % H
        s0 = st * BLOCK_S
        s_off = s0 + offs_s
        s_mask = s_off < S
        q = tl.load(
            qkv_ptr + (b * S + s_off)[:, None] * (3 * D) + h * Dh + offs_k[None, :],
            mask=s_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        kr = tl.load(
            kred_ptr + bh * Dh * P + offs_k[:, None] * P + offs_p[None, :],
            mask=k_mask[:, None] & p_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, kr)
        scores = scores * scale
        scores = tl.where(p_mask[None, :], scores, -1.0e30)
        row_max = tl.max(scores, axis=1)
        row_max = tl.where(s_mask, row_max, 0.0)
        exp_v = tl.exp(scores - row_max[:, None])
        exp_v = tl.where(p_mask[None, :], exp_v, 0.0)
        row_sum = tl.sum(exp_v, axis=1)
        row_sum = tl.where(row_sum > 0.0, row_sum, 1.0)
        w = exp_v / row_sum[:, None]
        vr = tl.load(
            vred_ptr + bh * P * Dh + offs_p[:, None] * Dh + offs_d[None, :],
            mask=p_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.dot(w, vr)
        tl.store(
            out_ptr + (b * S + s_off)[:, None] * D + h * Dh + offs_d[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=s_mask[:, None] & d_mask[None, :],
        )

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}
        self._prep = {}
        cores = 20
        try:
            import torch_npu

            limits = torch_npu.npu.npu_config.get_device_limit(0)
            for key in ("cube_core_num", "ai_core_num", "aicore_num"):
                val = limits.get(key)
                if val:
                    cores = int(val)
                    break
        except Exception:
            pass
        self.CUBE_CORE_NUM = max(1, cores)

    def forward(self, x, n_heads, projection_dim):
        q_proj, k_proj, v_proj, out_proj, key_reduce, value_reduce = (
            _build_layers(x, n_heads, projection_dim, self._cache)
        )
        batch, sequence, d_model = x.shape
        head_dim = d_model // n_heads
        B, S, D, H, P, Dh = batch, sequence, d_model, n_heads, projection_dim, head_dim
        M = B * S
        x = x.contiguous()
        dev, dt = x.device, x.dtype

        pkey = (S, D, H, P, dev, dt)
        if pkey in self._prep:
            wqkv, wkr, wvr, wo = self._prep[pkey]
        else:
            wqkv, wkr, wvr, wo = _build_prep(
                q_proj,
                k_proj,
                v_proj,
                out_proj,
                key_reduce,
                value_reduce,
                self._prep,
                pkey,
            )

        # 1) fused q*k*v projection: (M, D) @ (3D, D)^T -> (M, 3D)
        qkv = torch.empty((M, 3 * D), device=dev, dtype=dt)
        BM, BN, BK = 128, 128, 128
        cores = self.CUBE_CORE_NUM
        _gemm_kernel[(cores,)](
            x, wqkv, qkv, M, 3 * D, D, cores, BM, BN, BK
        )

        # 2) sequence reduction of k and v
        kred = torch.empty((B * H, Dh, P), device=dev, dtype=dt)
        vred = torch.empty((B * H, P, Dh), device=dev, dtype=dt)
        rbmp = triton.next_power_of_2(head_dim)
        RBM = rbmp if rbmp >= 16 else 16
        rbnp = triton.next_power_of_2(P)
        RBN = rbnp if rbnp >= 16 else 16
        rbkn = triton.next_power_of_2(S)
        RBK = rbkn if rbkn < 128 else 128
        if RBK < 16:
            RBK = 16
        _kv_reduce_kernel[(cores,)](
            qkv, wkr, wvr, kred, vred, B, S, D, H, P, cores, RBM, RBN, RBK
        )

        # 3) attention: softmax(q @ kred^T / sqrt(Dh)) @ vred
        attn = torch.empty((B, S, D), device=dev, dtype=dt)
        abkp = triton.next_power_of_2(head_dim)
        ABK = abkp if abkp >= 16 else 16
        abnp = triton.next_power_of_2(P)
        ABP = abnp if abnp >= 16 else 16
        aodp = triton.next_power_of_2(head_dim)
        AOD = aodp if aodp >= 16 else 16
        _attn_kernel[(cores,)](
            qkv, kred, vred, attn, B, S, D, H, P, cores, 16, ABK, ABP, AOD
        )

        # 4) output projection: (M, D) @ (D, D)^T -> (M, D)
        out = torch.empty((B, S, D), device=dev, dtype=dt)
        _gemm_kernel[(cores,)](
            attn, wo, out, M, D, D, cores, BM, BN, BK
        )
        return out

