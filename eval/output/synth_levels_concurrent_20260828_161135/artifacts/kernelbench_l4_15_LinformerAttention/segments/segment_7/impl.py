import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_cube_cores():
    try:
        import torch_npu

        limits = torch_npu.npu.npu_config.get_device_limit(0)
        for key in ("cube_core_num", "ai_core_num", "aicore_num"):
            val = limits.get(key)
            if val:
                return int(val)
        val = limits.get("vector_core_num")
        if val:
            return max(1, int(val) // 2)
    except Exception:
        pass
    return 24


CUBE_CORE_NUM = _get_cube_cores()


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
    stride_a,
    stride_b,
    num_pids,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ B[N, K]^T, all row-major, k contiguous.
    pid = tl.program_id(0).to(tl.int32)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total = tl.cdiv(M, BLOCK_M) * num_pid_n
    for block_idx in range(pid, total, num_pids):
        pid_m = block_idx // num_pid_n
        pid_n = block_idx % num_pid_n
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_a + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * stride_b + offs_k[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kmask = (k0 + offs_k) < K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & kmask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=kmask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_a
            b_ptrs += BLOCK_K
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
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
    num_pids,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # For each (b, h) pair:
    #   k_red[m, p] = sum_s k[b, s, h*Dh+m] * wk[p, s]
    #   v_red[p, m] = sum_s v[b, s, h*Dh+m] * wv[p, s]
    # qkv is (B*S, 3*D) row-major, row stride 3*D: q at col h*Dh,
    # k at col D + h*Dh, v at col 2*D + h*Dh.
    pid = tl.program_id(0).to(tl.int32)
    Dh = D // H
    D3 = 3 * D
    total = B * H
    offs_m = tl.arange(0, BLOCK_M)
    offs_p = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mmask = offs_m < Dh
    pmask = offs_p < P
    for bh in range(pid, total, num_pids):
        b = bh // H
        h = bh % H
        k_base = qkv_ptr + b * S * D3 + D + h * Dh + offs_m[:, None]
        v_base = k_base + D
        wk_ptrs = wk_ptr + offs_p[None, :] * S + offs_k[:, None]
        wv_ptrs = wv_ptr + offs_p[None, :] * S + offs_k[:, None]
        acc_k = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s0 in range(0, S, BLOCK_K):
            kmask = (s0 + offs_k) < S
            am = mmask[:, None] & kmask[None, :]
            ak = tl.load(
                k_base + (s0 + offs_k)[None, :] * D3,
                mask=am,
                other=0.0,
            )
            av = tl.load(
                v_base + (s0 + offs_k)[None, :] * D3,
                mask=am,
                other=0.0,
            )
            bk = tl.load(wk_ptrs, mask=kmask[:, None], other=0.0)
            bv = tl.load(wv_ptrs, mask=kmask[:, None], other=0.0)
            acc_k = tl.dot(ak, bk, acc_k, out_dtype=tl.float32)
            acc_v = tl.dot(av, bv, acc_v, out_dtype=tl.float32)
            wk_ptrs += BLOCK_K
            wv_ptrs += BLOCK_K
        tl.store(
            kred_ptr + bh * Dh * P + offs_m[:, None] * P + offs_p[None, :],
            acc_k,
            mask=mmask[:, None] & pmask[None, :],
        )
        tl.store(
            vred_ptr + bh * P * Dh + offs_p[None, :] * Dh + offs_m[:, None],
            acc_v,
            mask=mmask[:, None] & pmask[None, :],
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
    num_pids,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OD: tl.constexpr,
):
    # Per (b, h, s-tile): out[s, d] = softmax(q[s, :] @ k_red^T / sqrt(Dh)) @ v_red
    pid = tl.program_id(0).to(tl.int32)
    Dh = D // H
    scale = 1.0 / tl.sqrt(Dh.to(tl.float32))
    s_tiles = tl.cdiv(S, BLOCK_S)
    total = B * H * s_tiles
    offs_s = tl.arange(0, BLOCK_S)
    offs_kd = tl.arange(0, BLOCK_K)
    offs_p = tl.arange(0, BLOCK_P)
    offs_od = tl.arange(0, BLOCK_OD)
    dmask = offs_kd < Dh
    dm2 = offs_od < Dh
    pmask = offs_p < P
    for t in range(pid, total, num_pids):
        bh = t // s_tiles
        st = t % s_tiles
        b = bh // H
        h = bh % H
        s0 = st * BLOCK_S
        smask = (s0 + offs_s) < S
        q_ptrs = (
            qkv_ptr
            + (b * S + s0 + offs_s)[:, None] * (3 * D)
            + h * Dh
            + offs_kd[None, :]
        )
        q = tl.load(q_ptrs, mask=smask[:, None] & dmask[None, :], other=0.0)
        q = q.to(tl.float32)
        kr_ptrs = kred_ptr + bh * Dh * P + offs_kd[:, None] * P + offs_p[None, :]
        kr = tl.load(kr_ptrs, mask=dmask[:, None] & pmask[None, :], other=0.0)
        scores = tl.dot(q, kr, out_dtype=tl.float32)
        scores = scores * scale
        scores = tl.where(pmask[None, :], scores, float("-inf"))
        row_max = tl.max(scores, axis=1)
        row_max = tl.where(smask, row_max, 0.0)
        exp_v = tl.exp(scores - row_max[:, None])
        exp_v = tl.where(pmask[None, :], exp_v, 0.0)
        row_sum = tl.sum(exp_v, axis=1)
        row_sum = tl.where(row_sum > 0.0, row_sum, 1.0)
        w = exp_v / row_sum[:, None]
        vr_ptrs = vred_ptr + bh * P * Dh + offs_p[:, None] * Dh + offs_od[None, :]
        vr = tl.load(vr_ptrs, mask=pmask[:, None] & dm2[None, :], other=0.0)
        o = tl.dot(w, vr, out_dtype=tl.float32)
        o_ptrs = out_ptr + (b * S + s0 + offs_s)[:, None] * D + h * Dh + offs_od[None, :]
        tl.store(o_ptrs, o.to(out_ptr.dtype.element_ty), mask=smask[:, None] & dm2[None, :])

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}
        self._prep = {}

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
        nblocks = triton.cdiv(M, BM) * triton.cdiv(3 * D, BN)
        grid1 = nblocks if nblocks < CUBE_CORE_NUM else CUBE_CORE_NUM
        _gemm_kernel[(grid1,)](
            x, wqkv, qkv, M, 3 * D, D, D, D, grid1, BM, BN, BK
        )

        # 2) sequence reduction of k and v
        kred = torch.empty((B * H, Dh, P), device=dev, dtype=torch.float32)
        vred = torch.empty((B * H, P, Dh), device=dev, dtype=torch.float32)
        rbmp = triton.next_power_of_2(head_dim)
        RBM = rbmp if rbmp >= 16 else 16
        rbnp = triton.next_power_of_2(P)
        RBN = rbnp if rbnp >= 16 else 16
        seff = S if S < 128 else 128
        rbkn = triton.next_power_of_2(seff)
        RBK = rbkn if rbkn >= 16 else 16
        grid2 = B * H if B * H < CUBE_CORE_NUM else CUBE_CORE_NUM
        _kv_reduce_kernel[(grid2,)](
            qkv, wkr, wvr, kred, vred, B, S, D, H, P, grid2, RBM, RBN, RBK
        )

        # 3) attention: softmax(q @ kred^T / sqrt(Dh)) @ vred
        attn = torch.empty((B, S, D), device=dev, dtype=dt)
        abkp = triton.next_power_of_2(head_dim)
        ABK = abkp if abkp >= 16 else 16
        abnp = triton.next_power_of_2(P)
        ABP = abnp if abnp >= 16 else 16
        aodp = triton.next_power_of_2(head_dim)
        AOD = aodp if aodp >= 16 else 16
        ABs = 16
        nblocks_a = B * H * triton.cdiv(S, ABs)
        grid3 = nblocks_a if nblocks_a < CUBE_CORE_NUM else CUBE_CORE_NUM
        _attn_kernel[(grid3,)](
            qkv,
            kred,
            vred,
            attn,
            B,
            S,
            D,
            H,
            P,
            grid3,
            ABs,
            ABK,
            ABP,
            AOD,
        )

        # 4) output projection: (M, D) @ (D, D)^T -> (M, D)
        out = torch.empty((B, S, D), device=dev, dtype=dt)
        nblocks = triton.cdiv(M, BM) * triton.cdiv(D, BN)
        grid4 = nblocks if nblocks < CUBE_CORE_NUM else CUBE_CORE_NUM
        _gemm_kernel[(grid4,)](
            attn, wo, out, M, D, D, D, D, grid4, BM, BN, BK
        )
        return out

