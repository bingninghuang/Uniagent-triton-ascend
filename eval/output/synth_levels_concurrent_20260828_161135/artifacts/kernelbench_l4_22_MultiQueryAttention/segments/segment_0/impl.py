import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    torch_npu = None

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Generic tiled GEMM kernel: C[M, N] = A[M, K] @ W[N, K]^T (A row-major, W
# stored as [N, K] row-major i.e. transposed matmul).  Launched with a fixed
# grid of CUBE cores; each core processes several output tiles via an
# interleaved loop.
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    a_ptr, w_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BN = NUM_BLOCKS_N
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BN
        block_n = block_idx % NUM_BN

        offs_m = (block_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        offs_n = (block_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]

        a_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N
        k_mask = offs_k < K

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=a_mask & (k_mask[None, :] < K - k * BLOCK_K + BLOCK_K) & (offs_none_dummy := True) if False else a_ptrs and a_mask)
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            k_left = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=a_mask & (offs_k[None, :] < k_left), other=0.0)
            w = tl.load(w_ptrs, mask=n_mask & (k_mask[:, None] < k_left), other=0.0)
            accumulator = tl.dot(a, w.T, accumulator)
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K * N

        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=a_mask & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Flash-attention kernel for multi-query attention (n_kv_heads = 1).
#
# QKV buffer layout: [B, S, D + 2*HD]  (Q occupies cols 0..D, K cols D..D+HD,
# V cols D+HD..D+2HD).
#
# Each program computes, for one (batch, q-row-block):
#   - Q tile [BLOCK_M, D]   (all heads)
#   - full attention for all heads against the shared K, V [S, HD]
#   - output tile [BLOCK_M, D] stored to O buffer
# ---------------------------------------------------------------------------
@triton.jit
def flash_mqa_kernel(
    qkv_ptr, o_ptr,
    S,
    num_cores: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_MODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)

    NUM_Q_BLOCKS = tl.cdiv(S, BLOCK_M)
    NUM_BLOCKS = S  # placeholder, replaced below
    NUM_BLOCKS = (S // BLOCK_M + (1 if S % BLOCK_M != 0 else 0))
    NUM_B = tl.num_programs(0) // max(NUM_Q_BLOCKS, 1)
    NUM_B = NUM_B

    b = pid // NUM_Q_BLOCKS
    qk = pid % NUM_Q_BLOCKS

    offs_m = (qk * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
    m_mask = offs_m < S

    qkv_stride = D_MODEL + 2 * HEAD_DIM

    base = qkv_ptr + b.to(tl.int32) * S * qkv_stride + offs_m[:, None] * qkv_stride

    # ---- load full Q tile [BLOCK_M, D] ----
    acc0 = tl.zeros((BLOCK_M, D_MODEL), dtype=tl.float32)
    for d0 in range(0, D_MODEL, BLOCK_M):
        offs_d = (d0 + tl.arange(0, BLOCK_M)).to(tl.int32)
        q_tile = tl.load(base + offs_d[None, :],
                         mask=m_mask[:, None] & (offs_d[None, :] < D_MODEL),
                         other=0.0)
        acc0 = acc0
        q_part = tl.where(offs_d[None, :] < D_MODEL, q_tile.to(tl.float32), 0.0)
        # store q_part into the right position: we cannot index arbitrarily,
        # so we handle the common path with a single arange below instead.
        acc0 += 0.0

    # NOTE: the load above is generic but we need contiguous D offsets; the
    # implementation below does the Q gather per-head inside the KV loop to
    # keep tile widths constant.  Q tile is loaded once per d-chunk via
    # tl.dot would be a waste; plain loads suffice (Q is small).
    # We therefore re-load Q head slices directly in the attention loop from
    # the GM buffer using a strided arange (see below).

    offs_dn = tl.arange(0, HEAD_DIM).to(tl.int32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_all = tl.zeros((N_HEADS, BLOCK_M, HEAD_DIM), dtype=tl.float32)

    scale = (HEAD_DIM ** -0.5) * LOG2E

    for kv0 in range(0, S, BLOCK_N):
        offs_n = (kv0 + tl.arange(0, BLOCK_N)).to(tl.int32)
        n_mask = offs_n < S

        k_base = qkv_ptr + b.to(tl.int32) * S * qkv_stride \
            + offs_n[:, None] * qkv_stride + D_MODEL + offs_dn[None, :]
        v_base = k_base + HEAD_DIM

        k_tile = tl.load(k_base, mask=n_mask[:, None], other=0.0).to(tl.float32)
        v_tile = tl.load(v_base, mask=n_mask[:, None], other=0.0).to(tl.float32)

        for h in range(0, N_HEADS):
            q_row = base + h * HEAD_DIM + offs_dn[None, :]
            q_tile = tl.load(q_row, mask=m_mask[:, None], other=0.0).to(tl.float32)

            s = tl.dot(q_tile, k_tile.T)  # [BLOCK_M, BLOCK_N] fp32
            s = s * scale

            s = tl.where(n_mask[None, :], s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.math.exp2(s - m_new[:, None])

            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = tl.dot(p, v_tile.to(p.dtype), acc_all)
            m_i = m_new

    l_safe = l_i
    for h in range(0, N_HEADS):
        acc_h = tl.reshape(acc_all, (N_HEADS * BLOCK_M, HEAD_DIM))[h * BLOCK_M + tl.arange(0, BLOCK_M), :]
        out_h = acc_h * (1.0 / l_i)[:, None]
        o_ptrs = o_ptr + b.to(tl.int32) * S * D_MODEL \
            + offs_m[:, None] * D_MODEL + h * HEAD_DIM + offs_dn[None, :]
        tl.store(o_ptrs, out_h.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])

    l_safe_dummy = l_safe + 0.0


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

        if torch_npu is not None:
            try:
                _limit = torch_npu.npu.npu_config.get_device_limit(0)
                self.CUBE_CORE_NUM = _limit.get("cube_core_num", 24)
            except Exception:
                self.CUBE_CORE_NUM = 24
        else:
            self.CUBE_CORE_NUM = 24

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
            weights = (
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
            self._cache[key] = weights
        return self._cache[key]

    def forward(self, x, n_heads):
        torch.manual_seed(42)
        n_kv_heads = 1
        q_proj, k_proj, v_proj, out_proj = self._layers(
            x, n_heads, n_kv_heads
        )
        batch, sequence, d_model = x.shape
        head_dim = d_model // n_heads

        x2d = x.view(-1, d_model)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        M = batch * sequence

        # Build concatenated QKV weight [D + 2HD, D]
        w_qkv = torch.cat(
            [q_proj.weight, k_proj.weight, v_proj.weight], dim=0
        ).contiguous()
        N_qkv = d_model + 2 * head_dim
        w_out = out_proj.weight.contiguous()

        qkv = torch.empty(
            (batch, sequence, N_qkv), device=x.device, dtype=x.dtype
        )
        attn_out = torch.empty(
            (batch, sequence, d_model), device=x.device, dtype=x.dtype
        )

        num_cores = min(self.CUBE_CORE_NUM, 24)

        # ---- QKV GEMM ----
        BM = 64 if M > 16 else 16
        matmul_kernel[(num_cores,)](
            x2d, w_qkv, qkv,
            M, N_qkv, d_model,
            num_cores=num_cores,
            BLOCK_M=BM, BLOCK_N=64, BLOCK_K=64,
        )

        # ---- Flash attention ----
        flash_mqa_kernel[(num_cores,)](
            qkv, attn_out,
            sequence,
            num_cores=num_cores,
            N_HEADS=n_heads,
            HEAD_DIM=head_dim,
            D_MODEL=d_model,
            BLOCK_M=16,
            BLOCK_N=BLOCK_N,
        )

        # ---- output projection ----
        matmul_kernel[(num_cores,)](
            attn_out, w_out, out_out,
            M, d_model, d_model,
            num_cores=num_cores,
            BLOCK_M=BM, BLOCK_N=64, BLOCK_K=64,
        )

        return out_out.view(batch, sequence, d_model)
