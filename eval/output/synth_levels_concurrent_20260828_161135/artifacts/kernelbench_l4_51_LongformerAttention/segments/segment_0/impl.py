import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Hardware core counts (ascend910b1: 24 AI cores, 2 VEC + 1 CUBE per core)
# ---------------------------------------------------------------------------
def _get_vector_cores():
    try:
        import torch_npu  # noqa: F401

        return int(torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 48))
    except Exception:
        return 48


def _get_cube_cores():
    v = _get_vector_cores()
    return max(1, v // 2)


VECT_CORE_NUM = _get_vector_cores()
CUBE_CORE_NUM = _get_cube_cores()


# ---------------------------------------------------------------------------
# Kernel 1: scores = (Q @ K^T) / sqrt(d_k); invalid (masked) -> -1e9
#   q, k : [BH, S, DK] contiguous        (BH = B*H, flattened head index)
#   s    : [BH, S, S]  (i = row/query pos, j = col/key pos)
# valid(i, j) = (i-half <= j <= i+half) or i in G or j in G
# ---------------------------------------------------------------------------
@triton.jit
def _longformer_qk_kernel(
    q_ptr, k_ptr, s_ptr,
    S, DK, BH,
    scale_div, half_w, g0, g1,
    num_cores: tl.constexpr,
    BM: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    nbi = tl.cdiv(S, BM)
    nbj = tl.cdiv(S, BC)
    total = BH * nbi * nbj

    for blk in range(pid, total, num_cores):
        j_blk = blk % nbj
        t = blk // nbj
        i_blk = t % nbi
        bh = t // nbi

        offs_i = i_blk * BM + tl.arange(0, BM)
        offs_j = j_blk * BC + tl.arange(0, BC)

        base = bh * S * DK
        a_mask_row = offs_i[:, None] < S
        b_mask_row = offs_j[:, None] < S

        acc = tl.zeros((BM, BC), dtype=tl.float32)
        for k0 in range(0, DK, BK):
            offs_k = k0 + tl.arange(0, BK)
            a_mask = a_mask_row & (offs_k[None, :] < DK)
            b_mask = b_mask_row & (offs_k[None, :] < DK)
            a = tl.load(q_ptr + base + offs_i[:, None] * DK + offs_k[None, :],
                        mask=a_mask, other=0.0)
            b = tl.load(k_ptr + base + offs_j[:, None] * DK + offs_k[None, :],
                        mask=b_mask, other=0.0)
            acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)

        acc = acc / scale_div

        valid = ((offs_j[None, :] >= offs_i[:, None] - half_w) &
                 (offs_j[None, :] <= offs_i[:, None] + half_w)) | \
                ((offs_i[:, None] == g0) | (offs_i[:, None] == g1)) | \
                ((offs_j[None, :] == g0) | (offs_j[None, :] == g1))
        acc = tl.where(valid, acc, -1000000000.0)

        out_mask = a_mask_row & (offs_j[None, :] < S)
        tl.store(s_ptr + bh * S * S + offs_i[:, None] * S + offs_j[None, :],
                 acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Kernel 2: in-place row softmax over the S columns of scores [BH*S, S]
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_rows_kernel(
    sc_ptr,
    total_rows, S,
    num_cores: tl.constexpr,
    ROWS: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    n_blocks = tl.cdiv(total_rows, ROWS)

    for blk in range(pid, n_blocks, num_cores):
        row0 = blk * ROWS
        offs_r = (row0 + tl.arange(0, ROWS)).to(tl.int32)
        offs_c = tl.arange(0, BLOCK_S).to(tl.int32)

        m = (offs_r[:, None] < total_rows) & (offs_c[None, :] < S)
        x = tl.load(sc_ptr + offs_r[:, None] * S + offs_c[None, :],
                    mask=m, other=-1000000000.0)

        row_max = tl.max(x, axis=1)[:, None]
        exp_x = tl.exp(x - row_max)
        row_sum = tl.sum(exp_x, axis=1)[:, None]
        y = exp_x / row_sum

        tl.store(sc_ptr + offs_r[:, None] * S + offs_c[None, :],
                 y, mask=m)


# ---------------------------------------------------------------------------
# Kernel 3: out[b, s, h, d] = sum_j attn[bh, s, j] * v[bh, j, d]
#   attn : [BH, S, S]  (already softmaxed)
#   v    : [BH, S, DK]
#   out  : [B, S, H, DK] (transposed layout for the final projection)
# ---------------------------------------------------------------------------
@triton.jit
def _longformer_av_kernel(
    a_ptr, v_ptr, o_ptr,
    S, DK, H, HD, BH,
    num_cores: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    nbi = tl.cdiv(S, BM)
    nbd = tl.cdiv(DK, BN)
    total = BH * nbi * nbd

    for blk in range(pid, total, num_cores):
        d_blk = blk % nbd
        t = blk // nbd
        i_blk = t % nbi
        bh = t // nbi
        b = bh // H
        h = bh - b * H

        offs_i = i_blk * BM + tl.arange(0, BM)
        offs_d = d_blk * BN + tl.arange(0, BN)

        base_a = bh * S * S
        base_v = bh * S * DK

        a_row_mask = offs_i[:, None] < S
        d_col_mask = offs_d[None, :] < DK

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for j0 in range(0, S, BK):
            offs_j = j0 + tl.arange(0, BK)
            a = tl.load(a_ptr + base_a + offs_i[:, None] * S + offs_j[None, :],
                        mask=a_row_mask & (offs_j[None, :] < S), other=0.0)
            v = tl.load(v_ptr + base_v + offs_j[:, None] * DK + offs_d[None, :],
                        mask=(offs_j[:, None] < S) & d_col_mask, other=0.0)
            acc = tl.dot(a, v, acc, out_dtype=tl.float32)

        base_o = b * (S * HD) + h * DK
        tl.store(o_ptr + base_o + offs_i[:, None] * HD + offs_d[None, :],
                 acc, mask=a_row_mask & d_col_mask)


# ---------------------------------------------------------------------------
# Kernel 4: y[m, o] = sum_k x[m, k] * W[o, k] + bias[o]    (x @ W^T + b)
# ---------------------------------------------------------------------------
@triton.jit
def _proj_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    nbm = tl.cdiv(M, BM)
    nbn = tl.cdiv(N, BN)
    total = nbm * nbn

    for blk in range(pid, total, num_cores):
        n_blk = blk % nbn
        m_blk = blk // nbn

        offs_m = (m_blk * BM + tl.arange(0, BM)).to(tl.int32)
        offs_n = (n_blk * BN + tl.arange(0, BN)).to(tl.int32)

        m_mask = offs_m[:, None] < M
        n_mask = offs_n[:, None] < N

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                        mask=m_mask, other=0.0)
            w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                        mask=n_mask, other=0.0)
            acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)

        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

        tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=m_mask & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    """Triton-Ascend implementation of the Longformer attention operator.

    Mirrors the reference Model (same stateful attribute updates, same W_o
    creation so the random projection weights match exactly), but performs
    all numerical computation in Triton kernels:
      K1: scores = (Q K^T) / sqrt(d_k) with the longformer band+global mask
      K2: in-place row softmax
      K3: attn @ V, written in the transposed [B, S, H*DK] layout
      K4: output projection  out @ W_o.weight^T + W_o.bias
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.d_model = 512
        self.n_heads = 8
        self.window_size = 32
        self.global_attention_indices = [0, 511]
        self.d_k = self.d_model // self.n_heads
        self.dropout = nn.Dropout(p=0.0)
        self._cache = {}

    def forward(self, query, key, value):
        batch_size, n_heads, seq_len, d_k = query.shape
        if batch_size == 0 or n_heads == 0 or seq_len == 0 or d_k == 0:
            return value.transpose(1, 2).contiguous().view(
                batch_size, seq_len, n_heads * d_k)

        torch.manual_seed(42)

        # Infer model dimensions from input instead of hardcoding
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_model = n_heads * d_k
        self.window_size = min(self.window_size, seq_len)
        self.global_attention_indices = [
            min(idx, seq_len - 1)
            for idx in self.global_attention_indices if seq_len > 0
        ]

        # Sanitize inputs to avoid NaN/Inf propagating through matmul
        query = torch.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
        key = torch.nan_to_num(key, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        key_cache = (self.d_model, query.device, query.dtype)
        if key_cache not in self._cache:
            self._cache[key_cache] = nn.Linear(self.d_model, self.d_model).to(
                device=query.device, dtype=query.dtype)
        W_o = self._cache[key_cache]

        B = batch_size
        H = n_heads
        S = seq_len
        DK = d_k
        BH = B * H
        DM = self.d_model  # H * DK

        q2d = query.view(BH, S, DK)
        k2d = key.view(BH, S, DK)
        v2d = value.view(BH, S, DK)

        scores = torch.empty((BH, S, S), device=query.device, dtype=query.dtype)
        attn_out = torch.empty((B, S, H, DK), device=query.device, dtype=query.dtype)
        final = torch.empty((B, S, DM), device=query.device, dtype=query.dtype)

        half_w = self.window_size // 2
        g0 = self.global_attention_indices[0]
        g1 = self.global_attention_indices[1]

        # ---- K1: QK^T + scale + mask -------------------------------------
        BM1 = 64
        BC1 = 64
        BK1 = min(64, triton.next_power_of_2(DK))
        blk_total1 = BH * triton.cdiv(S, BM1) * triton.cdiv(S, BC1)
        grid1 = min(blk_total1, CUBE_CORE_NUM)
        _longformer_qk_kernel[(grid1,)](
            q2d, k2d, scores,
            S, DK, BH,
            float(math.sqrt(DK)), half_w, g0, g1,
            num_cores=grid1, BM=BM1, BC=BC1, BK=BK1,
        )

        # ---- K2: in-place row softmax ------------------------------------
        total_rows = BH * S
        ROWS = 4
        BLOCK_S = triton.next_power_of_2(S)
        blk_total2 = triton.cdiv(total_rows, ROWS)
        grid2 = min(blk_total2, VECT_CORE_NUM)
        _softmax_rows_kernel[(grid2,)](
            scores,
            total_rows, S,
            num_cores=grid2, ROWS=ROWS, BLOCK_S=BLOCK_S,
        )

        # ---- K3: attn @ V -> transposed layout ---------------------------
        BM3 = 64
        BN3 = min(64, triton.next_power_of_2(DK))
        BK3 = min(64, triton.next_power_of_2(S))
        blk_total3 = BH * triton.cdiv(S, BM3) * triton.cdiv(DK, BN3)
        grid3 = min(blk_total3, CUBE_CORE_NUM)
        _longformer_av_kernel[(grid3,)](
            scores, v2d, attn_out,
            S, DK, H, DM, BH,
            num_cores=grid3, BM=BM3, BN=BN3, BK=BK3,
        )

        # ---- K4: output projection (W_o) ---------------------------------
        M = B * S
        BM4 = 64
        BN4 = 64
        BK4 = 64
        blk_total4 = triton.cdiv(M, BM4) * triton.cdiv(DM, BN4)
        grid4 = min(blk_total4, CUBE_CORE_NUM)
        _proj_gemm_kernel[(grid4,)](
            attn_out, W_o.weight, W_o.bias, final,
            M, DM, DM,
            num_cores=grid4, BM=BM4, BN=BN4, BK=BK4,
        )

        return final
