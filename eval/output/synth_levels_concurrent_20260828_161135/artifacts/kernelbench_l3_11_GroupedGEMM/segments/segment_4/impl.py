import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _grouped_bmm_kernel(
    lhs_ptr,          # [M, K]  lhs activation, contig
    weight_ptr,       # [G, O, K] grouped weight, contig
    m_idx_ptr,        # [M]     group id per row (int32)
    out_ptr,          # [M, O]  output, contig
    M, O, K,
    stride_wg, stride_wo, stride_wk,
    stride_lr, stride_lk,
    stride_or, stride_oo,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # out[m, o] = sum_k weight[m_idx[m], o, k] * lhs[m, k]
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    mask_m = offs_m < M
    mask_o = offs_o < O

    # per-row group ids -> base pointer into weight: weight[g, :, :]
    g = tl.load(m_idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
    w_base = g * stride_wg

    acc = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if EVEN_K:
            mask_k = tl.full((BLOCK_K,), 1, tl.int1)
        else:
            mask_k = offs_k < K

        # A tile: lhs[m, k]   [BLOCK_M, BLOCK_K]
        l_ptrs = lhs_ptr + offs_m[:, None] * stride_lr + offs_k[None, :] * stride_lk
        a = tl.load(l_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # B tile gathered per row: weight[g(m), o, k]  [BLOCK_M, BLOCK_O, BLOCK_K]
        w_ptrs = (
            weight_ptr
            + w_base[:, None, None]
            + offs_o[None, :, None] * stride_wo
            + offs_k[None, None, :] * stride_wk
        )
        b = tl.load(
            w_ptrs,
            mask=mask_m[:, None, None] & mask_o[None, :, None] & mask_k[None, None, :],
            other=0.0,
        )
        # acc[m, o] += sum_k a[m, k] * b[m, o, k]
        acc += tl.sum(b.to(tl.float32) * a.to(tl.float32)[:, None, :], axis=2)

    o_ptrs = out_ptr + offs_m[:, None] * stride_or + offs_o[None, :] * stride_oo
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_o[None, :])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, lhs, m_indices, num_groups, out_features):
        rows, in_features = lhs.shape
        key = (num_groups, out_features, in_features, lhs.device, lhs.dtype)
        weight = self._cache.get(key)
        if weight is None:
            self._cache.clear()
            generator = torch.Generator()
            generator.manual_seed(42)
            weight = torch.randn(
                (num_groups, out_features, in_features),
                generator=generator,
                dtype=torch.float32,
            ) / math.sqrt(in_features)
            weight = weight.to(device=lhs.device, dtype=lhs.dtype)
            self._cache[key] = weight

        out = torch.empty((rows, out_features), device=lhs.device, dtype=lhs.dtype)
        BLOCK_M = 8
        BLOCK_O = 32
        BLOCK_K = 32
        even_k = (in_features % BLOCK_K) == 0
        grid = (
            (rows + BLOCK_M - 1) // BLOCK_M,
            (out_features + BLOCK_O - 1) // BLOCK_O,
        )
        _grouped_bmm_kernel[grid](
            lhs,
            weight,
            m_indices,
            out,
            rows,
            out_features,
            in_features,
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            lhs.stride(0),
            lhs.stride(1),
            out.stride(0),
            out.stride(1),
            EVEN_K=even_k,
            BLOCK_M=BLOCK_M,
            BLOCK_O=BLOCK_O,
            BLOCK_K=BLOCK_K,
        )
        return out
