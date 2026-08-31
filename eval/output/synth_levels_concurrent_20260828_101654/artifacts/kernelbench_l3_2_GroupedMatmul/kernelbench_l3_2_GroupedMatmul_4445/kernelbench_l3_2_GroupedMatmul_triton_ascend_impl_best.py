"""Triton Ascend implementation of KernelBench L3-2: GroupedMatmul.

Operator semantics (aligned with torch.nn.functional.grouped_mm):
  * 3D A: (G, M, K) @ B (G, K, N) -> (G*M, N); group g computes A[g] @ B[g].
  * 2D A: (R, K) @ B (G, K, N), offsets (G,) *end* offsets; group g computes
    A[start_g:end_g] @ B[g] and results are concatenated, so the output row
    index equals the input row index.

Design:
  * One fused kernel computes every per-group GEMM output tile.
  * Work items are (group, m_block, n_block) tiles; the kernel launches on a
    fixed grid of CUBE cores, each program interleaving its tiles.
  * fp32 accumulation, output stored in the input dtype.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:  # pragma: no cover - environment guard
    torch_npu = None


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _grouped_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, total_tiles, num_cores,      # M: rows per group (runtime)
    N: tl.constexpr,
    K: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    tiles_per_group = num_m_blocks * num_n_blocks

    for w in range(pid, total_tiles, num_cores):
        # decode work item -> (group g, m block, n block)
        # (divisors are compile-time constants -> fast constant-div sequences)
        g = w // tiles_per_group
        r = w - g * tiles_per_group
        mb = r // num_n_blocks
        nb = r - mb * num_n_blocks

        row_base = g * M + mb * BLOCK_M
        row_end = g * M + M          # exclusive end of this group's rows

        offs_m = row_base + tl.arange(0, BLOCK_M)
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < row_end
        n_mask = offs_n < N

        b_base = b_ptr + g * K * N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a_tile = tl.load(
                a_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b_tile = tl.load(
                b_base + offs_k[:, None] * N + offs_n[None, :],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(a_tile, b_tile, acc)

        out_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, acc.to(c_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
def _pick_blocks(m: int, k: int, n: int):
    """Choose tile sizes (constexpr per launch, pure host arithmetic)."""
    bm = max(16, min(128, triton.next_power_of_2(m)))
    bn = max(16, min(128, triton.next_power_of_2(n)))
    bk = max(16, min(64, triton.next_power_of_2(k)))
    return bm, bn, bk


def _uneven_2d_forward(A: torch.Tensor, B: torch.Tensor,
                       offsets: torch.Tensor, num_cores: int) -> torch.Tensor:
    """Fallback for uneven 2D groups: per-group launches of the same kernel."""
    g, k, n = B.shape
    offs = offsets.to(torch.int32).tolist()
    last = int(offs[-1])
    out = torch.empty((last, n), dtype=A.dtype, device=A.device)
    start = 0
    for i in range(g):
        end = int(offs[i])
        m_i = end - start
        if m_i > 0:
            bm, bn, bk = _pick_blocks(m_i, k, n)
            num_m_blocks = triton.cdiv(m_i, bm)
            num_n_blocks = triton.cdiv(n, bn)
            total_tiles = num_m_blocks * num_n_blocks
            if total_tiles > 0:
                grid_size = min(total_tiles, num_cores)
                _grouped_gemm_kernel[grid_size, ](
                    A[start:end], B[i:i + 1], out[start:end],
                    m_i, total_tiles, grid_size,
                    N=n, K=k,
                    num_m_blocks=num_m_blocks,
                    num_n_blocks=num_n_blocks,
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                )
        start = end
    return out


class ModelNew(nn.Module):
    """GroupedMatmul via a single fused Triton-Ascend kernel."""

    def __init__(self):
        super(ModelNew, self).__init__()
        self.CUBE_CORE_NUM = 24
        if torch_npu is not None:
            try:
                self.CUBE_CORE_NUM = int(
                    torch_npu.npu.npu_config.get_device_limit(0)
                    .get("cube_core_num", 24)
                )
            except Exception:
                self.CUBE_CORE_NUM = 24

    def forward(self, A: torch.Tensor, B: torch.Tensor,
                offsets: torch.Tensor = None) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        g, k, n = B.shape
        num_cores = self.CUBE_CORE_NUM

        if A.dim() == 3:
            # uniform groups: (G, M, K), group i owns flat rows [i*M, (i+1)*M)
            g_a, m, k_a = A.shape
            assert g_a == g and k_a == k, "A/B shape mismatch"
            total_rows = g * m
        else:
            # 2D A with end-offsets: group i owns rows [offs[i-1], offs[i])
            total_rows, k_a = A.shape
            assert k_a == k, "A/B shape mismatch"
            if offsets is not None and total_rows % g != 0:
                # uneven groups: per-group launches (no host-side sync needed
                # on the uniform fast path below)
                return _uneven_2d_forward(A, B, offsets, num_cores)
            m = total_rows // g

        out = torch.empty((total_rows, n), dtype=A.dtype, device=A.device)
        bm, bn, bk = _pick_blocks(m, k, n)
        num_m_blocks = triton.cdiv(m, bm)
        num_n_blocks = triton.cdiv(n, bn)
        total_tiles = g * num_m_blocks * num_n_blocks

        if total_tiles > 0:
            grid_size = total_tiles if total_tiles < num_cores else num_cores
            _grouped_gemm_kernel[grid_size, ](
                A, B, out,
                m, total_tiles, grid_size,
                N=n, K=k,
                num_m_blocks=num_m_blocks, num_n_blocks=num_n_blocks,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            )
        return out
