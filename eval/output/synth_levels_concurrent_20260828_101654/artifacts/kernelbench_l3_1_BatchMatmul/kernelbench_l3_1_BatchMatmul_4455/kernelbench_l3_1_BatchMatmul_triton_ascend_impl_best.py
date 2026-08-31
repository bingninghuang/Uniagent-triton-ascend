"""Triton Ascend implementation of torch.bmm (batch matrix multiplication).

For A (B, M, K) and B (B, K, N) -> C (B, M, N), where C[i] = A[i] @ B[i].

Design:
- Single @triton.jit kernel; 2D tile per (M, N) output per batch entry.
- Grid = (min(total_tiles, CUBE_CORE_NUM),) with an interleaved tile loop so
  all CUBE cores are used and each program handles multiple tiles when the
  tile count exceeds the core count.
- Block sizes are constexpr powers of two (>=16 for tl.dot, <=128) chosen
  host-side per shape.
- Accumulation in fp32, cast to output dtype on store.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _batch_matmul_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    b,
    i_m,
    i_n,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    offs_m = (i_m * BM + tl.arange(0, BM)).to(tl.int32)
    offs_n = (i_n * BN + tl.arange(0, BN)).to(tl.int32)
    offs_k = tl.arange(0, BK).to(tl.int32)

    a_ptrs = a_ptr + (b * M * K).to(tl.int32) + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + (b * K * N).to(tl.int32) + offs_k[:, None] * N + offs_n[None, :]

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        mask_k = offs_k < K - k * BK
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        bt = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a, bt, acc)
        a_ptrs += BK
        b_ptrs += BK * N

    c_ptrs = c_ptr + (b * M * N).to(tl.int32) + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _batch_matmul_small_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    GRID_MN,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GRID_N: tl.constexpr,
):
    # Path A: one (b, i_m, i_n) tile per program, no partition loop.
    pid = tl.program_id(0).to(tl.int32)
    b = pid // GRID_MN
    i_mn = pid - b * GRID_MN
    i_m = i_mn // GRID_N
    i_n = i_mn - i_m * GRID_N

    _batch_matmul_tile(a_ptr, b_ptr, c_ptr, b, i_m, i_n, M, N, K, BM, BN, BK)


@triton.jit
def _batch_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    GRID_MN,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NUM_CORES: tl.constexpr,
    GRID_N: tl.constexpr,
):
    # Path B: interleaved tile partition when tiles > cores.
    pid = tl.program_id(0).to(tl.int32)
    total_tiles = B * GRID_MN

    for tile in range(pid, total_tiles, NUM_CORES):
        b = tile // GRID_MN
        i_mn = tile - b * GRID_MN
        i_m = i_mn // GRID_N
        i_n = i_mn - i_m * GRID_N

        _batch_matmul_tile(a_ptr, b_ptr, c_ptr, b, i_m, i_n, M, N, K, BM, BN, BK)


def _pow2_block(x, cap):
    """Smallest power of two >= x, clamped to [16, cap]. Host-side helper."""
    v = 16
    while v < x and v < cap:
        v = v << 1
    return v


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cube_core_num = 24
        try:
            import torch_npu
            limit = torch_npu.npu.npu_config.get_device_limit(0)
            self.cube_core_num = int(limit.get("cube_core_num", 24))
        except Exception:
            self.cube_core_num = 24

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        batch = A.shape[0]
        M = A.shape[1]
        K = A.shape[2]
        N = B.shape[2]
        C = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)
        if batch * M * N == 0:
            return C

        BM = _pow2_block(M, 64)
        BN = _pow2_block(N, 64)
        BK = _pow2_block(K, 256)

        grid_m = (M + BM - 1) // BM
        grid_n = (N + BN - 1) // BN
        grid_mn = grid_m * grid_n
        total_tiles = batch * grid_mn

        if total_tiles <= self.cube_core_num:
            # Path A: one tile per program, no partition loop.
            _batch_matmul_small_kernel[(total_tiles,)](
                A, B, C,
                batch, M, N, K, grid_mn,
                BM=BM, BN=BN, BK=BK,
                GRID_N=grid_n,
            )
        else:
            # Path B: interleaved partition over the CUBE cores.
            _batch_matmul_kernel[(self.cube_core_num,)](
                A, B, C,
                batch, M, N, K, grid_mn,
                BM=BM, BN=BN, BK=BK,
                NUM_CORES=self.cube_core_num, GRID_N=grid_n,
            )
        return C