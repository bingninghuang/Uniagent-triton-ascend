import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Let the runtime/compiler split the logical block grid across the physical
# AI cores automatically (auto-blockify), instead of capping the grid to the
# core count and looping manually per program.
os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

try:
    import torch_npu
except Exception:
    torch_npu = None


@triton.jit
def matmul_trans_a_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a0, stride_b0,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Compute C = A.T @ B where:
      A: (M, N) row-major, B: (M, K) row-major, C: (N, K)
    C[i, j] = sum_{l=0}^{M-1} A[l, i] * B[l, j]

    The left operand is A read in transposed fashion (contiguous along
    output-row i), the right operand is B read row-major (contiguous along
    output-column j) -> "A transposed, B not transposed" pattern.
    One program per logical output block; the runtime auto-blockifies the
    grid onto the physical AI cores (TRITON_ALL_BLOCKS_PARALLEL).
    """
    pid = tl.program_id(0)
    NUM_BLOCKS_N = tl.cdiv(K, BLOCK_N)
    block_m = pid // NUM_BLOCKS_N
    block_n = pid - block_m * NUM_BLOCKS_N

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output row i
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output col j

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    for k in range(0, M, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)  # reduction index l

        # a tile loaded as A's normal row-major tile (BLOCK_K, BLOCK_M):
        # a[l, i] = A[l, i], contiguous along i -> coalesced; then
        # transpose in-tile so dot sees lhs (BLOCK_M, BLOCK_K) = A.T
        a_off = offs_k[:, None] * stride_a0 + offs_m[None, :] * 1
        if EVEN_K and EVEN_N:
            a = tl.load(a_ptr + a_off)
        else:
            a_mask = (offs_k[:, None] < M) & (offs_m[None, :] < N)
            a = tl.load(a_ptr + a_off, mask=a_mask, other=0.0)
        a_tile = tl.trans(a)  # (BLOCK_M, BLOCK_K)

        # b_tile (BLOCK_K, BLOCK_N): b_tile[l, j] = B[l, j]
        b_off = offs_k[:, None] * stride_b0 + offs_n[None, :] * 1
        if EVEN_K and EVEN_M:
            b = tl.load(b_ptr + b_off)
        else:
            b_mask = (offs_k[:, None] < M) & (offs_n[None, :] < K)
            b = tl.load(b_ptr + b_off, mask=b_mask, other=0.0)

        acc = tl.dot(a_tile, b, acc, out_dtype=ACC_DTYPE)

    c_off = offs_m[:, None] * K + offs_n[None, :]
    if EVEN_N and EVEN_K:
        tl.store(c_ptr + c_off, acc.to(c_ptr.dtype.element_ty))
    else:
        c_mask = (offs_m[:, None] < N) & (offs_n[None, :] < K)
        tl.store(c_ptr + c_off, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def _launch_matmul_trans_a(A, B, C, cube_core_num):
    """Host-side launch helper: pick blocks and launch the kernel."""
    m, n = A.shape          # m = reduction dim, n = output rows
    k = B.shape[1]          # k = output cols

    # Power-of-two, mask-free blocks: pick blocks that exactly divide
    # each dim (dims are powers of two in this benchmark).
    p_n = 1 << (n.bit_length() - 1)
    p_k = 1 << (k.bit_length() - 1)
    p_m = 1 << (m.bit_length() - 1)
    block_m = min(128, max(16, p_n))
    block_n = min(128, max(16, p_k))
    if A.element_size() == 2:
        block_k = min(128, max(16, p_m))
    else:
        block_k = min(64, max(16, p_m))

    even_m = (m % block_k == 0)
    even_n = (n % block_m == 0)
    even_k = (k % block_n == 0)

    # Launch one program per logical output block; the runtime
    # auto-blockifies them onto the physical AI cores.
    num_blocks = triton.cdiv(n, block_m) * triton.cdiv(k, block_n)

    matmul_trans_a_kernel[(num_blocks,)](
        A, B, C,
        m, n, k,
        A.stride(0), B.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        ACC_DTYPE=tl.float32,
        EVEN_M=even_m,
        EVEN_N=even_n,
        EVEN_K=even_k,
    )


class ModelNew(nn.Module):
    """Triton Ascend implementation of torch.matmul(A.T, B)."""

    def __init__(self):
        super(ModelNew, self).__init__()
        self.CUBE_CORE_NUM = 24
        if torch_npu is not None:
            try:
                props = torch_npu.npu.npu_config.get_device_limit(0)
                core_num = props.get("cube_core_num", None)
                if core_num is None:
                    core_num = props.get("matrix_core_num", None)
                if core_num:
                    self.CUBE_CORE_NUM = int(core_num)
            except Exception:
                pass

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        n = A.shape[1]
        k = B.shape[1]
        C = torch.empty((n, k), device=A.device, dtype=A.dtype)
        _launch_matmul_trans_a(A, B, C, self.CUBE_CORE_NUM)
        return C
