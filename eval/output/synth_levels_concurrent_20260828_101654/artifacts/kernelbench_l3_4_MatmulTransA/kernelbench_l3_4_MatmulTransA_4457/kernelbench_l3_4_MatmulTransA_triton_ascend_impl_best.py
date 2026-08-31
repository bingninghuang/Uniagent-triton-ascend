import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transa_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C = A.T @ B
    # A: (K, M) row-major, B: (K, N) row-major, C: (M, N)
    # C[i, j] = sum_k A[k, i] * B[k, j]
    pid = tl.program_id(0)

    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n

    for block_idx in range(pid, num_blocks, num_cores):
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n

        i0 = block_m * BLOCK_M
        j0 = block_n * BLOCK_N

        a_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(K, M),
            strides=(M, 1),
            offsets=(0, i0),
            block_shape=(BLOCK_K, BLOCK_M),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(K, N),
            strides=(N, 1),
            offsets=(0, j0),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            # A tile: (BLOCK_K, BLOCK_M), A[kk, i]
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            a = tl.trans(a)  # (BLOCK_M, BLOCK_K) tile of A^T
            # B tile: (BLOCK_K, BLOCK_N), B[kk, j]
            b = tl.load(b_block_ptr, boundary_check=(0, 1))
            acc = tl.dot(a, b, acc)
            a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        c_block_ptr = tl.make_block_ptr(
            base=C_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(i0, j0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(c_block_ptr, acc.to(C_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def matmul_transa_small_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C = A.T @ B
    # A: (K, M) row-major, B: (K, N) row-major, C: (M, N)
    # Small-grid path: grid = num_blocks, one output block per program, no core loop.
    pid = tl.program_id(0)

    num_blocks_n = tl.cdiv(N, BLOCK_N)
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n

    i0 = block_m * BLOCK_M
    j0 = block_n * BLOCK_N

    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(K, M),
        strides=(M, 1),
        offsets=(0, i0),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, j0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        # A tile: (BLOCK_K, BLOCK_M), A[kk, i]
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        a = tl.trans(a)  # (BLOCK_M, BLOCK_K) tile of A^T
        # B tile: (BLOCK_K, BLOCK_N), B[kk, j]
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc)
        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(i0, j0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(C_ptr.dtype.element_ty), boundary_check=(0, 1))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch_npu
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 24)
        except Exception:
            self.CUBE_CORE_NUM = 24

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        K, M = A.shape
        K2, N = B.shape

        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        num_cores = self.CUBE_CORE_NUM

        LARGE_M, LARGE_N, LARGE_K = 128, 128, 32
        num_blocks = triton.cdiv(M, LARGE_M) * triton.cdiv(N, LARGE_N)

        if num_blocks <= num_cores:
            # Small-grid path: one output block per core, no per-core block loop.
            SMALL_M, SMALL_N, SMALL_K = 64, 64, 32
            small_blocks = triton.cdiv(M, SMALL_M) * triton.cdiv(N, SMALL_N)
            matmul_transa_small_kernel[(small_blocks,)](
                A, B, C,
                M, N, K,
                SMALL_M, SMALL_N, SMALL_K,
            )
        else:
            # Large-grid path: fixed CUBE-core grid, each core loops over blocks.
            matmul_transa_kernel[(num_cores,)](
                A, B, C,
                M, N, K,
                num_cores,
                LARGE_M, LARGE_N, LARGE_K,
            )
        return C