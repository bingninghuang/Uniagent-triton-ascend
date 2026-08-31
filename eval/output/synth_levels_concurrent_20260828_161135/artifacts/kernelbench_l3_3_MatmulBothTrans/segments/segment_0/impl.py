import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_both_trans_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C = A.T @ B.T.

    A is stored (K, M) row-major, B is stored (N, K) row-major,
    C is stored (M, N) row-major.
    C[i, j] = sum_t A[t, i] * B[j, t]
    """
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_block_m * num_block_n

    for block_idx in range(pid, num_blocks, num_cores):
        block_m = block_idx // num_block_n
        block_n = block_idx - block_m * num_block_n

        offs_am = block_m * BLOCK_M + offs_m
        offs_bn = block_n * BLOCK_N + offs_n

        # A tile: A[t, i] -> offset t*M + i, tile shape (BLOCK_M, BLOCK_K) indexed [i, t]
        a_ptrs = A_ptr + offs_am[:, None] + offs_k[None, :] * M
        # B tile: B[j, t] -> offset j*K + t, tile shape (BLOCK_K, BLOCK_N) indexed [t, j]
        b_ptrs = B_ptr + offs_k[:, None] + offs_bn[None, :] * K

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K - k * BLOCK_K
            a_mask = k_mask[None, :] & (offs_am[:, None] < M)
            b_mask = k_mask[:, None] & (offs_bn[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * M
            b_ptrs += BLOCK_K

        c_ptrs = C_ptr + offs_am[:, None] * N + offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        K, M = A.shape
        N, _ = B.shape
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        itemsize = A.element_size()
        if itemsize == 2:
            if M >= 256 and K >= 128:
                BM, BN, BK = 256, 128, 64
            else:
                BM, BN, BK = 64, 64, 32
        else:
            if M >= 256 and K >= 128:
                BM, BN, BK = 128, 128, 32
            else:
                BM, BN, BK = 64, 64, 32

        num_cores = 24
        grid = (num_cores,)
        matmul_both_trans_kernel[grid](
            A, B, C, M, N, K, num_cores, BM, BN, BK
        )
        return C