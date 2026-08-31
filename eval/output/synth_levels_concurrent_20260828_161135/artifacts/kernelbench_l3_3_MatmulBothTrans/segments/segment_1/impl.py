import torch
import torch.nn as nn
import triton
import triton.language as tl

# Dynamically read the physical core count (no hardcoding).
# This is a mix kernel (tl.dot + vector ops): grid must not exceed the
# cube core count, take min(vector cores, cube cores).
try:
    import torch_npu
    _DEV_LIMIT = torch_npu.npu.npu_config.get_device_limit(0)
    _NUM_VEC = _DEV_LIMIT.get('vector_core_num', 24)
    _NUM_CUBE = _DEV_LIMIT.get('cube_core_num',
                               _DEV_LIMIT.get('ai_core_num', _NUM_VEC))
    NUM_AI_CORES = min(_NUM_VEC, _NUM_CUBE)
except Exception:
    NUM_AI_CORES = 24


@triton.jit
def matmul_both_trans_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    grid_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C = A.T @ B.T.

    A is stored (K, M) row-major, B is stored (N, K) row-major,
    C is stored (M, N) row-major.
    C[i, j] = sum_t A[t, i] * B[j, t]

    Contiguous (non-interleaved) partitioning: program `pid` owns a
    contiguous, load-balanced range of output-tile blocks.
    """
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_block_m * num_block_n

    # Balanced contiguous split of output tiles across programs.
    bpc = num_blocks // grid_size
    rem = num_blocks - bpc * grid_size
    start = pid * bpc + tl.minimum(pid, rem)
    extra = tl.minimum(tl.maximum(rem - pid, 0), 1)

    n_k_blocks = tl.cdiv(K, BLOCK_K)

    for block_idx in range(start, start + bpc + extra):
        block_m = block_idx // num_block_n
        block_n = block_idx - block_m * num_block_n

        offs_am = block_m * BLOCK_M + offs_m
        offs_bn = block_n * BLOCK_N + offs_n
        am_valid = offs_am < M
        bn_valid = offs_bn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, n_k_blocks):
            koff = k * BLOCK_K
            k_mask = offs_k < K - koff
            # A tile: A[t, i] -> offset t*M + i, tile (BLOCK_M, BLOCK_K) as [i, t]
            a_ptrs = A_ptr + offs_am[:, None] + (koff + offs_k)[None, :] * M
            # B tile: B[j, t] -> offset j*K + t, tile (BLOCK_K, BLOCK_N) as [t, j]
            b_ptrs = B_ptr + (koff + offs_k)[:, None] + offs_bn[None, :] * K
            a = tl.load(a_ptrs,
                        mask=k_mask[None, :] & am_valid[:, None],
                        other=0.0)
            b = tl.load(b_ptrs,
                        mask=k_mask[:, None] & bn_valid[None, :],
                        other=0.0)
            acc = tl.dot(a, b, acc)

        c_ptrs = C_ptr + offs_am[:, None] * N + offs_bn[None, :]
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty),
                 mask=am_valid[:, None] & bn_valid[None, :])


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

        num_blocks = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        grid_size = min(num_blocks, NUM_AI_CORES)
        matmul_both_trans_kernel[(grid_size,)](
            A, B, C, M, N, K, grid_size, BM, BN, BK
        )
        return C