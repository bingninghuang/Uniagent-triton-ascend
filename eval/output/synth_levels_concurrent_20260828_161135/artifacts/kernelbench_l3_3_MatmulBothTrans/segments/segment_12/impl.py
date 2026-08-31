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
    grid_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C = A.T @ B.T.

    A is stored (K, M) row-major, B is stored (N, K) row-major,
    C is stored (M, N) row-major.
    C[i, j] = sum_t A[t, i] * B[j, t]

    Tiles are loaded so the memory-contiguous dim is the tile's LAST
    dimension (row-major tile loads with 512B row width), which the
    linalg->cube lowering can vectorize:
      aT (BLOCK_K, BLOCK_M): aT[k, m] = A[k, m]      (contiguous along m)
      bT (BLOCK_N, BLOCK_K): bT[n, k] = B[n, k]      (contiguous along k)
    The inner loop is a standard GEMM for C^T = B^T... i.e.
    C^T[n, m] = sum_k bT[n, k] * aT[k, m] = dot(bT, aT),
    followed by one transpose of the output tile before storing.

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

        acc_T = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k in range(0, n_k_blocks):
            koff = k * BLOCK_K
            k_mask = offs_k < K - koff
            # aT tile (BLOCK_K, BLOCK_M): [kk, mm] -> offset (koff+kk)*M + offs_am
            aT_ptrs = A_ptr + (koff + offs_k)[:, None] * M + offs_am[None, :]
            # bT tile (BLOCK_N, BLOCK_K): [nn, kk] -> offset offs_bn*K + koff+kk
            bT_ptrs = B_ptr + offs_bn[:, None] * K + (koff + offs_k)[None, :]
            aT = tl.load(aT_ptrs,
                         mask=k_mask[:, None] & am_valid[None, :],
                         other=0.0)
            bT = tl.load(bT_ptrs,
                         mask=bn_valid[:, None] & k_mask[None, :],
                         other=0.0)
            acc_T = tl.dot(bT, aT, acc_T)

        acc = tl.trans(acc_T)
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
        # 512B row-width tiling (examples/triton-ascend-matmul guide, "A、B
        # 都转置" case): the A^T tile's contiguous dim (M) and the B^T tile's
        # contiguous dim (K) are each one 512B row: 256 elem for fp16/bf16
        # and 128 elem for fp32. N-tile follows the guide's N0=128.
        if itemsize == 2:
            BM, BN, BK = 256, 128, 256
        else:
            BM, BN, BK = 128, 128, 128

        num_blocks = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        # Launch one program per physical core at most; use one program per
        # output tile when there are fewer tiles than cores (empty programs
        # would otherwise still be scheduled).
        if num_blocks <= NUM_AI_CORES:
            grid_size = num_blocks
        else:
            grid_size = NUM_AI_CORES
        matmul_both_trans_kernel[(grid_size,)](
            A, B, C, M, N, K, grid_size, BM, BN, BK,
            multibuffer=True,
        )
        return C