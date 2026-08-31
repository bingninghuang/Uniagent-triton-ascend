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
    EVEN_K: tl.constexpr,
):
    """Compute C = A.T @ B.T.

    A is stored (K, M) row-major, B is stored (N, K) row-major,
    C is stored (M, N) row-major.
    C[i, j] = sum_t A[t, i] * B[j, t]

    Contiguous (non-interleaved) partitioning: program `pid` owns a
    contiguous, load-balanced range of output-tile blocks.

    Address offsets that are invariant over the K loop are precomputed
    per output tile; each K iteration only adds the scalar offset of
    the loop variable (no mutable accumulators inside the loop).
    """
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # fp32 offsets for vectorized boundary comparisons.
    offs_m_f = offs_m.to(tl.float32)
    offs_n_f = offs_n.to(tl.float32)
    offs_k_f = offs_k.to(tl.float32)
    M_f = M.to(tl.float32)
    N_f = N.to(tl.float32)
    K_f = K.to(tl.float32)

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
        # fp32 boundary comparisons (vectorized).
        am_valid = offs_am.to(tl.float32) < M_f
        bn_valid = offs_bn.to(tl.float32) < N_f

        # K-invariant parts of the address offsets, precomputed per tile:
        #   A tile element [i, t] is A_ptr + i + t*M
        #   B tile element [t, j] is B_ptr + j*K + t
        a_col_off = offs_k * M          # (BLOCK_K,)
        b_row_off = offs_bn * K         # (BLOCK_N,)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, n_k_blocks):
            koff = k * BLOCK_K
            if EVEN_K:
                # K is a multiple of BLOCK_K: no K-boundary mask needed.
                a = tl.load(A_ptr + offs_am[:, None] + a_col_off[None, :] + koff * M,
                            mask=am_valid[:, None], other=0.0)
                b = tl.load(B_ptr + b_row_off[None, :] + offs_k[:, None] + koff,
                            mask=bn_valid[None, :], other=0.0)
            else:
                k_mask = offs_k_f < K_f - koff
                a = tl.load(A_ptr + offs_am[:, None] + a_col_off[None, :] + koff * M,
                            mask=k_mask[None, :] & am_valid[:, None], other=0.0)
                b = tl.load(B_ptr + b_row_off[None, :] + offs_k[:, None] + koff,
                            mask=k_mask[:, None] & bn_valid[None, :], other=0.0)
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
        # Launch one program per physical core at most; use one program per
        # output tile when there are fewer tiles than cores (empty programs
        # would otherwise still be scheduled).
        if num_blocks <= NUM_AI_CORES:
            grid_size = num_blocks
        else:
            grid_size = NUM_AI_CORES
        even_k = (K % BK) == 0
        matmul_both_trans_kernel[(grid_size,)](
            A, B, C, M, N, K, grid_size, BM, BN, BK, even_k
        )
        return C