import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transa_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N_A, K_B,
    stride_al, stride_bl,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C (N_A x K_B) = A^T (N_A x M) @ B (M x K_B), A: (M, N_A), B: (M, K_B)
    # Both input tiles are loaded in natural layout (k rows, stride-1 cols) for
    # coalesced access; the A transpose is done in-register with tl.trans.
    pid = tl.program_id(0).to(tl.int32)
    num_pid_m = (N_A + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (K_B + BLOCK_N - 1) // BLOCK_N
    num_blocks = num_pid_m * num_pid_n
    blocks_per_core = (num_blocks + num_cores - 1) // num_cores

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for j in range(0, blocks_per_core):
        block_idx = pid * blocks_per_core + j
        pid_m = block_idx // num_pid_n
        pid_n = block_idx - pid_m * num_pid_n

        offs_m_block = pid_m * BLOCK_M + offs_m
        offs_n_block = pid_n * BLOCK_N + offs_n
        mask_m = offs_m_block < N_A
        mask_n = offs_n_block < K_B

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, M, BLOCK_K):
            k_off = k + offs_k
            k_mask = k_off < M
            a = tl.load(a_ptr + k_off[:, None] * stride_al + offs_m_block[None, :],
                        mask=k_mask[:, None] & mask_m[None, :], other=0.0)
            b = tl.load(b_ptr + k_off[:, None] * stride_bl + offs_n_block[None, :],
                        mask=k_mask[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(tl.trans(a), b, acc, out_dtype=tl.float32)

        block_valid = block_idx < num_blocks
        c_ptrs = c_ptr + offs_m_block[:, None] * K_B + offs_n_block[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
                 mask=block_valid & (mask_m[:, None] & mask_n[None, :]))


def _pow2_block(n: int, lo: int, hi: int) -> int:
    p = 16
    while p < n:
        p <<= 1
    if p < lo:
        p = lo
    if p > hi:
        p = hi
    return p


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        try:
            import torch_npu
            self.cube_core_num = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 24)
        except Exception:
            self.cube_core_num = 24

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        M, N_A = A.shape
        K_B = B.shape[1]

        C = torch.empty((N_A, K_B), device=A.device, dtype=A.dtype)

        block_k_max = 64 if A.dtype == torch.float32 else 128
        BLOCK_M = _pow2_block(N_A, 16, 128)
        BLOCK_N = _pow2_block(K_B, 16, 128)
        BLOCK_K = _pow2_block(M, 16, block_k_max)

        num_blocks = triton.cdiv(N_A, BLOCK_M) * triton.cdiv(K_B, BLOCK_N)
        grid_size = num_blocks if num_blocks < self.cube_core_num else self.cube_core_num

        matmul_transa_kernel[(grid_size,)](
            A, B, C,
            M, N_A, K_B,
            A.stride(0), B.stride(0),
            num_cores=grid_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return C
