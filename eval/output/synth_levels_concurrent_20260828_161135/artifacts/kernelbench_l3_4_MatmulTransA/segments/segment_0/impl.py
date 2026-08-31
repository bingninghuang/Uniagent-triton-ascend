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
    pid = tl.program_id(0).to(tl.int32)
    num_pid_m = tl.cdiv(N_A, BLOCK_M)
    num_pid_n = tl.cdiv(K_B, BLOCK_N)
    num_blocks = num_pid_m * num_pid_n

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for block_idx in range(pid, num_blocks, num_cores):
        pid_m = block_idx // num_pid_n
        pid_n = block_idx % num_pid_n

        offs_m_block = pid_m * BLOCK_M + offs_m
        offs_n_block = pid_n * BLOCK_N + offs_n
        mask_m = offs_m_block < N_A
        mask_n = offs_n_block < K_B

        # A tile: A_t[m, k] = A[k, m] -> offset k * stride_al + m
        a_ptrs = a_ptr + offs_m_block[:, None] + offs_k[None, :] * stride_al
        # B tile: B[k, n] -> offset k * stride_bl + n
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bl + offs_n_block[None, :])

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, M, BLOCK_K):
            k_mask = (k + offs_k) < M
            a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_al
            b_ptrs += BLOCK_K * stride_bl

        c_ptrs = c_ptr + offs_m_block[:, None] * K_B + offs_n_block[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


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
        grid_size = min(num_blocks, self.cube_core_num)

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
