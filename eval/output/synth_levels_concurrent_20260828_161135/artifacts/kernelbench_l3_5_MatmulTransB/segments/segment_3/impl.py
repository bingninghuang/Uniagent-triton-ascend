import torch
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:  # torch_npu may not be importable in some environments
    torch_npu = None


@triton.jit
def matmul_transb_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Fixed-grid launch: grid = (num_cores,), each core strides through blocks
    pid = tl.program_id(0)
    NUM_BLOCKS_M = (M + BLOCK_M - 1) // BLOCK_M
    NUM_BLOCKS_N = (N + BLOCK_N - 1) // BLOCK_N
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        m_base = block_m * BLOCK_M
        n_base = block_n * BLOCK_N

        # Define row/col offsets OUTSIDE the k-loop so they are visible
        # after the loop for the store.
        offs_2d_m = m_base + offs_m          # [BLOCK_M]
        offs_2d_n = n_base + offs_n          # [BLOCK_N]

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_2d_k = k + offs_k               # [BLOCK_K]

            # A tile [BLOCK_M, BLOCK_K]: A is [M, K] row-major
            a_mask = (offs_2d_m[:, None] < M) & (offs_2d_k[None, :] < K)
            a = tl.load(
                a_ptr + offs_2d_m[:, None] * K + offs_2d_k[None, :],
                mask=a_mask, other=0.0,
            )

            # B tile [BLOCK_K, BLOCK_N]: B is [N, K] row-major, we need B^T[k, n] = B[n, k]
            b_mask = (offs_2d_k[:, None] < K) & (offs_2d_n[None, :] < N)
            b = tl.load(
                b_ptr + offs_2d_n[None, :] * K + offs_2d_k[:, None],
                mask=b_mask, other=0.0,
            )

            accumulator = tl.dot(a, b, accumulator)

        # C tile [BLOCK_M, BLOCK_N]: C = A @ B^T, C is [M, N] row-major
        c_mask = (offs_2d_m[:, None] < M) & (offs_2d_n[None, :] < N)
        c_val = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(
            c_ptr + offs_2d_m[:, None] * N + offs_2d_n[None, :],
            c_val, mask=c_mask,
        )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # CUBE core count for MatMul; read once at init, never in forward.
        self.CUBE_CORE_NUM = 24
        if torch_npu is not None:
            try:
                self.CUBE_CORE_NUM = int(
                    torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 24)
                )
            except Exception:
                self.CUBE_CORE_NUM = 24
        if self.CUBE_CORE_NUM <= 0:
            self.CUBE_CORE_NUM = 24

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape[0], A.shape[1]
        N = B.shape[0]
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 128
        grid_size = self.CUBE_CORE_NUM

        matmul_transb_kernel[(grid_size,)](
            A, B, C, M, N, K, grid_size, BLOCK_M, BLOCK_N, BLOCK_K
        )
        return C
