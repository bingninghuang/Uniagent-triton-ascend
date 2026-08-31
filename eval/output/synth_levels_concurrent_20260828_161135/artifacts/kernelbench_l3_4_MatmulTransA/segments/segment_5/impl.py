import torch
import torch.nn as nn
import triton
import triton.language as tl


def _matmul_configs():
    # Hand-written config set (custom autotune). All dims >= 16 (tl.dot min),
    # power of two, BLOCK_K <= 64 for fp32 (UB / L0B tile limits).
    combos = [
        (128, 128, 64),
        (64, 128, 64),
        (128, 64, 64),
        (128, 128, 32),
        (64, 64, 64),
        (128, 64, 32),
        (32, 64, 64),
        (64, 32, 64),
        (64, 64, 32),
        (16, 64, 64),
        (64, 16, 64),
        (32, 16, 32),
        (16, 16, 16),
        (16, 32, 32),
    ]
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
            for bm, bn, bk in combos]


@triton.autotune(configs=_matmul_configs(), key=["M", "N_A", "K_B"])
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
    # Each core handles a contiguous range of output blocks (contiguous task
    # partitioning); the scalar if-guard drops the extra iterations when the
    # block count is not a multiple of the core count.
    pid = tl.program_id(0).to(tl.int32)
    num_pid_m = (N_A + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (K_B + BLOCK_N - 1) // BLOCK_N
    num_blocks = num_pid_m * num_pid_n
    blocks_per_core = (num_blocks + num_cores - 1) // num_cores

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Float32 forms of the sizes for lane-wise boundary comparisons
    # (fp32 comparisons enable vectorization, per coding rules).
    N_A_f = N_A.to(tl.float32)
    K_B_f = K_B.to(tl.float32)
    M_f = M.to(tl.float32)

    for j in range(0, blocks_per_core):
        block_idx = pid * blocks_per_core + j
        if block_idx < num_blocks:
            pid_m = block_idx // num_pid_n
            pid_n = block_idx - pid_m * num_pid_n

            offs_m_block = pid_m * BLOCK_M + offs_m
            offs_n_block = pid_n * BLOCK_N + offs_n
            mask_m = offs_m_block.to(tl.float32) < N_A_f
            mask_n = offs_n_block.to(tl.float32) < K_B_f

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, M, BLOCK_K):
                k_off = k + offs_k
                k_mask = k_off.to(tl.float32) < M_f
                a = tl.load(a_ptr + k_off[:, None] * stride_al + offs_m_block[None, :],
                            mask=k_mask[:, None] & mask_m[None, :], other=0.0)
                b = tl.load(b_ptr + k_off[:, None] * stride_bl + offs_n_block[None, :],
                            mask=k_mask[:, None] & mask_n[None, :], other=0.0)
                acc = tl.dot(tl.trans(a), b, acc, out_dtype=tl.float32)

            c_ptrs = c_ptr + offs_m_block[:, None] * K_B + offs_n_block[None, :]
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
                     mask=mask_m[:, None] & mask_n[None, :])


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

        # Shape-based grid: at most one core per 128x128 output tile, capped
        # at the cube core count. Every autotuned config uses tiles <= 128, so
        # grid <= num_blocks for any config; each core then loops over its
        # contiguous chunk of blocks (if-guarded for the remainder).
        nb128 = ((N_A + 127) // 128) * ((K_B + 127) // 128)
        grid_size = nb128 if nb128 < self.cube_core_num else self.cube_core_num

        matmul_transa_kernel[(grid_size,)](
            A, B, C,
            M, N_A, K_B,
            A.stride(0), B.stride(0),
            num_cores=grid_size,
        )
        return C
