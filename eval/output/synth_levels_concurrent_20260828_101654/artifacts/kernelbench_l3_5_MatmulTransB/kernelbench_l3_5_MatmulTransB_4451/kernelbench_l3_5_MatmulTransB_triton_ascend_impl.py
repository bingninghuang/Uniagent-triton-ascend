import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _matmul_transb_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    NUM_PROGS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # A : (M, K)  row-major contiguous   -> out[i, j] = sum_r A[i, r] * B[j, r]
    # B : (N, K)  row-major contiguous   -> B.T is the (K, N) right-hand side
    # C : (M, N)  row-major contiguous
    # All inputs are made contiguous on the host, so row strides equal K / K / N.
    pid = tl.program_id(0).to(tl.int32)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_blocks = num_pid_m * num_pid_n

    # contiguous partitioning: balanced contiguous block range per program
    blocks_per_core = total_blocks // NUM_PROGS
    remainder = total_blocks - blocks_per_core * NUM_PROGS
    start_block = pid * (blocks_per_core + 1)
    my_blocks = blocks_per_core + 1
    is_tail = pid >= remainder
    start_block = tl.where(is_tail, remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core, start_block)
    my_blocks = tl.where(is_tail, blocks_per_core, my_blocks)

    for i in range(my_blocks):
        block_idx = start_block + i
        pid_m = block_idx // num_pid_n
        pid_n = block_idx - pid_m * num_pid_n

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

        a_base = a_ptr + offs_m[:, None] * K
        # load a contiguous (BLOCK_N, BLOCK_K) sub-tile of B, then transpose
        b_base = b_ptr + offs_n[:, None] * K
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]

        m_mask = offs_m[:, None] < M
        n_mask = offs_n[:, None] < N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
            k_off = k_iter * BLOCK_K
            k_mask = (k_off + offs_k)[None, :] < K
            a = tl.load(a_base + k_off + offs_k[None, :], mask=m_mask & k_mask, other=0.0)
            b = tl.load(b_base + k_off + offs_k[None, :], mask=n_mask & k_mask, other=0.0)
            accumulator = tl.dot(a, tl.trans(b), accumulator)

        c = accumulator.to(c_ptr.dtype.element_ty)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def _pow2_clamp(x, lo, hi):
    if x < 1:
        x = 1
    v = triton.next_power_of_2(x)
    if v < lo:
        v = lo
    elif v > hi:
        v = hi
    return v


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cube_cores = 24
        try:
            import torch_npu
            info = torch_npu.npu.npu_config.get_device_limit(0)
            for key in ("cube_core_num", "aicore_num", "aic_num", "num_aicore"):
                try:
                    v = int(info.get(key, 0))
                    if v > 0:
                        self.cube_cores = v
                        break
                except Exception:
                    pass
        except Exception:
            pass

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        M = A.shape[0]
        K = A.shape[1]
        N = B.shape[0]
        out = torch.empty((M, N), device=A.device, dtype=A.dtype)

        if A.dtype.itemsize == 2:  # fp16 / bf16
            BLOCK_M = _pow2_clamp(M, 16, 128)
            BLOCK_N = _pow2_clamp(N, 16, 128)
            BLOCK_K = _pow2_clamp(K, 16, 256)
        else:  # fp32
            BLOCK_M = _pow2_clamp(M, 16, 64)
            BLOCK_N = _pow2_clamp(N, 16, 64)
            BLOCK_K = _pow2_clamp(K, 16, 128)

        num_blocks = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        if num_blocks > self.cube_cores:
            num_progs = self.cube_cores
        else:
            num_progs = num_blocks

        _matmul_transb_kernel[(num_progs,)](
            A, B, out,
            M, N, K,
            NUM_PROGS=num_progs,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return out
