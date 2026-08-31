import torch
import triton
import triton.language as tl


def _get_cube_core_num():
    try:
        import torch_npu

        lim = torch_npu.npu.npu_config.get_device_limit(0)
        n = int(lim.get("cube_core_num", 24))
        return n if n > 0 else 24
    except Exception:
        return 24


# Point 3: parameters that are fixed for the lifetime of a single launch
# (problem sizes, batch count, strides) are declared as tl.constexpr so the
# compiler can constant-fold the tiling math, unroll the K loop when it is
# short, and drop runtime bounds checks.
#
# Point 12: two specialized kernel paths.
#  - bmm_kernel_small: total_blocks <= num cores. Each program handles exactly
#    one (batch, m-tile, n-tile) block; no grid-stride loop, so the compiler
#    can issue all MTE loads back-to-back ahead of the cube ops.
#  - bmm_kernel_large: total_blocks > num cores. Persistent cores with a
#    grid-stride loop over the block space.

# Point 12: small-grid path. One program per output tile, direct pid mapping,
# no grid-stride loop, so all MTE loads can be issued ahead of the cube ops.
@triton.jit
def bmm_kernel_small(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_B: tl.constexpr,
    sa_b: tl.constexpr,
    sa_m: tl.constexpr,
    sa_k: tl.constexpr,
    sb_b: tl.constexpr,
    sb_k: tl.constexpr,
    sb_n: tl.constexpr,
    sc_b: tl.constexpr,
    sc_m: tl.constexpr,
    sc_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    blocks_per_batch = tl.cdiv(M, BLOCK_M) * num_pid_n
    pid_b = pid // blocks_per_batch
    pid_m = (pid % blocks_per_batch) // num_pid_n
    pid_n = pid % num_pid_n

    a_base = a_ptr + pid_b * sa_b
    b_base = b_ptr + pid_b * sb_b
    c_base = c_ptr + pid_b * sc_b

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_base + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = b_base + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n

    mask_m = (offs_m < M)[:, None]
    mask_n = (offs_n < N)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = mask_m & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & mask_n
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_k

    c_ptrs = c_base + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
    c_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# Point 12: large-grid path. Persistent cores with a grid-stride loop when
# there are more output tiles than cube cores.
@triton.jit
def bmm_kernel_large(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_B: tl.constexpr,
    sa_b: tl.constexpr,
    sa_m: tl.constexpr,
    sa_k: tl.constexpr,
    sb_b: tl.constexpr,
    sb_k: tl.constexpr,
    sb_n: tl.constexpr,
    sc_b: tl.constexpr,
    sc_m: tl.constexpr,
    sc_n: tl.constexpr,
    NUM_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    blocks_per_batch = num_pid_m * num_pid_n
    total_blocks = blocks_per_batch * NUM_B

    for block_idx in range(pid, total_blocks, NUM_C):
        pid_b = block_idx // blocks_per_batch
        pid_m = (block_idx % blocks_per_batch) // num_pid_n
        pid_n = block_idx % num_pid_n

        a_base = a_ptr + pid_b * sa_b
        b_base = b_ptr + pid_b * sb_b
        c_base = c_ptr + pid_b * sc_b

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_base + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
        b_ptrs = b_base + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n

        mask_m = (offs_m < M)[:, None]
        mask_n = (offs_n < N)[None, :]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a_mask = mask_m & ((k + offs_k[None, :]) < K)
            b_mask = ((k + offs_k[:, None]) < K) & mask_n
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * sa_k
            b_ptrs += BLOCK_K * sb_k

        c_ptrs = c_base + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
        c_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.CUBE_CORE_NUM = _get_cube_core_num()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.contiguous()
        B = B.contiguous()
        num_b, M, K = A.shape
        K2, N = B.shape[1], B.shape[2]
        C = torch.empty((num_b, M, N), device=A.device, dtype=A.dtype)

        # Quantize block sizes to a small fixed set to keep the number of
        # distinct compiled kernels small.
        if M >= 128 and N >= 128:
            BM, BN = 32, 32
        elif M >= 32 and N >= 32:
            if num_b >= 2:
                BM, BN = 32, 32
            else:
                BM, BN = 16, 16
        else:
            BM, BN = 16, 16
        # Point 22: latency-bound — each tl.dot pays a fixed issue + cube/vector
        # sync cost regardless of its size. Size BLOCK_K to the full K (smallest
        # power of 2 >= K, min 16, capped at 256 for the UB budget) so each
        # program issues exactly ONE dot instead of K/BK dots in a loop.
        if K <= 16:
            BK = 16
        elif K <= 32:
            BK = 32
        elif K <= 64:
            BK = 64
        elif K <= 128:
            BK = 128
        else:
            BK = 256

        # Point 12: small-grid path when every core handles a single block,
        # persistent large-grid path otherwise.
        total_blocks = num_b * ((M + BM - 1) // BM) * ((N + BN - 1) // BN)
        # Point 3: compile options. multibuffer=True double-buffers the A/B
        # tiles so MTE1 DMA of the next block overlaps the current block's
        # cube dot (targets the cube_stall observed in profiling).
        if total_blocks <= self.CUBE_CORE_NUM:
            bmm_kernel_small[(total_blocks,)](
                A, B, C,
                M, N, K, num_b,
                A.stride(0), A.stride(1), A.stride(2),
                B.stride(0), B.stride(1), B.stride(2),
                C.stride(0), C.stride(1), C.stride(2),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                multibuffer=True,
            )
        else:
            bmm_kernel_large[(self.CUBE_CORE_NUM,)](
                A, B, C,
                M, N, K, num_b,
                A.stride(0), A.stride(1), A.stride(2),
                B.stride(0), B.stride(1), B.stride(2),
                C.stride(0), C.stride(1), C.stride(2),
                self.CUBE_CORE_NUM,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                multibuffer=True,
            )
        return C
