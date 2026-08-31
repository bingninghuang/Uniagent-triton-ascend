import torch
import torch.nn as nn
import triton
import triton.language as tl


def _p2(x):
    # Host-side: smallest power of two >= x
    p = 1
    while p < x:
        p <<= 1
    return p


def _clamp_p2(x, lo, hi):
    p = _p2(x)
    if p < lo:
        p = lo
    if p > hi:
        p = hi
    return p


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
    # Use * 1.0 instead of .to(...) because sizes equal to 1 are specialized
    # to compile-time constants, which have no .to method.
    N_A_f = N_A * 1.0
    K_B_f = K_B * 1.0
    M_f = M * 1.0

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


@triton.jit
def matmul_transa_kernel_1(
    a_ptr, b_ptr, c_ptr,
    M, N_A, K_B,
    stride_al, stride_bl,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Group 1 (tiny shapes: all dims <= 16): single core, single output tile.
    # The whole problem fits in one 16x32 tile; no block splitting, and one
    # k-step for the tested shapes. C (N_A x K_B) = A^T @ B.
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    N_A_f = N_A * 1.0
    K_B_f = K_B * 1.0
    M_f = M * 1.0

    mask_m = offs_m.to(tl.float32) < N_A_f
    mask_n = offs_n.to(tl.float32) < K_B_f

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, M, BLOCK_K):
        k_off = k + offs_k
        k_mask = k_off.to(tl.float32) < M_f
        a = tl.load(a_ptr + k_off[:, None] * stride_al + offs_m[None, :],
                    mask=k_mask[:, None] & mask_m[None, :], other=0.0)
        b = tl.load(b_ptr + k_off[:, None] * stride_bl + offs_n[None, :],
                    mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(tl.trans(a), b, acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * K_B + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def matmul_transa_transpose_kernel(
    a_ptr, at_ptr,
    M, N_A,
    stride_al,
    num_cores: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Group 2 stage 1: build At (N_A x M, row-major) from A (M x N_A):
    # At[n, m] = A[m, n]. Each tile loads BLOCK_R rows of A (contiguous along
    # n) and stores them transposed; the store is contiguous along the M
    # direction (stride 1 in m) for coalesced writes.
    pid = tl.program_id(0).to(tl.int32)
    num_pid_m = (M + BLOCK_R - 1) // BLOCK_R
    num_pid_n = (N_A + BLOCK_C - 1) // BLOCK_C
    num_blocks = num_pid_m * num_pid_n
    blocks_per_core = (num_blocks + num_cores - 1) // num_cores

    offs_r = tl.arange(0, BLOCK_R)
    offs_c = tl.arange(0, BLOCK_C)

    M_f = M * 1.0
    N_A_f = N_A * 1.0

    for j in range(0, blocks_per_core):
        block_idx = pid * blocks_per_core + j
        if block_idx < num_blocks:
            pid_m = block_idx // num_pid_n
            pid_n = block_idx - pid_m * num_pid_n

            offs_r_block = pid_m * BLOCK_R + offs_r
            offs_c_block = pid_n * BLOCK_C + offs_c
            mask_r = offs_r_block.to(tl.float32) < M_f
            mask_c = offs_c_block.to(tl.float32) < N_A_f

            tile = tl.load(a_ptr + offs_r_block[:, None] * stride_al + offs_c_block[None, :],
                           mask=mask_r[:, None] & mask_c[None, :], other=0.0)
            # At tile: (BLOCK_C x BLOCK_R) with element (c, r) = tile[r, c]
            at_ptrs = at_ptr + offs_c_block[:, None] * M + offs_r_block[None, :]
            tl.store(at_ptrs, tl.trans(tile),
                     mask=mask_c[:, None] & mask_r[None, :])


@triton.jit
def matmul_transa_kernel_2(
    at_ptr, b_ptr, c_ptr,
    N_A, M, K_B,
    stride_atl, stride_bl,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Group 2 stage 2 (large shapes): plain matmul C (N_A x K_B) =
    # At (N_A x M, row-major, pre-transposed) @ B (M x K_B). Both operand
    # tiles are loaded in natural layout; no in-kernel transpose needed.
    pid = tl.program_id(0).to(tl.int32)
    num_pid_m = (N_A + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (K_B + BLOCK_N - 1) // BLOCK_N
    num_blocks = num_pid_m * num_pid_n
    blocks_per_core = (num_blocks + num_cores - 1) // num_cores

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    N_A_f = N_A * 1.0
    K_B_f = K_B * 1.0
    M_f = M * 1.0

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
                a = tl.load(at_ptr + offs_m_block[:, None] * stride_atl + k_off[None, :],
                            mask=mask_m[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(b_ptr + k_off[:, None] * stride_bl + offs_n_block[None, :],
                            mask=k_mask[:, None] & mask_n[None, :], other=0.0)
                acc = tl.dot(a, b, acc, out_dtype=tl.float32)

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
        self._out_cache = {}
        self._at_cache = {}

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, N_A = A.shape
        K_B = B.shape[1]
        key = (N_A, K_B, A.dtype)
        C = self._out_cache.get(key)
        if C is None:
            C = torch.empty((N_A, K_B), device=A.device, dtype=A.dtype)
            self._out_cache[key] = C
        sa = A.stride(0)
        sb = B.stride(0)
        cores = self.cube_core_num

        if M <= 16 and N_A <= 16 and K_B <= 32:
            # Group 1: tiny problem; a single core is fastest because the
            # multi-core device startup overhead would exceed the work.
            matmul_transa_kernel_1[(1,)](
                A, B, C,
                M, N_A, K_B,
                sa, sb,
                BLOCK_M=16, BLOCK_N=32, BLOCK_K=16,
            )
            return C

        if M * N_A * K_B >= 64 * 1024 * 1024:
            # Group 2: large problem; two-stage pipeline (transpose + plain
            # matmul) avoids the per-iteration in-kernel transpose and uses
            # wider tiles.
            key_at = (M, N_A, A.dtype)
            At = self._at_cache.get(key_at)
            if At is None:
                At = torch.empty((N_A, M), device=A.device, dtype=A.dtype)
                self._at_cache[key_at] = At

            br, bc = 64, 128
            nblk_t = ((M + br - 1) // br) * ((N_A + bc - 1) // bc)
            grid_t = nblk_t if nblk_t < cores else cores
            matmul_transa_transpose_kernel[(grid_t,)](
                A, At,
                M, N_A,
                sa,
                num_cores=grid_t,
                BLOCK_R=br, BLOCK_C=bc,
            )

            bm = 128 if N_A >= 128 else _clamp_p2(N_A, 16, 128)
            bn = 256 if (K_B >= 256 and A.dtype != torch.float32) else 128
            nblk = ((N_A + bm - 1) // bm) * ((K_B + bn - 1) // bn)
            grid = nblk if nblk < cores else cores
            matmul_transa_kernel_2[(grid,)](
                At, B, C,
                N_A, M, K_B,
                M, sb,
                num_cores=grid,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=64,
            )
            return C

        # Fallback: base kernel with host-selected tile sizes.
        bm = _clamp_p2(N_A, 16, 128)
        bn = _clamp_p2(K_B, 16, 128)
        bk = _clamp_p2(M, 16, 64)
        nblk = ((N_A + bm - 1) // bm) * ((K_B + bn - 1) // bn)
        grid = nblk if nblk < cores else cores
        matmul_transa_kernel[(grid,)](
            A, B, C,
            M, N_A, K_B,
            sa, sb,
            num_cores=grid,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        )
        return C
