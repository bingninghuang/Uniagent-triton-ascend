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
        # avoid remainder operator (always scalar-lowered on this backend);
        # equivalent to block_idx % NUM_BLOCKS_N
        block_n = block_idx - block_m * NUM_BLOCKS_N

        m_base = block_m * BLOCK_M
        n_base = block_n * BLOCK_N

        # Define row/col offsets OUTSIDE the k-loop so they are visible
        # after the loop for the store.
        offs_2d_m = m_base + offs_m          # [BLOCK_M]
        offs_2d_n = n_base + offs_n          # [BLOCK_N]

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_2d_k = k + offs_k               # [BLOCK_K]

            # A tile [BLOCK_M, BLOCK_K]: A is [M, K] row-major, K contiguous
            a_mask = (offs_2d_m[:, None] < M) & (offs_2d_k[None, :] < K)
            a = tl.load(
                a_ptr + offs_2d_m[:, None] * K + offs_2d_k[None, :],
                mask=a_mask, other=0.0,
            )

            # B is [N, K] row-major. Load the natural [BLOCK_N, BLOCK_K] tile
            # (K contiguous, coalesced) and transpose in-register to the
            # [BLOCK_K, BLOCK_N] right-operand layout expected by CUBE.
            b_mask = (offs_2d_n[:, None] < N) & (offs_2d_k[None, :] < K)
            b_nk = tl.load(
                b_ptr + offs_2d_n[:, None] * K + offs_2d_k[None, :],
                mask=b_mask, other=0.0,
            )
            b = tl.trans(b_nk)

            accumulator += tl.dot(a, b)

        # C tile [BLOCK_M, BLOCK_N]: C = A @ B^T, C is [M, N] row-major
        c_mask = (offs_2d_m[:, None] < M) & (offs_2d_n[None, :] < N)
        c_val = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(
            c_ptr + offs_2d_m[:, None] * N + offs_2d_n[None, :],
            c_val, mask=c_mask,
        )


def _next_pow2_le(x):
    # Largest power of two <= x (>= 1).
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _pick_config(M, K, N, el_bytes):
    # BLOCK_M: power of two in [16, 128] (CUBE m0 range). Thin long-K shapes
    # halve m0 so the M dimension yields more output blocks (parallel cores).
    bm = max(16, min(128, _next_pow2_le(M)))
    if M <= 64 and K >= 512:
        bm = max(16, bm // 2)

    # BLOCK_K: power of two in [16, cap]; fp32 capped at 128 (512B row width,
    # half the L0 budget), fp16/bf16 at 256 (L0A/L0B 64KB exactly).
    cap_k = 128 if el_bytes == 4 else 256
    bk = max(16, min(cap_k, _next_pow2_le(K)))

    # BLOCK_N: choose power of two in [16, 128] so the total number of C-tile
    # blocks stays close to the CUBE core count (target 16 active cores).
    # L0B constraint: BLOCK_N * BLOCK_K * el_bytes <= 64KB.
    nbm = (M + bm - 1) // bm
    best_bn, best_err = 16, None
    for bn in (128, 64, 32, 16):
        if bn * bk * el_bytes > 65536:
            continue
        err = abs(nbm * ((N + bn - 1) // bn) - 16)
        if best_err is None or err < best_err or (err == best_err and bn < best_bn):
            best_bn, best_err = bn, err
    return bm, best_bn, bk


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # CUBE core count for MatMul; read once at init, never in forward.
        self.CUBE_CORE_NUM = 20
        if torch_npu is not None:
            try:
                self.CUBE_CORE_NUM = int(
                    torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
                )
            except Exception:
                self.CUBE_CORE_NUM = 20
        if self.CUBE_CORE_NUM <= 0:
            self.CUBE_CORE_NUM = 20

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape[0], A.shape[1]
        N = B.shape[0]
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        el_bytes = 4 if A.dtype == torch.float32 else 2
        bm, bn, bk = _pick_config(M, K, N, el_bytes)
        num_blocks = ((M + bm - 1) // bm) * ((N + bn - 1) // bn)
        # Small-grid path: 1 block per program (no multi-block partition loop
        # iterations); large-grid path: grid clamped to the CUBE core count and
        # each core strides through its share of blocks.
        grid_size = min(num_blocks, self.CUBE_CORE_NUM)

        matmul_transb_kernel[(grid_size,)](
            A, B, C, M, N, K, self.CUBE_CORE_NUM, bm, bn, bk
        )
        return C
