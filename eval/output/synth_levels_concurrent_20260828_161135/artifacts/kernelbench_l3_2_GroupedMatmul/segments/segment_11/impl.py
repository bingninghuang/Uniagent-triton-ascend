import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_pow2(x):
    n = 1
    while n < x:
        n <<= 1
    return n


def _cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def _grouped_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    num_pid_m: tl.constexpr,
    num_pid_n: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    tiles_per_prog: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_g = num_pid_m * num_pid_n
    total_tiles = G * tiles_per_g

    t_start = pid * tiles_per_prog
    t_end = t_start + tiles_per_prog
    if t_end > total_tiles:
        t_end = total_tiles

    for t in range(t_start, t_end):
        g = t // tiles_per_g
        r = t - g * tiles_per_g
        pid_m = r // num_pid_n
        pid_n = r - pid_m * num_pid_n

        start = g * M

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        row_mask = offs_m.to(tl.float32) < (M + 0.0)
        col_mask = offs_n.to(tl.float32) < (N + 0.0)

        a_row_off = (start + offs_m)[:, None] * K
        b_tile_off = g * K * N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_off = k0 + offs_k
            k_mask = k_off.to(tl.float32) < (K + 0.0)
            a = tl.load(A_ptr + a_row_off + k_off[None, :],
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(B_ptr + b_tile_off + k_off[:, None] * N + offs_n[None, :],
                        mask=k_mask[:, None] & col_mask[None, :], other=0.0)
            acc = tl.dot(a, b, acc)

        c_ptrs = C_ptr + (start + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(
            c_ptrs,
            acc.to(C_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )


def _query_num_cores():
    cores = 0
    try:
        import torch_npu

        limit = torch_npu.npu.npu_config.get_device_limit(0)
        cores = limit.get("cube_core_num", 0)
        if not cores:
            cores = limit.get("ai_core_num", 0)
        if not cores:
            v = limit.get("vector_core_num", 0)
            cores = v // 2 if v else 0
    except Exception:
        cores = 0
    if cores is None or cores <= 0:
        cores = 24  # ascend910b1: 24 AI cores
    return cores


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cores = _query_num_cores()

    def forward(self, A: torch.Tensor, B: torch.Tensor,
                offsets=None) -> torch.Tensor:
        is_2d = A.dim() == 2
        if is_2d:
            total_rows, K = A.shape
            G = B.shape[0]
            N = B.shape[2]
            m_g = total_rows // G if (total_rows % G == 0) else total_rows
        else:
            G, m_g, K = A.shape
            total_rows = G * m_g
            N = B.shape[2]

        A_flat = A.contiguous().view(total_rows, K)
        B_c = B.contiguous()
        C_flat = torch.empty((total_rows, N), dtype=A.dtype, device=A.device)

        BLOCK_M = 16 if m_g <= 16 else (
            32 if m_g <= 32 else 64
        )
        BLOCK_N = 16
        if N > 16:
            BLOCK_N = 32
            if N > 32:
                BLOCK_N = 64
        if K < 64:
            BLOCK_K = _next_pow2(K)
            if BLOCK_K < 16:
                BLOCK_K = 16
        else:
            BLOCK_K = 64

        num_pid_m = _cdiv(m_g, BLOCK_M)
        num_pid_n = _cdiv(N, BLOCK_N)
        total_tiles = G * num_pid_m * num_pid_n
        if total_tiles < self.num_cores:
            grid_size = total_tiles
        else:
            grid_size = self.num_cores
        tiles_per_prog = _cdiv(total_tiles, grid_size)

        _grouped_gemm_kernel[(grid_size,)](
            A_flat,
            B_c,
            C_flat,
            K,
            N,
            num_pid_m,
            num_pid_n,
            G,
            m_g,
            tiles_per_prog=tiles_per_prog,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        if is_2d:
            return C_flat
        return C_flat.view(G, m_g, N) 