import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:  # pragma: no cover
    torch_npu = None

# Fallback only used if the device query fails (910B1 has 24 CUBE cores).
FALLBACK_CUBE_CORES = 24


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
    offs_ptr,
    total_tiles,
    chunk_tiles,
    K: tl.constexpr,
    N: tl.constexpr,
    num_pid_m: tl.constexpr,
    num_pid_n: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    IS_2D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_PROGS: tl.constexpr,
    G_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load the whole (small) group-boundary table once, outside the tile
    # loop. Per-tile start/end are then extracted from these vectors.
    if IS_2D:
        g_idx = tl.arange(0, G_BLOCK)
        g_mask = (g_idx.to(tl.float32) < G)
        ends_vec = tl.load(offs_ptr + g_idx, mask=g_mask, other=0)
        starts_vec = tl.load(
            offs_ptr + g_idx - 1,
            mask=g_mask & (g_idx.to(tl.float32) > 0),
            other=0,
        )

    # Contiguous task split (no interleaving): every program owns a
    # contiguous run of tiles. First `rem` programs take one extra tile.
    rem_tiles = total_tiles - NUM_PROGS * chunk_tiles
    extra = (pid.to(tl.float32) < rem_tiles.to(tl.float32)).to(tl.int32)
    count = chunk_tiles + extra
    tile_base = pid * chunk_tiles + tl.minimum(pid, rem_tiles)

    for local in range(0, count):
        tile_idx = tile_base + local
        tiles_per_g = num_pid_m * num_pid_n
        g = tile_idx // tiles_per_g
        pid_in = tile_idx - g * tiles_per_g
        pid_m = pid_in // num_pid_n
        pid_n = pid_in - pid_m * num_pid_n

        if IS_2D:
            end = tl.sum(tl.where(g_idx == g, ends_vec, 0))
            start = tl.sum(tl.where(g_idx == g, starts_vec, 0))
            m_g = end - start
        else:
            start = g * M
            m_g = M

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # fp32 comparisons (vectorized path per code conventions)
        row_mask = offs_m.to(tl.float32) < m_g.to(tl.float32)
        col_mask = offs_n.to(tl.float32) < N

        a_row_off = (start + offs_m)[:, None] * K
        b_tile_off = g * K * N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_off = k0 + offs_k
            k_mask = k_off.to(tl.float32) < K
            a = tl.load(
                A_ptr + a_row_off + k_off[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                B_ptr + b_tile_off + k_off[:, None] * N + offs_n[None, :],
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(a, b, acc)

        c_ptrs = C_ptr + (start + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(
            c_ptrs,
            acc.to(C_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Mix (matrix + vector) kernel: grid may not exceed the CUBE core
        # count; read it dynamically (never hardcode the core count).
        self._cube_cores = FALLBACK_CUBE_CORES
        if torch_npu is not None:
            try:
                nc = int(
                    torch_npu.npu.npu_config.get_device_limit(0).get(
                        'cube_core_num', 0
                    )
                )
                if nc > 0:
                    self._cube_cores = nc
            except Exception:
                pass

    def forward(self, A: torch.Tensor, B: torch.Tensor,
                offsets=None) -> torch.Tensor:
        if A.dim() == 2:
            total_rows, K = A.shape
            G = B.shape[0]
            N = B.shape[2]
            m_g_est = total_rows // G if (total_rows % G == 0) else total_rows
            is_2d = True
            m_val = m_g_est
        else:
            G, m_val, K = A.shape
            total_rows = G * m_val
            N = B.shape[2]
            m_g_est = m_val
            is_2d = False

        A_flat = A.contiguous().view(total_rows, K)
        B_c = B.contiguous()
        C_flat = torch.empty((total_rows, N), dtype=A.dtype, device=A.device)
        offs_arg = offsets if is_2d else A_flat

        BLOCK_M = 16 if m_g_est <= 16 else (
            32 if m_g_est <= 32 else 64
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

        if is_2d:
            num_pid_m = _cdiv(total_rows // G, BLOCK_M) if (
                total_rows % G == 0) else _cdiv(total_rows, BLOCK_M)
        else:
            num_pid_m = _cdiv(m_val, BLOCK_M)
        num_pid_n = _cdiv(N, BLOCK_N)
        tiles_total = G * num_pid_m * num_pid_n
        # One program per AI Core at most; each program owns a contiguous
        # chunk of tiles (first `tiles_total % num_progs` take one extra).
        if tiles_total < self._cube_cores:
            num_progs = tiles_total
        else:
            num_progs = self._cube_cores
        chunk_tiles = tiles_total // num_progs

        _grouped_gemm_kernel[(num_progs,)](
            A_flat,
            B_c,
            C_flat,
            offs_arg,
            tiles_total,
            chunk_tiles,
            K,
            N,
            num_pid_m,
            num_pid_n,
            G,
            m_val,
            IS_2D=is_2d,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            NUM_PROGS=num_progs,
            G_BLOCK=_next_pow2(G),
        )

        if is_2d:
            return C_flat
        return C_flat.view(G, m_val, N)