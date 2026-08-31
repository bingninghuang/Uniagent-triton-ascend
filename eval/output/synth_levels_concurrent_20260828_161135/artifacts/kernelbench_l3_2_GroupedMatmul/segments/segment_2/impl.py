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
    offs_ptr,
    K,
    N,
    num_pid_m,
    num_pid_n,
    G,
    M,
    IS_2D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_g = num_pid_m * num_pid_n
    g = pid // tiles_per_g
    pid_in = pid % tiles_per_g
    pid_m = pid_in // num_pid_n
    pid_n = pid_in % num_pid_n

    if IS_2D:
        end = tl.load(offs_ptr + g)
        start = tl.load(offs_ptr + g - 1, mask=(g > 0), other=0)
        m_g = end - start
    else:
        start = g * M
        m_g = M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    row_mask = offs_m < m_g
    col_mask = offs_n < N

    a_ptrs = A_ptr + (start + offs_m)[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + g * K * N + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = C_ptr + (start + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(C_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

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
        BLOCK_N = min(64, max(16, _next_pow2(N)))
        BLOCK_K = 64 if K >= 64 else min(64, max(16, _next_pow2(K)))

        if is_2d:
            num_pid_m = _cdiv(total_rows // G, BLOCK_M) if (
                total_rows % G == 0) else _cdiv(total_rows, BLOCK_M)
        else:
            num_pid_m = _cdiv(m_val, BLOCK_M)
        num_pid_n = _cdiv(N, BLOCK_N)
        grid = (G * num_pid_m * num_pid_n,)

        _grouped_gemm_kernel[grid](
            A_flat,
            B_c,
            C_flat,
            offs_arg,
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
        )

        if is_2d:
            return C_flat
        return C_flat.view(G, m_val, N)