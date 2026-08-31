import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:  # pragma: no cover - torch_npu should exist on NPU hosts
    torch_npu = None


@triton.jit
def _grouped_gemm_kernel(
    a_ptr,        # (R, K)  A treated as flat (num_rows_total, K)
    b_ptr,        # (L, K, N)
    c_ptr,        # (R_out, N)
    offs_ptr,     # (L,) int32 end offsets (2D case), unused for 3D
    num_groups,   # L
    K,
    N,
    m_per_group,  # 3D rows per group (m), unused for 2D
    num_cores: tl.constexpr,
    IS_2D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_nb = tl.cdiv(N, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    col_mask = offs_n[None, :] < N
    offset = 0
    for g in range(0, num_groups):
        if IS_2D:
            end = tl.load(offs_ptr + g)
            start = tl.load(offs_ptr + g - 1, mask=(g > 0), other=0)
        else:
            start = g * m_per_group
            end = start + m_per_group
        m_g = end - start
        n_mb = tl.cdiv(m_g, BLOCK_M)
        tasks_g = n_mb * n_nb
        t_first = (pid - offset) % num_cores
        t_first = tl.where(t_first < 0, t_first + num_cores, t_first)
        for t in range(t_first, tasks_g, num_cores):
            mb = t // n_nb
            bn = t - mb * n_nb
            row_base = start + mb * BLOCK_M
            col_base = bn * BLOCK_N
            b_group_base = g * (K * N)
            a_rows = row_base + offs_m
            a_row_mask = a_rows[:, None] < end
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                k_offs = k0 + offs_k
                k_mask = k_offs < K
                a = tl.load(
                    a_ptr + a_rows[:, None] * K + k_offs[None, :],
                    mask=a_row_mask & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_ptr + b_group_base + k_offs[:, None] * N
                    + col_base + offs_n[None, :],
                    mask=k_mask[:, None] & col_mask,
                    other=0.0,
                )
                acc = tl.dot(a, b, acc)
            c_rows = row_base + offs_m
            tl.store(
                c_ptr + c_rows[:, None] * N + col_base + offs_n[None, :],
                acc,
                mask=(c_rows[:, None] < end) & col_mask,
            )
        offset += tasks_g


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        core_num = 20
        if torch_npu is not None:
            try:
                core_num = torch_npu.npu.npu_config.get_device_limit(0).get(
                    "cube_core_num", 20
                )
            except Exception:
                core_num = 20
        self.CUBE_CORE_NUM = int(core_num)
        self._dummy_int32 = None

    def forward(self, A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor = None):
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()

        L, K, N = B.shape
        if A.dim() == 3:
            m = A.shape[1]
            R = L * m
            is_2d = False
            m_for_tile = m
        else:
            R = A.shape[0]
            is_2d = True
            m_for_tile = R // L if L > 0 else R

        out = torch.empty((R, N), device=A.device, dtype=A.dtype)

        if R == 0 or N == 0 or K == 0:
            return out

        if is_2d:
            offs = offsets
            if offs.dtype != torch.int32:
                offs = offs.to(torch.int32)
            if not offs.is_contiguous():
                offs = offs.contiguous()
        else:
            if self._dummy_int32 is None:
                self._dummy_int32 = torch.empty(
                    1, dtype=torch.int32, device=A.device
                )
            offs = self._dummy_int32

        if m_for_tile >= 128:
            BLOCK_M = 128
        elif m_for_tile >= 64:
            BLOCK_M = 64
        elif m_for_tile >= 32:
            BLOCK_M = 32
        else:
            BLOCK_M = 16
        if N >= 128:
            BLOCK_N = 128
        elif N >= 64:
            BLOCK_N = 64
        elif N >= 32:
            BLOCK_N = 32
        else:
            BLOCK_N = 16
        if K >= 64:
            BLOCK_K = 64
        elif K >= 32:
            BLOCK_K = 32
        else:
            BLOCK_K = 16

        grid = (self.CUBE_CORE_NUM,)
        _grouped_gemm_kernel[grid](
            A,
            B,
            out,
            offs,
            L,
            K,
            N,
            m_for_tile,
            num_cores=self.CUBE_CORE_NUM,
            IS_2D=is_2d,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return out