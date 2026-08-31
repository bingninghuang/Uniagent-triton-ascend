import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
    _HAS_TORCH_NPU = True
except Exception:
    _HAS_TORCH_NPU = False


@triton.jit
def matmul_both_trans_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_k, stride_a_m,
    stride_b_n, stride_b_k,
    stride_c_m, stride_c_n,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C (M, N) = A.T @ B.T, where A (K, M) and B (N, K) are row-major.
    # Strategy: D = B @ A  (N, M) with all-row-major tiles, then C = D^T.
    pid = tl.program_id(0).to(tl.int32)
    NUM_PIDS_M = tl.cdiv(M, BLOCK_M)
    NUM_PIDS_N = tl.cdiv(N, BLOCK_N)
    NUM_PIDS = NUM_PIDS_M * NUM_PIDS_N

    offs_m = tl.arange(0, BLOCK_M).to(tl.int32)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int32)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

    for pid_mn in range(pid, NUM_PIDS, num_cores):
        pid_m = pid_mn // NUM_PIDS_N
        pid_n = pid_mn - pid_m * NUM_PIDS_N   # avoid % (scalar-degrades)

        offs_am = pid_m * BLOCK_M + offs_m          # M dim (cols of A / rows of C)
        offs_bn = pid_n * BLOCK_N + offs_n          # N dim (rows of B / cols of C)
        mask_m = offs_am < M
        mask_n = offs_bn < N

        # A tile (BLOCK_K, BLOCK_M): A[k, m], row-major (contiguous along m / axis 1)
        a_ptrs = a_ptr + (offs_k[:, None] * stride_a_k
                          + offs_am[None, :] * stride_a_m)
        # B tile (BLOCK_N, BLOCK_K): B[n, k], row-major (contiguous along k / axis 1)
        b_ptrs = b_ptr + (offs_bn[:, None] * stride_b_n
                          + offs_k[None, :] * stride_b_k)

        # acc (BLOCK_N, BLOCK_M) = tile of D = B @ A;  C[m, n] = D[n, m]
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            mask_k = offs_k < K - k
            a = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :],
                        other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :],
                        other=0.0)
            acc = tl.dot(b, a, acc, out_dtype=tl.float32)
            a_ptrs += BLOCK_K * stride_a_k
            b_ptrs += BLOCK_K * stride_b_k

        c_ptrs = c_ptr + (offs_am[:, None] * stride_c_m
                          + offs_bn[None, :] * stride_c_n)
        tl.store(c_ptrs, tl.trans(acc).to(c_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        if _HAS_TORCH_NPU:
            try:
                limits = torch_npu.npu.npu_config.get_device_limit(0)
                self.CUBE_CORE_NUM = int(limits.get("cube_core_num", 20))
            except Exception:
                self.CUBE_CORE_NUM = 20
        else:
            self.CUBE_CORE_NUM = 20

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: (K, M), B: (N, K)  ->  C = A.T @ B.T : (M, N)
        K = A.shape[0]
        M = A.shape[1]
        N = B.shape[0]
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # clamp(next_power_of_2(x), 16, hi) with ternaries (no loops/calls)
        p2m = triton.next_power_of_2(M)
        BM = 16 if (p2m < 16) else (128 if (p2m > 128) else p2m)
        p2n = triton.next_power_of_2(N)
        BN = 16 if (p2n < 16) else (128 if (p2n > 128) else p2n)
        is_lowp = (A.dtype == torch.float16) or (A.dtype == torch.bfloat16)
        p2k = triton.next_power_of_2(K)
        if is_lowp:
            BK = 16 if (p2k < 16) else (128 if (p2k > 128) else p2k)
        else:
            BK = 16 if (p2k < 16) else (64 if (p2k > 64) else p2k)

        num_blocks = ((M + BM - 1) // BM) * ((N + BN - 1) // BN)
        cores = self.CUBE_CORE_NUM
        grid = (num_blocks if (num_blocks < cores) else cores,)

        matmul_both_trans_kernel[grid](
            A, B, C,
            M, N, K,
            M, 1,     # A strides: (K, M) row-major -> (k stride M, m stride 1)
            K, 1,     # B strides: (N, K) row-major -> (n stride K, k stride 1)
            N, 1,     # C strides: (M, N) row-major
            cores,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        return C
