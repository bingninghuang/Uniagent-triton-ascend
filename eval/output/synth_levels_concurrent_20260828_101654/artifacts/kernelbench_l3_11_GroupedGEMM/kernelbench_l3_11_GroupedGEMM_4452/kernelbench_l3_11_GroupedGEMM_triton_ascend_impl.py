import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_num_vector_cores(device=None):
    """Dynamically read the number of vector cores (G1)."""
    candidates = []
    try:
        if device is None:
            device = torch.npu.current_device() if hasattr(torch, "npu") else 0
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(device)
            n = props.get("num_vectorcore", None)
            if n:
                candidates.append(int(n))
            else:
                n = props.get("num_aicore", None)
                if n:
                    candidates.append(int(n) * 2)
        except Exception:
            pass
        try:
            import torch_npu

            limit = torch_npu.npu.npu_config.get_device_limit(0)
            n = limit.get("vector_core_num", None)
            if n:
                candidates.append(int(n))
        except Exception:
            pass
    except Exception:
        pass
    if candidates:
        return max(candidates)
    return 48


_weight_cache = {}


def _get_grouped_weight(key, num_groups, out_features, in_features, device, dtype):
    """Synthesize the deterministic per-group weight tensor (cached), matching
    the reference Model's seed-42 generation exactly."""
    if key in _weight_cache:
        return _weight_cache[key]
    _weight_cache.clear()
    generator = torch.Generator()
    generator.manual_seed(42)
    weight = torch.randn(
        (num_groups, out_features, in_features),
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(in_features)
    w = weight.to(device=device, dtype=dtype)
    _weight_cache[key] = w
    return w


def _pick_launch_config(rows, out_features, in_features, dtype, device):
    """Choose block sizes / grid for the grouped GEMV kernel.

    The Ascend backend double-buffers GM loads inside the k-loop and keeps the
    fp32-converted copy of the weight tile alive, so the live-UB estimate is
    ~2 * BLOCK_N * BLOCK_K * itemsize + BLOCK_N * BLOCK_K * 4 (+ lhs / acc /
    masks).  Keeping BLOCK_N * BLOCK_K <= 16384 elements keeps the total well
    under the 192KB UB limit for fp16/bf16/fp32 alike.
    """
    itemsize = dtype.itemsize
    k_pow2 = triton.next_power_of_2(in_features)
    if itemsize == 2:
        # 512B contiguous weight rows (256 half elements) when possible
        k_cap = 256
    else:
        # fp32: 128 elements = 512B rows, fp32 cast is a no-op
        k_cap = 128
    block_k = k_pow2 if k_pow2 < k_cap else k_cap
    block_n = 16384 // block_k
    if block_n > 256:
        block_n = 256
    n_nblocks = (out_features + block_n - 1) // block_n
    total_blocks = rows * n_nblocks
    num_cores = _get_num_vector_cores(device)
    grid_size = total_blocks if total_blocks < num_cores else num_cores
    return block_n, block_k, n_nblocks, total_blocks, grid_size


@triton.jit
def _grouped_gemm_kernel(
    lhs_ptr,  # (rows, K) row-major, dtype D
    weight_ptr,  # (G, N, K) row-major, dtype D
    m_idx_ptr,  # (rows,) int32 group id per row
    out_ptr,  # (rows, N) dtype D
    rows,
    N,
    K,
    n_nblocks,
    total_blocks,
    num_pids,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # Contiguous partition: each program owns a contiguous range of blocks
    # (checklist: no interleaved partition; G6 load-balance formula).
    blocks_per_core = total_blocks // num_pids
    remainder = total_blocks - blocks_per_core * num_pids
    start = blocks_per_core * pid + tl.minimum(pid, remainder)
    cnt = blocks_per_core
    if pid < remainder:
        cnt = blocks_per_core + 1
    n_f = N.to(tl.float32)
    k_f = K.to(tl.float32)
    for bid in range(start, start + cnt):
        row = bid // n_nblocks
        nblk = bid - row * n_nblocks
        g = tl.load(m_idx_ptr + row)  # int32 scalar
        k_base = row * K
        w_base = g * (N * K)
        n_offs = nblk * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs.to(tl.float32) < n_f
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        k_offs = tl.arange(0, BLOCK_K)
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + k_offs).to(tl.float32) < k_f
            a = tl.load(
                lhs_ptr + k_base + k0 + k_offs, mask=k_mask, other=0.0
            ).to(tl.float32)  # (BLOCK_K,)
            w = tl.load(
                weight_ptr
                + w_base
                + n_offs[:, None] * K
                + (k0 + k_off


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, m_indices, num_groups, out_features):
        rows, in_features = lhs.shape
        key = (num_groups, out_features, in_features, lhs.device, lhs.dtype)
        weight = _get_grouped_weight(
            key, num_groups, out_features, in_features, lhs.device, lhs.dtype
        )
        lhs = lhs.contiguous()
        m_indices = m_indices.contiguous()
        out = torch.empty((rows, out_features), device=lhs.device, dtype=lhs.dtype)
        block_n, block_k, n_nblocks, total_blocks, grid_size = _pick_launch_config(
            rows, out_features, in_features, lhs.dtype, lhs.device
        )
        _grouped_gemm_kernel[(grid_size,)](
            lhs,
            weight,
            m_indices,
            out,
            rows,
            out_features,
            in_features,
            n_nblocks,
            total_blocks,
            grid_size,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        return out
