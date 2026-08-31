import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


_WEIGHT_CACHE = {}


def _get_num_cores(device_index=0):
    """Dynamically read the AI core count; never hardcode (G1)."""
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(device_index)
        if props is not None:
            get = getattr(props, "get", None)
            if get is not None:
                for key in ("num_aicore", "aicore_num", "num_cores", "num_ai_cores"):
                    v = get(key)
                    if v:
                        return int(v)
    except Exception:
        pass
    try:
        import torch_npu

        limit = torch_npu.npu.npu_config.get_device_limit(0)
        v = limit.get("vector_core_num", 48)
        if v:
            return int(v) // 2
    except Exception:
        pass
    return 24


def _build_weight(device, dtype, num_groups, out_features, in_features):
    """Deterministic weight cache; identical semantics to the reference Model."""
    key = (num_groups, out_features, in_features, device, dtype)
    weight = _WEIGHT_CACHE.get(key)
    if weight is None:
        _WEIGHT_CACHE.clear()
        generator = torch.Generator()
        generator.manual_seed(42)
        weight = torch.randn(
            (num_groups, out_features, in_features),
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(in_features)
        weight = weight.to(device=device, dtype=dtype)
        _WEIGHT_CACHE[key] = weight
    return weight


@triton.jit
def _grouped_gemm_kernel(
    lhs_ptr,  # [rows, K]
    w_ptr,  # [G, N, K]
    mi_ptr,  # [rows] int32
    out_ptr,  # [rows, N]
    n_nblk,
    total_units,
    K: tl.constexpr,
    N: tl.constexpr,
    nprog: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N

    for u in range(pid, total_units, nprog):
        r = u // n_nblk
        nb = u % n_nblk
        n0 = nb * BLOCK_N
        m = tl.load(mi_ptr + r)

        x_base = lhs_ptr + r * K
        w_base = w_ptr + m * (N * K) + n0 * K

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + offs_k) < K
            x = tl.load(x_base + k0 + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            w_offs = (n0 + offs_n)[:, None] * K + (k0 + offs_k)[None, :]
            w_mask = (n0 + offs_n < N)[:, None] & k_mask[None, :]
            w = tl.load(w_base + w_offs, mask=w_mask, other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        out_offs = r * N + n0 + offs_n
        tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


def _run_grouped_gemm(lhs, m_indices, weight, out, rows, N, K, device_index):
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 128:
        BLOCK_N = 128
    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K > 64:
        BLOCK_K = 64
    n_nblk = triton.cdiv(N, BLOCK_N)
    total_units = rows * n_nblk
    num_cores = _get_num_cores(device_index)
    nprog = total_units if total_units < num_cores else num_cores

    _grouped_gemm_kernel[(nprog,)](
        lhs,
        weight,
        m_indices,
        out,
        n_nblk,
        total_units,
        K=K,
        N=N,
        nprog=nprog,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, lhs, m_indices, num_groups, out_features):
        rows, in_features = lhs.shape
        weight = _build_weight(lhs.device, lhs.dtype, num_groups, out_features, in_features)
        out = torch.empty((rows, out_features), device=lhs.device, dtype=lhs.dtype)
        _run_grouped_gemm(
            lhs.contiguous(),
            m_indices.contiguous(),
            weight,
            out,
            rows,
            out_features,
            in_features,
            0,
        )
        return out
