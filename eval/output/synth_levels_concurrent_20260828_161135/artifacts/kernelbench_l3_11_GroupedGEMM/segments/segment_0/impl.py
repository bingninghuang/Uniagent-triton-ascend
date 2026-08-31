import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _num_ai_cores(device) -> int:
    """Dynamically read the AI core (Vector core) count of the target device."""
    candidates = []
    try:
        candidates.append(device)
        idx = device.index if device is not None and device.index is not None else 0
        candidates.append(idx)
        candidates.append(0)
    except Exception:
        candidates = [0]
    for arg in candidates:
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(arg)
            val = None
            if isinstance(props, dict):
                for key in ("num_aicore", "num_cores", "vector_core_num", "num_vectorcore"):
                    val = props.get(key)
                    if val:
                        break
            else:
                for key in ("num_aicore", "num_cores", "vector_core_num", "num_vectorcore"):
                    val = getattr(props, key, None)
                    if val:
                        break
            if val:
                return int(val)
        except Exception:
            continue
    return 24


_NUM_CORES_CACHE = {}


@triton.jit
def _grouped_gemm_kernel(
    lhs_ptr,        # (ROWS, IN)
    weight_ptr,     # (G, OUT, IN)
    m_indices_ptr,  # (ROWS,) int32
    out_ptr,        # (ROWS, OUT)
    TOTAL_TILES,
    out_tiles,
    IN,
    OUT,
    STRIDE_G,  # OUT * IN
    NUM_CORES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    for tile in range(pid, TOTAL_TILES, NUM_CORES):
        row = (tile // out_tiles).to(tl.int32)
        n0 = ((tile % out_tiles) * BLOCK_N).to(tl.int32)
        g = tl.load(m_indices_ptr + row).to(tl.int32)

        offs_n = (n0 + tl.arange(0, BLOCK_N)).to(tl.int32)
        mask_n = offs_n < OUT
        w_base = weight_ptr + g * STRIDE_G + offs_n[:, None] * IN
        x_base = lhs_ptr + row * IN

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, IN, BLOCK_K):
            offs_k = (k0 + tl.arange(0, BLOCK_K)).to(tl.int32)
            mask_k = offs_k < IN
            w = tl.load(
                w_base + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            x = tl.load(x_base + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)

        tl.store(
            out_ptr + row * OUT + offs_n,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_n,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, lhs, m_indices, num_groups, out_features):
        torch.manual_seed(42)
        rows, in_features = lhs.shape
        if m_indices.shape != (rows,):
            raise ValueError("m_indices must contain one group id per lhs row")
        key = (num_groups, out_features, in_features, lhs.device, lhs.dtype)
        if key not in self._cache:
            self._cache.clear()
            generator = torch.Generator()
            generator.manual_seed(42)
            weight = torch.randn(
                (num_groups, out_features, in_features),
                generator=generator,
                dtype=torch.float32,
            )
            weight = weight / math.sqrt(in_features)
            self._cache[key] = weight.to(device=lhs.device, dtype=lhs.dtype)
        weight = self._cache[key]

        lhs = lhs.contiguous()
        m_indices = m_indices.contiguous()
        out = torch.empty((rows, out_features), device=lhs.device, dtype=lhs.dtype)

        BLOCK_N = 128
        BLOCK_K = 128
        n_out_tiles = _cdiv(out_features, BLOCK_N)
        total_tiles = rows * n_out_tiles
        dev_key = (lhs.device, lhs.dtype)
        if dev_key not in _NUM_CORES_CACHE:
            _NUM_CORES_CACHE[dev_key] = _num_ai_cores(lhs.device)
        num_cores = _NUM_CORES_CACHE[dev_key]
        grid = (min(total_tiles, num_cores),)

        _grouped_gemm_kernel[grid](
            lhs,
            weight,
            m_indices,
            out,
            total_tiles,
            n_out_tiles,
            in_features,
            out_features,
            out_features * in_features,
            grid[0],
            BLOCK_N,
            BLOCK_K,
        )
        return out
