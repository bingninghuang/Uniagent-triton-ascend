import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:  # pragma: no cover
    torch_npu = None


# ---------------------------------------------------------------------------
# Generic batched GEMM on Ascend: for batch l in [0, L):
#   out[l, 0:RM, 0:NM] = W0[0:RM, 0:K] @ X[l, 0:K, 0:NM]
# W0 is shared across batches (row-major (RM, K) with strides (s_w0, s_w1)).
# X and out are laid out as (L, K, NM) / (L, RM, NM) with batch stride s_lb.
# ---------------------------------------------------------------------------
@triton.jit
def gemm_kernel(
    w_ptr, x_ptr, out_ptr,
    L, RM, NM, K,
    s_w0, s_w1,
    s_x0, s_x1,
    s_o0, s_o1,
    s_lb,
    CUBE_CORE_NUM: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    offs_r = tl.arange(0, BR)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    num_rm = tl.cdiv(RM, BR)
    num_nm = tl.cdiv(NM, BN)
    blocks_per_batch = num_rm * num_nm
    num_blocks = L * blocks_per_batch

    o_r = tl.arange(0, BR)
    o_n = tl.arange(0, BN)

    for block_idx in range(tl.program_id(0), num_blocks, CUBE_CORE_NUM):
        li = block_idx // blocks_per_batch
        rem = block_idx - li * blocks_per_batch
        ri = rem // num_nm
        ni = rem - ri * num_nm

        r0 = ri * BR
        n0 = ni * BN
        r_offs = r0 + o_r
        n_offs = n0 + o_n

        w_ptrs = w_ptr + r_offs[:, None] * s_w0 + offs_k[None, :] * s_w1
        x_ptrs = x_ptr + li * s_lb + offs_k[:, None] * s_x0 + n_offs[None, :] * s_x1

        acc = tl.zeros((BR, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k_offs = k0 + offs_k
            k_ok = k_offs < K
            a = tl.load(w_ptrs, mask=(r_offs[:, None] < RM) & k_ok[None, :], other=0.0)
            bt = tl.load(x_ptrs, mask=k_ok[:, None] & (n_offs[None, :] < NM), other=0.0)
            acc = tl.dot(a, bt, acc, out_dtype=tl.float32)
            w_ptrs += BK * s_w1
            x_ptrs += BK * s_x0

        o_ptrs = out_ptr + li * s_lb + r_offs[:, None] * s_o0 + n_offs[None, :] * s_o1
        tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty),
                 mask=(r_offs[:, None] < RM) & (n_offs[None, :] < NM))


# ---------------------------------------------------------------------------
# Row-wise softmax over the last (M) dimension of a contiguous (rows, M)
# tensor. One program per row; two streaming passes (online max/sum, then
# normalize) so that arbitrary M fits in the UB.
# ---------------------------------------------------------------------------
@triton.jit
def softmax_row_kernel(
    src_ptr, dst_ptr,
    M,
    VEC_CORE_NUM: tl.constexpr,
    BM: tl.constexpr,
):
    r0 = tl.program_id(0)
    offs = tl.arange(0, BM)

    m_i = -3.4e38
    l_i = 0.0
    for m0 in range(0, M, BM):
        idx = m0 + offs
        x = tl.load(src_ptr + r0 * M + idx, mask=idx < M, other=-3.4e38).to(tl.float32)
        mx = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, mx)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    inv_l = 1.0 / l_i
    for m0 in range(0, M, BM):
        idx = m0 + offs
        x = tl.load(src_ptr + r0 * M + idx, mask=idx < M, other=-3.4e38).to(tl.float32)
        y = tl.exp(x - m_i) * inv_l
        tl.store(dst_ptr + r0 * M + idx, y.to(dst_ptr.dtype.element_ty), mask=idx < M)


def _pick_gemm_blocks(rm, nm, k, cube_num):
    # (BR, BN, BK): keep C-tile fp32 <= 64KB and honor 256B alignment for fp16
    # rows where possible; bias toward more blocks for better load balance.
    if nm >= 2048 and rm >= 128:
        return 128, 128, 64
    if nm >= 512 and rm >= 64:
        return 64, 128, 64
    return 64, 64, 64


class ModelNew(nn.Module):
    """Triton-Ascend implementation of Double Attention (bilinear attention)."""

    def __init__(self):
        super().__init__()
        self._cache = {}
        try:
            limit = torch_npu.npu.npu_config.get_device_limit(0)
            self.VEC_CORE_NUM = limit.get("vector_core_num", 48)
            self.CUBE_CORE_NUM = limit.get("cube_core_num", 24)
        except Exception:
            self.VEC_CORE_NUM = 48
            self.CUBE_CORE_NUM = 24

    def _get_modules(self, c_m, c_n, reconstruct, x):
        in_channels = x.shape[1]
        c = in_channels
        key = (in_channels, c_m, c_n, reconstruct, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            modules = [
                nn.Conv2d(c, c_m, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, c_n, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, c_n, 1).to(device=x.device, dtype=x.dtype)
            ]
            if reconstruct:
                modules.append(nn.Conv2d(c_m, c, 1).to(device=x.device, dtype=x.dtype))
            self._cache[key] = modules
            torch.set_rng_state(rng_state)
        return self._cache[key]

    def _launch_gemm(self, L, w, x, out, rm, k, nm, s_w0, s_w1, s_x0, s_x1,
                     s_o0, s_o1, s_lb):
        br, bn, bk = _pick_gemm_blocks(rm, nm, k, self.CUBE_CORE_NUM)
        gemm_kernel[(self.CUBE_CORE_NUM,)](
            w, x, out,
            L, rm, nm, k,
            s_w0, s_w1,
            s_x0, s_x1,
            s_o0, s_o1,
            s_lb,
            self.CUBE_CORE_NUM,
            BR=br, BN=bn, BK=bk,
        )

    def forward(self, x, a=None, v=None):
        torch.manual_seed(42)
        b, c, h, w = x.shape
        if b == 0 or c == 0 or h == 0 or w == 0:
            return x
        in_channels = c
        c_m = c
        c_n = max(1, c // 8)
        reconstruct = True

        modules = self._get_modules(c_m, c_n, reconstruct, x)
        convA, convB, convV, convR = modules

        M = h * w
        dev = x.device
        dt = x.dtype
        x2 = x.view(b, c, M)

        A = torch.empty((b, c, M), device=dev, dtype=dt)
        B = torch.empty((b, c_n, M), device=dev, dtype=dt)
        V = torch.empty((b, c_n, M), device=dev, dtype=dt)
        G = torch.empty((b, c, c_n), device=dev, dtype=dt)
        H = torch.empty((b, c, c_n), device=dev, dtype=dt)
        out = torch.empty((b, c, M), device=dev, dtype=dt)

        # conv weights are (out, in, 1, 1) row-major -> row strides (in, 1)
        wA = convA.weight
        wB = convB.weight
        wV = convV.weight
        wR = convR.weight

        # 1) raw projections: A = W_A x, B = W_B x, V = W_V x
        self._launch_gemm(b, wA, x2, A, c_m, c, M, c, 1, M, 1, M, 1, c * M)
        self._launch_gemm(b, wB, x2, B, c_n, c, M, c, 1, M, 1, M, 1, c * M)
        self._launch_gemm(b, wV, x2, V, c_n, c, M, c, 1, M, 1, M, 1, c * M)

        # 2) softmax along the spatial (M) dimension, in float32 internally
        bm = min(triton.next_power_of_2(max(M, 16)), 8192)
        sm_grid = (b * c_n,)
        softmax_row_kernel[sm_grid](B, B, M, self.VEC_CORE_NUM, BM=bm)
        softmax_row_kernel[sm_grid](V, V, M, self.VEC_CORE_NUM, BM=bm)

        # 3) G = A @ B^T   (b, c, M) x (b, M, c_n) -> (b, c, c_n)
        self._launch_gemm(b, A, B, G, c_m, M, c_n, M, 1, 1, M, c_n, 1, c * c_n)

        # 4) H = W_R @ G   (b, c, c) x (b, c, c_n) -> (b, c, c_n)
        self._launch_gemm(b, wR, G, H, c, c, c_n, c, 1, c_n, 1, c_n, 1, c * c_n)

        # 5) out = H @ V   (b, c, c_n) x (b, c_n, M) -> (b, c, M)
        self._launch_gemm(b, H, V, out, c, c_n, M, c_n, 1, M, 1, M, 1, c * M)

        return out.view(b, c, h, w)
