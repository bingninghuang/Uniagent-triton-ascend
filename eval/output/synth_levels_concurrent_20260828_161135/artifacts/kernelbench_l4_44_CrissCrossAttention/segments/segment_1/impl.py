import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: fused Q/K/V projection (three 1x1 convs as one GEMM)
#   x:     [b, c, h, w] (NCHW contiguous)
#   W_all: [N, c]  where N = 2*c8 + c  (rows: q(c8), k(c8), v(c)) contiguous
#   bias:  [N]  contiguous
#   proj:  [b, h, w, N]  (NHWC, per-pixel channel vector: q | k | v)
# Compute per pixel m = i*w+j : proj[b, m, n] = sum_ch W_all[n, ch]*x[b, ch, m] + bias[n]
# ---------------------------------------------------------------------------
@triton.jit
def _proj_kernel(
    x_ptr, w_ptr, bias_ptr, proj_ptr,
    hw, c, N, n_nblk,
    M,  # h*w pixels per batch
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # pixel idx m = i*w+j
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output channel n
    offs_k = tl.arange(0, BLOCK_K)                     # input channel chunk

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < c

    base_x = x_ptr + b.to(tl.int32) * (c * hw)
    # tile_x[k, m] = x[b, k, m]
    x_ptrs = base_x + offs_k[:, None].to(tl.int32) * hw + offs_m[None, :].to(tl.int32)
    # tile_w[n, k] = W_all[n, k]
    w_ptrs = w_ptr + offs_n[:, None] * c + offs_k[None, :]

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(0, c, BLOCK_K):
        tile_x = tl.load(x_ptrs, mask=(mask_k[:, None] & mask_m[None, :]), other=0.0)
        tile_w = tl.load(w_ptrs, mask=(mask_n[:, None] & mask_k[None, :]), other=0.0)
        acc = tl.dot(tile_w, tile_x, acc)
        x_ptrs += BLOCK_K * hw
        w_ptrs += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias[:, None]

    # store proj[b, m, n] -> offset b*M*N + m*N + n
    out_ptrs = proj_ptr + b.to(tl.int32) * (M * N) + offs_m[None, :].to(tl.int32) * N + offs_n[:, None]
    tl.store(out_ptrs, acc, mask=(mask_m[None, :] & mask_n[:, None]))


# ---------------------------------------------------------------------------
# Kernel 2: criss-cross attention energies
# For query pixel (i, j):
#   E_H[ii'] = sum_ch q[b,i,j,ch] * k[b,ii',j,ch]   (mask ii'==i -> -inf)
#   E_W[jj'] = sum_ch q[b,i,j,ch] * k[b,i,jj',ch]
#   eH buffer: [b, h, w, h] , eW buffer: [b, h, w, w]  (fp32)
# ---------------------------------------------------------------------------
@triton.jit
def _energy_kernel(
    proj_ptr, eH_ptr, eW_ptr,
    hw, w, h, c8, N,
    BC8,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    j = tl.program_id(0)
    i = tl.program_id(1)
    b = tl.program_id(2)

    offs_h = tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, BLOCK_W)
    offs_c8 = tl.arange(0, BC8)

    mask_h = offs_h < h
    mask_w = offs_w < w
    mask_c8 = offs_c8 < c8

    base = proj_ptr + b.to(tl.int32) * (hw * N)

    # q vector at pixel (i, j): proj[b, i*w+j, 0:c8]
    q = tl.load(base + (i * w + j) * N + offs_c8, mask=mask_c8, other=0.0).to(tl.float32)

    # k column: pixels (ii, j), channel offset c8
    kcol_ptrs = base + (offs_h[:, None] * w + j).to(tl.int32) * N + (c8 + offs_c8[None, :])
    kcol = tl.load(kcol_ptrs, mask=(mask_h[:, None] & mask_c8[None, :]), other=0.0).to(tl.float32)
    eH = tl.sum(q[None, :] * kcol, axis=1)
    inf_val = float("-inf")
    eH = tl.where(offs_h == i, inf_val, eH)

    # k row: pixels (i, jj), channel offset c8
    krow_ptrs = base + (i * w + offs_w[:, None]).to(tl.int32) * N + (c8 + offs_c8[None, :])
    krow = tl.load(krow_ptrs, mask=(mask_w[:, None] & mask_c8[None, :]), other=0.0).to(tl.float32)
    eW = tl.sum(q[None, :] * krow, axis=1)

    # store eH[b, i, j, ii'] and eW[b, i, j, jj']  (buffers [b,h,w,h] / [b,h,w,w])
    eH_ptrs = eH_ptr + ((b * h + i) * w + j) * h + offs_h
    tl.store(eH_ptrs, eH, mask=mask_h)
    eW_ptrs = eW_ptr + ((b * h + i) * w + j) * w + offs_w
    tl.store(eW_ptrs, eW, mask=mask_w)


# ---------------------------------------------------------------------------
# Kernel 3: softmax over concat(E_H, E_W) + value aggregation + residual
# Grid: (c_chunks, pixels(m), b) handled row/col-wise per (i, j, cblock)
#   out = gamma * (out_H + out_W) + x
# out_H[ch] = sum_{ii'} S_H[ii'] * v[b, ii', j, ch]
# out_W[ch] = sum_{jj'} S_W[jj'] * v[b, i, jj', ch]
# ---------------------------------------------------------------------------
@triton.jit
def _att_out_kernel(
    proj_ptr, eH_ptr, eW_ptr, x_ptr, out_ptr, gamma_ptr,
    hw, w, h, c, c8, N,
    BC,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = tl.program_id(2)

    i = pid_m // w
    j = pid_m % w

    offs_h = tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, BLOCK_W)
    offs_c = pid_c * BC + tl.arange(0, BC)

    mask_h = offs_h < h
    mask_w = offs_w < w
    mask_c = offs_c < c

    base = proj_ptr + b.to(tl.int32) * (hw * N)

    # energies (fp32)
    eH = tl.load(eH_ptr + ((b * h + i) * w + j) * h + offs_h, mask=mask_h, other=float("-inf"))
    eW = tl.load(eW_ptr + ((b * h + i) * w + j) * w + offs_w, mask=mask_w, other=float("-inf"))

    mH = tl.max(eH, axis=0)
    mW = tl.max(eW, axis=0)
    m_all = tl.maximum(mH, mW)
    # protect the all-(-inf) line (cannot happen here since w>=2 has unmasked entries)
    m_all = tl.where(tl.isinf(m_all) & (m_all < 0), 0.0, m_all)

    eH -= m_all
    eW -= m_all
    pE = tl.exp(eH)
    pW = tl.exp(eW)
    z = tl.sum(pE, axis=0) + tl.sum(pW, axis=0)
    z = tl.where(z == 0.0, 1.0, z)
    attH = pE / z
    attW = pW / z

    # v column: pixels (ii, j), channel offset 2*c8
    vcol_ptrs = base + (offs_h[:, None] * w + j).to(tl.int32) * N + (2 * c8 + offs_c[None, :])
    vcol = tl.load(vcol_ptrs, mask=(mask_h[:, None] & mask_c[None, :]), other=0.0).to(tl.float32)
    # v row: pixels (i, jj), channel offset 2*c8
    vrow_ptrs = base + (i * w + offs_w[:, None]).to(tl.int32) * N + (2 * c8 + offs_c[None, :])
    vrow = tl.load(vrow_ptrs, mask=(mask_w[:, None] & mask_c[None, :]), other=0.0).to(tl.float32)

    out_H = tl.sum(attH[:, None] * vcol, axis=0)
    out_W = tl.sum(attW[:, None] * vrow, axis=0)

    gamma = tl.load(gamma_ptr).to(tl.float32)

    # x[b, ch, i, j]: 1D load over channels at pixel (i, j)
    x_ptrs = x_ptr + b.to(tl.int32) * (c * hw) + offs_c.to(tl.int32) * hw + (i * w + j)
    xv = tl.load(x_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    out = gamma * (out_H + out_W) + xv
    out_ptr_loc = out_ptr + b.to(tl.int32) * (c * hw) + offs_c.to(tl.int32) * hw + (i * w + j)
    tl.store(out_ptr_loc, out, mask=mask_c)


# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}
        try:
            import torch_npu
            devlimit = torch_npu.npu.npu_config.get_device_limit(0)
            self._VEC_CORE_NUM = int(devlimit.get('vector_core_num', 48))
            self._CUBE_CORE_NUM = int(devlimit.get('cube_core_num', 24))
        except Exception:
            self._VEC_CORE_NUM = 48
            self._CUBE_CORE_NUM = 24

    def _get_weights(self, c, device, dtype):
        key = (c, device, dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            self._cache[key] = (
                nn.Conv2d(c, c // 8, 1).to(device=device, dtype=dtype),
                nn.Conv2d(c, c // 8, 1).to(device=device, dtype=dtype),
                nn.Conv2d(c, c, 1).to(device=device, dtype=dtype),
                nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
            )
            torch.set_rng_state(rng_state)
        return self._cache[key]

    def forward(self, x):
        x = x.contiguous()
        b, c, h, w = x.shape
        hw = h * w
        M = hw
        c8 = c // 8
        N = 2 * c8 + c

        conv_q, conv_k, conv_v, gamma = self._get_weights(c, x.device, x.dtype)

        # W_all [N, c] = [Wq; Wk; Wv] with 1x1 conv weights [out, in, 1, 1] -> [out, in]
        wq = conv_q.weight.view(conv_q.weight.shape[0], conv_q.weight.shape[1])
        wk = conv_k.weight.view(conv_k.weight.shape[0], conv_k.weight.shape[1])
        wv = conv_v.weight.view(conv_v.weight.shape[0], conv_v.weight.shape[1])
        w_all = torch.cat([wq, wk, wv], dim=0).contiguous()
        bias_all = torch.cat([conv_q.bias, conv_k.bias, conv_v.bias], dim=0).contiguous()

        proj = torch.empty((b, h, w, N), device=x.device, dtype=x.dtype)
        eH = torch.empty((b, h, w, h), device=x.device, dtype=torch.float32)
        eW = torch.empty((b, h, w, w), device=x.device, dtype=torch.float32)
        out = torch.empty_like(x)

        # ---- kernel 1: projection ----
        BLOCK_K = 64
        BLOCK_M = 32
        BLOCK_N = 64
        grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), b)
        _proj_kernel[grid1](
            x, w_all, bias_all, proj,
            hw, c, N, triton.cdiv(N, BLOCK_N),
            M,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # ---- kernel 2: energies ----
        BC8 = max(16, triton.next_power_of_2(c8))
        BLOCK_H = max(16, triton.next_power_of_2(h))
        BLOCK_W = max(16, triton.next_power_of_2(w))
        grid2 = (w, h, b)
        _energy_kernel[grid2](
            proj, eH, eW,
            hw, w, h, c8, N, BC8,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )

        # ---- kernel 3: softmax + values + residual ----
        BC = 32
        grid3 = (triton.cdiv(c, BC), M, b)
        _att_out_kernel[grid3](
            proj, eH, eW, x, out, gamma,
            hw, w, h, c, c8, N, BC,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )

        return out
