import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:
    torch_npu = None


# ---------------------------------------------------------------------------
# Kernel 1: global average pooling  x (b, c, h, w) -> ych (b, c)
# ---------------------------------------------------------------------------
@triton.jit
def gpool_kernel(x_ptr, y_ptr, total_rows, hw, inv_hw,
                 ROWS_PER_PROG: tl.constexpr, BLOCK_HW: tl.constexpr,
                 num_pids):
    pid = tl.program_id(0).to(tl.int32)
    offs = tl.arange(0, BLOCK_HW).to(tl.int32)
    mask = offs < hw
    for r0 in range(pid * ROWS_PER_PROG, total_rows, ROWS_PER_PROG * num_pids):
        for i in tl.static_range(ROWS_PER_PROG):
            row = r0 + i
            if row < total_rows:
                base = row * hw
                vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
                s = tl.sum(vals)
                tl.store(y_ptr + row, (s * inv_hw).to(y_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Kernel 2: channel-branch MLP (Linear1+ReLU, then Linear2 as separate kernels
#   to force intermediate materialization in the input dtype)
#   ych (b, c) --W1(r, c), b1(r)--> h1 (b, r) --W2(c, r), b2(c)--> ych2 (b, c)
# ---------------------------------------------------------------------------
@triton.jit
def mlp1_kernel(ych_ptr, w1_ptr, b1_ptr, h1_ptr, B, C, R,
                BM: tl.constexpr, BN1: tl.constexpr, BK1: tl.constexpr):
    m = tl.arange(0, BM).to(tl.int32)
    m_mask = m < B
    n1 = tl.arange(0, BN1).to(tl.int32)
    acc = tl.zeros((BM, BN1), dtype=tl.float32)
    for k in range(0, C, BK1):
        koff = k + tl.arange(0, BK1).to(tl.int32)
        a = tl.load(ych_ptr + m[:, None] * C + koff[None, :],
                    mask=m_mask[:, None] & (koff[None, :] < C), other=0.0)
        w = tl.load(w1_ptr + n1[None, :] * C + koff[:, None],
                    mask=(koff[:, None] < C) & (n1[None, :] < R), other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32)
    b1v = tl.load(b1_ptr + n1, mask=n1 < R, other=0.0).to(tl.float32)
    h1 = (acc + b1v[None, :]).to(h1_ptr.dtype.element_ty)
    h1 = tl.maximum(h1, 0.0)
    tl.store(h1_ptr + m[:, None] * R + n1[None, :], h1,
             mask=m_mask[:, None] & (n1[None, :] < R))


# ---------------------------------------------------------------------------
# Kernel 2b: Linear2:  h1 (b, r) -> ych2 (b, c);  W2 (c, r) row-major
# ---------------------------------------------------------------------------
@triton.jit
def mlp2_kernel(h1_ptr, w2_ptr, b2_ptr, y2_ptr, B, C, R,
                BM: tl.constexpr, BK2: tl.constexpr, BN2: tl.constexpr):
    cb = tl.program_id(0).to(tl.int32)
    m = tl.arange(0, BM).to(tl.int32)
    m_mask = m < B
    n2 = cb * BN2 + tl.arange(0, BN2).to(tl.int32)
    n2_mask = n2 < C
    acc = tl.zeros((BM, BN2), dtype=tl.float32)
    for k in range(0, R, BK2):
        koff = k + tl.arange(0, BK2).to(tl.int32)
        a = tl.load(h1_ptr + m[:, None] * R + koff[None, :],
                    mask=m_mask[:, None] & (koff[None, :] < R), other=0.0)
        w = tl.load(w2_ptr + n2[None, :] * R + koff[:, None],
                    mask=(koff[:, None] < R) & n2_mask[None, :], other=0.0)
        acc = tl.dot(a, w, acc, out_dtype=tl.float32)
    b2v = tl.load(b2_ptr + n2, mask=n2_mask, other=0.0).to(tl.float32)
    y2 = (acc + b2v[None, :]).to(y2_ptr.dtype.element_ty)
    tl.store(y2_ptr + m[:, None] * C + n2[None, :], y2,


# ---------------------------------------------------------------------------
# Kernel 3: 1x1 convolution as GEMM
#   x (b, C, HW) [channels-first], w (R, C) row-major, bias (R,)
#   -> o (b, R, HW)
# ---------------------------------------------------------------------------
@triton.jit
def conv1x1_kernel(x_ptr, w_ptr, b_ptr, o_ptr, C, R, HW, total, num_pids,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    nwt = tl.cdiv(HW, BN)
    m = tl.arange(0, BM).to(tl.int32)
    m_mask = m < R
    for t in range(tl.program_id(0).to(tl.int32), total, num_pids):
        bidx = t // nwt
        wt = t % nwt
        p0 = wt * BN
        po = p0 + tl.arange(0, BN).to(tl.int32)
        po_mask = po < HW
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, C, BK):
            koff = k + tl.arange(0, BK).to(tl.int32)
            k_mask = koff < C
            a = tl.load(w_ptr + m[:, None] * C + koff[None, :],
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            bld = tl.load(x_ptr + bidx * C * HW + koff[:, None] * HW + po[None, :],
                          mask=k_mask[:, None] & po_mask[None, :], other=0.0)
            acc = tl.dot(a, bld, acc, out_dtype=tl.float32)
        bv = tl.load(b_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
        out = (acc + bv[:, None]).to(o_ptr.dtype.element_ty)
        tl.store(o_ptr + bidx * R * HW + m[:, None] * HW + po[None, :], out,
                 mask=m_mask[:, None] & po_mask[None, :])
             mask=m_mask[:, None] & n2_mask[None, :])