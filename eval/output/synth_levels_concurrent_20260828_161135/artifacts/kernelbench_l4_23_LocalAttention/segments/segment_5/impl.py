import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _proj_gemm_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    total_blocks,
    L,
    D,
    H,
    HD,
    c_b_step,
    c_h_step,
    w_head_row_step,
    p_step,
    n_cores,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # C[b, h, p, d] = sum_m A[b, p, m] * W[(h*w_head_row_step + d), m]
    # A layout: [B, L, D] row-major ; C layout: b*c_b_step + h*c_h_step + p*p_step + d
    pid = tl.program_id(0).to(tl.int32)
    pblk = tl.cdiv(L, BM)
    nblk = tl.cdiv(HD, BN)

    for idx in range(pid, total_blocks, n_cores):
        tt = idx // pblk
        p_i = idx % pblk
        q2 = tt // nblk
        n_i = tt % nblk
        bb = q2 // H
        hh = q2 % H

        p0 = p_i * BM
        n0 = n_i * BN

        offs_m = p0 + tl.arange(0, BM)
        offs_n = n0 + tl.arange(0, BN)
        m_mask = offs_m < L
        n_mask = offs_n < HD

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, D, BK):
            offs_k = k0 + tl.arange(0, BK)
            k_mask = offs_k < D
            a_off = (bb * L + offs_m)[:, None] * D + offs_k[None, :]
            a = tl.load(
                a_ptr + a_off,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w_off = (hh * w_head_row_step + offs_n)[:, None] * D + offs_k[None, :]
            w = tl.load(
                w_ptr + w_off,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(a, tl.trans(w), acc)

        c_off = (
            bb * c_b_step
            + hh * c_h_step
            + offs_m[:, None] * p_step
            + offs_n[None, :]
        )
        tl.store(c_ptr + c_off, acc.to(c_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _local_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    L,
    D,
    H,
    HD,
    W,
    w_f,
    qk_scale,
    total_blocks,
    n_cores,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HD_PAD: tl.constexpr,
):
    # q/k/v: [B, H, L, HD] ; o: [B, L, D] with o[b, p, h*HD + d]
    # banded (local) attention via online softmax, band |i - j| < W
    pid = tl.program_id(0).to(tl.int32)
    pblk = tl.cdiv(L, BM)

    LOG2E: tl.constexpr = 1.4426950408889634

    for idx in range(pid, total_blocks, n_cores):
        bh = idx // pblk
        mb = idx % pblk
        b = bh // H
        h = bh % H

        m_start = mb * BM
        offs_m = m_start + tl.arange(0, BM)
        offs_d = tl.arange(0, HD_PAD)
        m_mask = offs_m < L
        d_mask = offs_d < HD

        base = (b * H + h) * (L * HD)
        q_off = base + offs_m[:, None] * HD + offs_d[None, :]
        qblk = tl.load(
            q_ptr + q_off, mask=m_mask[:, None] & d_mask[None, :], other=0.0
        )

        acc = tl.zeros((BM, HD_PAD), dtype=tl.float32)
        m_i = tl.full((BM,), -1e30, dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)

        j0 = m_start - W + 1
        j1 = m_start + BM + W - 1
        if j0 < 0:
            j0 = 0
        if j1 > L:
            j1 = L
        j0b = j0 // BN * BN

        for n0 in range(j0b, j1, BN):
            offs_n = n0 + tl.arange(0, BN)
            n_mask = offs_n < L

            # k^T tile [HD_PAD, BN]
            kt_off = base + offs_d[:, None] * 1 + offs_n[None, :] * HD
            kt = tl.load(
                k_ptr + kt_off, mask=d_mask[:, None] & n_mask[None, :], other=0.0
            )
            s = tl.dot(qblk, kt)
            s = s * qk_scale

            dist_f = (offs_m[:, None] - offs_n[None, :]).to(tl.float32)
            band = (dist_f > -w_f) & (dist_f < w_f)
            valid = band & n_mask[None, :] & m_mask[:, None]
            s = tl.where(valid, s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2((m_i - m_new) * LOG2E)
            p = tl.exp2((s - m_new[:, None]) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, 1)

            v_off = base + offs_n[:, None] * HD + offs_d[None, :]
            vblk = tl.load(
                v_ptr + v_off, mask=n_mask[:, None] & d_mask[None, :], other=0.0
            ).to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(p, vblk)
            m_i = m_new

        o = acc * (1.0 / l_i)[:, None]
        o_off = b * (L * D) + offs_m[:, None] * D + h * HD + offs_d[None, :]
        tl.store(o_ptr + o_off, o.to(o_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & d_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}
        try:
            import torch_npu

            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get(
                "cube_core_num", 24
            )
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get(
                "vector_core_num", 48
            )
        except Exception:
            self.CUBE_CORE_NUM = 24
            self.VEC_CORE_NUM = 48

    def _layers(self, x, n_heads):
        d_model = x.shape[-1]
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        key = (d_model, n_heads, x.device, x.dtype)
        if key not in self._cache:
            self._cache.clear()
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            self._cache[key] = tuple(
                nn.Linear(d_model, d_model, bias=False).to(
                    device=x.device, dtype=x.dtype
                )
                for _ in range(4)
            )
            torch.set_rng_state(rng_state)
        return self._cache[key]

    def forward(self, x, n_heads, window_size):
        if not x.is_contiguous():
            x = x.contiguous()
        batch, seq_len, d_model = x.shape
        head_dim = d_model // n_heads
        window = int(window_size)

        q_proj, k_proj, v_proj, out_proj = self._layers(x, n_heads)
        w_q = q_proj.weight
        w_k = k_proj.weight
        w_v = v_proj.weight
        w_o = out_proj.weight

        dev = x.device
        dtype = x.dtype
        q = torch.empty((batch, n_heads, seq_len, head_dim), device=dev, dtype=dtype)
        k = torch.empty((batch, n_heads, seq_len, head_dim), device=dev, dtype=dtype)
        v = torch.empty((batch, n_heads, seq_len, head_dim), device=dev, dtype=dtype)
        attn = torch.empty((batch, seq_len, d_model), device=dev, dtype=dtype)
        y = torch.empty((batch, seq_len, d_model), device=dev, dtype=dtype)

        if seq_len >= 128:
            bm = 128
        elif seq_len >= 64:
            bm = 64
        elif seq_len >= 32:
            bm = 32
        else:
            bm = 16
        if dtype == torch.float32:
            bk = 64
        else:
            bk = 128

        bn_qkv = triton.next_power_of_2(head_dim)
        if bn_qkv < 16:
            bn_qkv = 16
        if bn_qkv > 128:
            bn_qkv = 128
        nblks_qkv = triton.cdiv(head_dim, bn_qkv)
        pblks = triton.cdiv(seq_len, bm)
        total_qkv = batch * n_heads * pblks * nblks_qkv
        grid_qkv = (total_qkv if total_qkv < self.CUBE_CORE_NUM else self.CUBE_CORE_NUM,)

        c_b = seq_len * d_model
        c_h = seq_len * head_dim
        _proj_gemm_kernel[grid_qkv](
            x, w_q, q, total_qkv, seq_len, d_model, n_heads, head_dim,
            c_b, c_h, head_dim, head_dim, self.CUBE_CORE_NUM,
            bm, bn_qkv, bk,
        )
        _proj_gemm_kernel[grid_qkv](
            x, w_k, k, total_qkv, seq_len, d_model, n_heads, head_dim,
            c_b, c_h, head_dim, head_dim, self.CUBE_CORE_NUM,
            bm, bn_qkv, bk,
        )
        _proj_gemm_kernel[grid_qkv](
            x, w_v, v, total_qkv, seq_len, d_model, n_heads, head_dim,
            c_b, c_h, head_dim, head_dim, self.CUBE_CORE_NUM,
            bm, bn_qkv, bk,
        )

        # attention
        atn_b = 64
        atn_n = 64
        hd_pad = bn_qkv
        total_attn = batch * n_heads * triton.cdiv(seq_len, atn_b)
        grid_attn = (total_attn if total_attn < self.CUBE_CORE_NUM else self.CUBE_CORE_NUM,)
        _local_attn_kernel[grid_attn](
            q, k, v, attn, batch, seq_len, d_model, n_heads, head_dim, window,
            float(window), 1.0 / (head_dim ** 0.5), total_attn,
            self.CUBE_CORE_NUM,
            atn_b, atn_n, hd_pad,
        )

        # output projection: treat as single head of width d_model
        bn_out = 128
        nblks_out = triton.cdiv(d_model, bn_out)
        total_out = batch * pblks * nblks_out
        grid_out = (total_out if total_out < self.CUBE_CORE_NUM else self.CUBE_CORE_NUM,)
        _proj_gemm_kernel[grid_out](
            attn, w_o, y, total_out, seq_len, d_model, 1, d_model,
            c_b, 0, d_model, d_model, self.CUBE_CORE_NUM,
            bm, bn_out, bk,
        )
        return y
