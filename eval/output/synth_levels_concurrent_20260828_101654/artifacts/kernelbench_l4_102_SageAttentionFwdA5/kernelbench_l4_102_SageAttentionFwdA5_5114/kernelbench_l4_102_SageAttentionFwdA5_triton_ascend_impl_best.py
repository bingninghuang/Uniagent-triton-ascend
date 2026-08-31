import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:  # pragma: no cover
    torch_npu = None

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@triton.jit
def _sage_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    B, H, Hkv, S,
    stride_qb, stride_qh,
    stride_kb, stride_kh,
    stride_vb, stride_vh,
    stride_ob, stride_oh,
    stride_lb, stride_lh,
    N_QBLKS, N_KBLKS, N_BLOCKS, NUM_CORES,
    stride_qs, stride_ks, stride_vs,
    sm_scale,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    R: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453
    scale = sm_scale.to(tl.float32) * LOG2E

    pid = tl.program_id(0)
    d_offs = tl.arange(0, D)
    for block_idx in range(pid, N_BLOCKS, NUM_CORES):
        bh = block_idx // N_QBLKS
        qb = block_idx - bh * N_QBLKS
        b = bh // H
        h = bh - b * H
        hkv = h // R

        q0 = qb * BM
        q_rows = q0 + tl.arange(0, BM)
        row_mask = q_rows < S
        row_off2d = q_rows[:, None]

        q_base = q_ptr + b * stride_qb + h * stride_qh
        qk = tl.load(q_base + row_off2d * stride_qs + d_offs[None, :],
                     mask=row_mask[:, None], other=0.0)
        if IS_BF16:
            qk = qk.to(tl.float16)

        acc = tl.zeros((BM, D), dtype=tl.float32)
        m_i = tl.full((BM,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)

        k_base = k_ptr + b * stride_kb + hkv * stride_kh
        v_base = v_ptr + b * stride_vb + hkv * stride_vh

        # Fully valid key blocks: for causal, columns [0, q0);
        # for non-causal, all blocks covering [0, S).
        n_full = q0 // BK if IS_CAUSAL else N_KBLKS
        for kb in range(0, n_full):
            kcols = kb * BK + tl.arange(0, BK)
            kmask = (kcols < S)[:, None]
            k_tile = tl.load(k_base + kcols[:, None] * stride_ks + d_offs[None, :],
                             mask=kmask, other=0.0)
            kt = tl.trans(k_tile)
            if IS_BF16:
                kt = kt.to(tl.float16)
            s = tl.dot(qk, kt) * scale
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp2(s - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            v_tile = tl.load(v_base + kcols[:, None] * stride_vs + d_offs[None, :],
                             mask=kmask, other=0.0)
            if IS_BF16:
                v_tile = v_tile.to(tl.float16)
            acc = acc + tl.dot(p.to(tl.float16), v_tile)
            m_i = m_new

        if IS_CAUSAL:
            n_part = (q0 + BM + BK - 1) // BK
            for kb in range(n_full, n_part):
                kcols = kb * BK + tl.arange(0, BK)
                cmask = (kcols < S)[:, None]
                col_mask = cmask & (kcols[None, :] <= q_rows[:, None])
                k_tile = tl.load(k_base + kcols[:, None] * stride_ks + d_offs[None, :],
                                 mask=cmask, other=0.0)
                kt = tl.trans(k_tile)
                if IS_BF16:
                    kt = kt.to(tl.float16)
                s = tl.dot(qk, kt) * scale
                s = tl.where(col_mask, s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                p = tl.exp2(s - m_new[:, None])
                alpha = tl.exp2(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                v_tile = tl.load(v_base + kcols[:, None] * stride_vs + d_offs[None, :],
                                 mask=cmask, other=0.0)
                if IS_BF16:
                    v_tile = v_tile.to(tl.float16)
                acc = acc + tl.dot(p.to(tl.float16), v_tile)
                m_i = m_new

        l_inv = 1.0 / l_i
        o_tile = (acc * l_inv[:, None]).to(o_ptr.dtype.element_ty)
        o_base = o_ptr + b * stride_ob + h * stride_oh
        tl.store(o_base + row_off2d * stride_qs + d_offs[None, :], o_tile,
                 mask=row_mask[:, None])
        lse_val = (m_i + tl.math.log2(l_i)) * LN2
        lse_base = lse_ptr + b * stride_lb + h * stride_lh
        tl.store(lse_base + q_rows, lse_val, mask=row_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cube_cores = 24
        try:
            self._cube_cores = int(
                torch_npu.npu.npu_config.get_device_limit(0).get(
                    "cube_core_num", 24))
        except Exception:
            self._cube_cores = 24

    def forward(self, q, k, v, tensor_layout, is_causal, sm_scale, return_lse):
        is_causal = int(is_causal)
        sm_scale = float(sm_scale)
        return_lse = int(return_lse)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if tensor_layout == "NHD":
            B, S, H, D = q.shape
            Hkv, Sk, Dk = k.shape[1], k.shape[2], k.shape[3]
            R = H // Hkv
            stride_qb, stride_qh, stride_qs = S * H * D, D, H * D
            stride_kb, stride_kh, stride_ks = Sk * Hkv * D, D, Hkv * D
            stride_vb, stride_vh, stride_vs = stride_kb, stride_kh, stride_ks
            stride_ob, stride_oh = stride_qb, stride_qh
            stride_lb, stride_lh, stride_ls = S * H, 1, H
        else:
            B, H, S, D = q.shape
            Hkv, Sk, Dk = k.shape[1], k.shape[2], k.shape[3]
            R = H // Hkv
            stride_qb, stride_qh, stride_qs = H * S * D, S * D, D
            stride_kb, stride_kh, stride_ks = Hkv * Sk * D, Sk * D, D
            stride_vb, stride_vh, stride_vs = stride_kb, stride_kh, stride_ks
            stride_ob, stride_oh = stride_qb, stride_qh
            stride_lb, stride_lh, stride_ls = H * S, S, 1

        o = torch.empty_like(q)
        if tensor_layout == "NHD":
            lse = torch.empty(B, S, H, dtype=torch.float32, device=q.device)
        else:
            lse = torch.empty(B, H, S, dtype=torch.float32, device=q.device)

        BM, BK = 64, 64
        N_QBLKS = triton.cdiv(S, BM)
        N_KBLKS = triton.cdiv(S, BK)
        N_BLOCKS = B * H * N_QBLKS
        grid_size = N_BLOCKS if N_BLOCKS < self._cube_cores else self._cube_cores

        _sage_fwd_kernel[(grid_size,)](
            q, k, v, o, lse,
            B, H, Hkv, S,
            stride_qb, stride_qh,
            stride_kb, stride_kh,
            stride_vb, stride_vh,
            stride_ob, stride_oh,
            stride_lb, stride_lh,
            N_QBLKS, N_KBLKS, N_BLOCKS, grid_size,
            stride_qs, stride_ks, stride_vs,
            sm_scale,
            D=D, BM=BM, BK=BK, R=R,
            IS_CAUSAL=is_causal == 1,
            IS_BF16=(q.dtype == torch.bfloat16),
        )

        if return_lse:
            return o, lse
        return o
