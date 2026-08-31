import torch
import triton
import triton.language as tl


@triton.jit
def _dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, lse_ptr, dq_ptr,
    B, H, S_q, S_k,
    stride_qb, stride_qh,
    stride_kb, stride_kh,
    stride_vb, stride_vh,
    stride_dob, stride_doh,
    stride_ob, stride_oh,
    stride_lse_b, stride_lse_h,
    scale,
    rep,
    num_cores,
    D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(S_q, BLOCK_M)
    num_blocks = B * H * num_m_blocks

    for block_idx in range(pid, num_blocks, num_cores):
        bh = block_idx // num_m_blocks
        m_block = block_idx - bh * num_m_blocks
        b = bh // H
        h = bh - b * H
        h_kv = h // rep

        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        row_valid = offs_m < S_q

        q = tl.load(
            q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * D + offs_d[None, :],
            mask=row_valid[:, None], other=0.0).to(tl.float32)
        do = tl.load(
            do_ptr + b * stride_dob + h * stride_doh + offs_m[:, None] * D + offs_d[None, :],
            mask=row_valid[:, None], other=0.0).to(tl.float32)
        o = tl.load(
            o_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * D + offs_d[None, :],
            mask=row_valid[:, None], other=0.0).to(tl.float32)
        lse = tl.load(
            lse_ptr + b * stride_lse_b + h * stride_lse_h + offs_m,
            mask=row_valid, other=0.0).to(tl.float32)
        d_row = tl.sum(do * o, axis=1)

        dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

        if IS_CAUSAL:
            hi = tl.maximum(tl.minimum(S_k, (m_block + 1) * BLOCK_M + S_k - S_q), 0)
        else:
            hi = S_k

        for j0 in range(0, hi, BLOCK_N):
            offs_n = j0 + tl.arange(0, BLOCK_N)
            col_valid = offs_n < S_k
            k_j = tl.load(
                k_ptr + b * stride_kb + h_kv * stride_kh + offs_n[:, None] * D + offs_d[None, :],
                mask=col_valid[:, None], other=0.0).to(tl.float32)
            v_j = tl.load(
                v_ptr + b * stride_vb + h_kv * stride_vh + offs_n[:, None] * D + offs_d[None, :],
                mask=col_valid[:, None], other=0.0).to(tl.float32)

            valid = row_valid[:, None] & col_valid[None, :]
            if IS_CAUSAL:
                valid = valid & (offs_n[None, :] <= offs_m[:, None] + (S_k - S_q))
            s = tl.dot(q, tl.trans(k_j)) * scale
            p = tl.exp(tl.where(valid, s, -float("inf")) - lse[:, None])
            dp = tl.dot(do, tl.trans(v_j))
            ds = p * (dp - d_row[:, None])

            dq_acc = tl.dot(ds, k_j, dq_acc)

        tl.store(
            dq_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * D + offs_d[None, :],
            (dq_acc * scale).to(dq_ptr.dtype.element_ty),
            mask=row_valid[:, None],
        )


@triton.jit
def _dkdv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, lse_ptr, dk_ptr, dv_ptr,
    B, H_kv, S_q, S_k,
    stride_qb, stride_qh,
    stride_kb, stride_kh,
    stride_vb, stride_vh,
    stride_dob, stride_doh,
    stride_ob, stride_oh,
    stride_lse_b, stride_lse_h,
    scale,
    rep,
    num_cores,
    D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(S_k, BLOCK_N)
    num_m_blocks = tl.cdiv(S_q, BLOCK_M)
    num_blocks = B * H_kv * num_n_blocks

    for block_idx in range(pid, num_blocks, num_cores):
        bhk = block_idx // num_n_blocks
        n_block = block_idx - bhk * num_n_blocks
        b = bhk // H_kv
        h_kv = bhk - b * H_kv

        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        col_valid = offs_n < S_k

        kb = b * stride_kb + h_kv * stride_kh
        vb = b * stride_vb + h_kv * stride_vh

        k_j = tl.load(
            k_ptr + kb + offs_n[:, None] * D + offs_d[None, :],
            mask=col_valid[:, None], other=0.0).to(tl.float32)
        v_j = tl.load(
            v_ptr + vb + offs_n[:, None] * D + offs_d[None, :],
            mask=col_valid[:, None], other=0.0).to(tl.float32)
        k_t = tl.trans(k_j)
        v_t = tl.trans(v_j)

        dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
        dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)

        if IS_CAUSAL:
            i_start = tl.maximum(n_block * BLOCK_N - (S_k - S_q), 0) // BLOCK_M
        else:
            i_start = 0

        for r in range(rep):
            h_q = h_kv * rep + r
            q_base = b * stride_qb + h_q * stride_qh
            do_base = b * stride_dob + h_q * stride_doh
            o_base = b * stride_ob + h_q * stride_oh
            lse_base = b * stride_lse_b + h_q * stride_lse_h

            for i_block in range(i_start, num_m_blocks):
                offs_m = i_block * BLOCK_M + tl.arange(0, BLOCK_M)
                row_valid = offs_m < S_q
                q = tl.load(
                    q_ptr + q_base + offs_m[:, None] * D + offs_d[None, :],
                    mask=row_valid[:, None], other=0.0).to(tl.float32)
                do = tl.load(
                    do_ptr + do_base + offs_m[:, None] * D + offs_d[None, :],
                    mask=row_valid[:, None], other=0.0).to(tl.float32)
                o = tl.load(
                    o_ptr + o_base + offs_m[:, None] * D + offs_d[None, :],
                    mask=row_valid[:, None], other=0.0).to(tl.float32)
                lse = tl.load(
                    lse_ptr + lse_base + offs_m,
                    mask=row_valid, other=0.0).to(tl.float32)

                valid = row_valid[:, None] & col_valid[None, :]
                if IS_CAUSAL:
                    valid = valid & (offs_n[None, :] <= offs_m[:, None] + (S_k - S_q))
                s = tl.dot(q, k_t) * scale
                p = tl.exp(tl.where(valid, s, -float("inf")) - lse[:, None])
                dp = tl.dot(do, v_t)
                d_row = tl.sum(do * o, axis=1)
                ds = p * (dp - d_row[:, None])

                dv_acc = tl.dot(tl.trans(p), do, dv_acc)
                dk_acc = tl.dot(tl.trans(ds), q, dk_acc)

        tl.store(
            dk_ptr + kb + offs_n[:, None] * D + offs_d[None, :],
            (dk_acc * scale).to(dk_ptr.dtype.element_ty),
            mask=col_valid[:, None],
        )
        tl.store(
            dv_ptr + vb + offs_n[:, None] * D + offs_d[None, :],
            dv_acc.to(dv_ptr.dtype.element_ty),
            mask=col_valid[:, None],
        )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        try:
            import torch_npu
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except Exception:
            self.CUBE_CORE_NUM = 24

    def forward(self, grad_output, query, key, value, attn_output, logsumexp,
                is_causal=False, enable_gqa=False):
        B, H, S_q, D = query.shape
        H_kv = key.shape[1]
        S_k = key.shape[2]
        scale = 1.0 / (D ** 0.5)

        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        do = grad_output.contiguous()
        o = attn_output.contiguous()
        lse = logsumexp.contiguous()

        rep = (H // H_kv) if (enable_gqa and H != H_kv) else 1
        is_causal = bool(is_causal)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BLOCK_M = 32
        BLOCK_N = 32
        CORES = self.CUBE_CORE_NUM

        # ---- dq kernel: grid over q-row blocks ----
        num_m = (S_q + BLOCK_M - 1) // BLOCK_M
        n_blocks = B * H * num_m
        g1 = n_blocks if n_blocks < CORES else CORES
        _dq_kernel[(g1,)](
            q, k, v, do, o, lse, dq,
            B, H, S_q, S_k,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            do.stride(0), do.stride(1),
            o.stride(0), o.stride(1),
            lse.stride(0), lse.stride(1),
            scale,
            rep,
            g1,
            D=D,
            IS_CAUSAL=is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        # ---- dk/dv kernel: grid over k-row blocks ----
        num_n = (S_k + BLOCK_N - 1) // BLOCK_N
        n_blocks2 = B * H_kv * num_n
        g2 = n_blocks2 if n_blocks2 < CORES else CORES
        _dkdv_kernel[(g2,)](
            q, k, v, do, o, lse, dk, dv,
            B, H_kv, S_q, S_k,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            do.stride(0), do.stride(1),
            o.stride(0), o.stride(1),
            lse.stride(0), lse.stride(1),
            scale,
            rep,
            g2,
            D=D,
            IS_CAUSAL=is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk, dv
