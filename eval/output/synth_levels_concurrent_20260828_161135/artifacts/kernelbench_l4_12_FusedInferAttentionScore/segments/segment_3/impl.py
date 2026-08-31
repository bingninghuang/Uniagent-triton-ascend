import torch
import triton
import triton.language as tl


def _get_num_cores(device=0):
    # G1: dynamically read core count, no hardcoding.
    try:
        from triton.runtime.driver import active
        props = active.utils.get_device_properties(device)
        for attr in ("num_aicore", "num_vectorcore", "multi_processor_count"):
            n = getattr(props, attr, None)
            if n and int(n) > 0:
                return int(n)
    except Exception:
        pass
    try:
        import torch_npu
        n = torch_npu.npu.npu_config.get_device_limit(device).get("vector_core_num", 24)
        return int(n)
    except Exception:
        return 24


@triton.jit
def _fused_attn_fwd(
    Q, K, V, O, M,
    batch, num_heads, gqa, sq, skv, head_dim, num_grids,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_os, stride_oh,
    stride_mb, stride_mh, stride_ms, stride_md,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    HAS_MASK: tl.constexpr, MASK_BOOL: tl.constexpr, EVEN_SKV: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    pid = tl.program_id(0).to(tl.int32)
    NUM_M = (sq + BLOCK_M - 1) // BLOCK_M
    NUM_TOTAL = batch * num_heads * NUM_M

    offs_m0 = tl.arange(0, BLOCK_M).to(tl.int32)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int32)

    for block_idx in range(pid, NUM_TOTAL, num_grids):
        m_blk = block_idx % NUM_M
        t = block_idx // NUM_M
        h = t % num_heads
        b = t // num_heads
        hkv = h // gqa

        offs_m0 = m_blk * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
        m_row_ok = offs_m0 < sq
        d_ok = offs_d < head_dim

        q_ptrs = (Q + b * stride_qb + h * stride_qh
                  + offs_m0[:, None] * stride_qs + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=m_row_ok[:, None] & d_ok[None, :], other=0.0)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for start_n in range(0, skv, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N).to(tl.int32)
            n_ok = offs_n < skv

            k_ptrs = (K + b * stride_kb + hkv * stride_kh
                      + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks)
            k = tl.load(k_ptrs, mask=d_ok[:, None] & n_ok[None, :], other=0.0)

            qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N] fp32
            qk = qk * scale

            if not EVEN_SKV:
                qk = tl.where(n_ok[None, :], qk, float("-inf"))

            if HAS_MASK:
                mm_ptrs = (M + b * stride_mb + h * stride_mh
                           + offs_m0[:, None] * stride_ms + offs_n[None, :] * stride_md)
                load_ok = m_row_ok[:, None] & n_ok[None, :]
                if MASK_BOOL:
                    mm = tl.load(mm_ptrs, mask=load_ok, other=1)
                    qk = tl.where(mm.to(tl.int1), qk, float("-inf"))
                else:
                    mm = tl.load(mm_ptrs, mask=load_ok, other=0.0)
                    qk = qk + mm.to(tl.float32)

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2((m_i - m_new) * LOG2E)
            p = tl.math.exp2((qk - m_new[:, None]) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            v_ptrs = (V + b * stride_vb + hkv * stride_vh
                      + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
            v = tl.load(v_ptrs, mask=n_ok[:, None] & d_ok[None, :], other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(Q.dtype.element_ty), v)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        o = acc / l_safe[:, None]
        o = tl.where(l_i[:, None] == 0.0, 0.0, o)

        o_ptrs = (O + b * stride_ob + offs_m0[:, None] * stride_os
                  + (h * head_dim + offs_d[None, :]) * stride_oh)
        tl.store(o_ptrs, o.to(Q.dtype.element_ty), mask=m_row_ok[:, None] & d_ok[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                pse_shift=None, atten_mask=None, actual_seq_lengths=None,
                actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None,
                dequant_scale2=None, quant_scale2=None, quant_offset2=None,
                antiquant_scale=None, antiquant_offset=None, block_table=None,
                query_padding_size=None, kv_padding_size=None,
                key_antiquant_scale=None, key_antiquant_offset=None,
                value_antiquant_scale=None, value_antiquant_offset=None,
                key_shared_prefix=None, value_shared_prefix=None,
                actual_shared_prefix_len=None, query_rope=None, key_rope=None,
                key_rope_antiquant_scale=None, num_heads=1, scale=1.0,
                pre_tokens=2147483647, next_tokens=2147483647,
                input_layout="BSH", num_key_value_heads=0, sparse_mode=0,
                inner_precise=0, block_size=0, antiquant_mode=0,
                softmax_lse_flag=False, key_antiquant_mode=0, value_antiquant_mode=0):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        H = num_heads
        nkv = num_key_value_heads if num_key_value_heads > 0 else H
        gqa = H // nkv
        dtype = query.dtype

        if query.dim() == 3:
            # BSH: [B, Sq, H*D]
            B, sq, hidden = query.shape
            skv = key.shape[1]
            head_dim = hidden // H
            q_strides = (query.stride(0), head_dim, H * head_dim, 1)
            kv_head_dim = key.shape[2] // nkv
            k_strides = (key.stride(0), kv_head_dim, nkv * kv_head_dim, 1)
            v_strides = (value.stride(0), kv_head_dim, nkv * kv_head_dim, 1)
        else:
            # 4D: BNSD [B, H, S, D] or BSND [B, S, H, D]
            B = query.shape[0]
            head_dim = query.shape[-1]
            if input_layout == "BNSD":
                Ht = query.shape[1]
                sq = query.shape[2]
                q_strides = (query.stride(0), query.stride(1), query.stride(2), 1)
            else:  # BSND
                sq = query.shape[1]
                Ht = query.shape[2]
                q_strides = (query.stride(0), query.stride(2), query.stride(1), 1)
            nkv_t = key.shape[1] if input_layout == "BNSD" else key.shape[2]
            skv = key.shape[2] if input_layout == "BNSD" else key.shape[1]
            kv_head_dim = key.shape[-1]
            if input_layout == "BNSD":
                k_strides = (key.stride(0), key.stride(1), key.stride(2), 1)
                v_strides = (value.stride(0), value.stride(1), value.stride(2), 1)
            else:
                k_strides = (key.stride(0), key.stride(2), key.stride(1), 1)
                v_strides = (value.stride(0), value.stride(2), value.stride(1), 1)
            H = Ht
            nkv = nkv_t
            gqa = H // nkv

        out = torch.empty((B, sq, H * head_dim), dtype=dtype, device=query.device)

        if atten_mask is not None:
            HAS_MASK = True
            MASK_BOOL = (atten_mask.dtype == torch.bool)
            mask = atten_mask.contiguous()
            if mask.dim() == 4:
                m_strides = (mask.stride(0), 0 if mask.shape[1] == 1 else mask.stride(1),
                             mask.stride(2), mask.stride(3))
            else:
                m_strides = (mask.stride(0) if mask.dim() > 1 else 0,
                             0,
                             mask.stride(-2) if mask.dim() > 1 else mask.shape[-1],
                             1)
        else:
            HAS_MASK = False
            MASK_BOOL = False
            mask = query  # dummy pointer
            m_strides = (0, 0, 0, 1)

        BLOCK_D = triton.next_power_of_2(head_dim)
        if BLOCK_D >= 256:
            BLOCK_M = 16
            BLOCK_N = 32 if BLOCK_D >= 512 else 64
        else:
            BLOCK_M = 64
            BLOCK_N = 64

        NUM_M = triton.cdiv(sq, BLOCK_M)
        NUM_TOTAL = B * H * NUM_M
        num_cores = _get_num_cores(query.device.index or 0)
        grid_size = num_cores if NUM_TOTAL > num_cores else NUM_TOTAL

        _fused_attn_fwd[(grid_size,)](
            query, key, value, out, mask,
            B, H, gqa, sq, skv, head_dim, grid_size,
            q_strides[0], q_strides[1], q_strides[2], q_strides[3],
            k_strides[0], k_strides[1], k_strides[2], k_strides[3],
            v_strides[0], v_strides[1], v_strides[2], v_strides[3],
            out.stride(0), out.stride(1), out.stride(2),
            m_strides[0], m_strides[1], m_strides[2], m_strides[3],
            float(scale),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            HAS_MASK=HAS_MASK, MASK_BOOL=MASK_BOOL,
            EVEN_SKV=(skv % BLOCK_N == 0),
        )
        return out, None
