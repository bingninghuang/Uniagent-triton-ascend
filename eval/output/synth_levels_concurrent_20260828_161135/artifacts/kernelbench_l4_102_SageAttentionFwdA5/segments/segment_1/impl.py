"""SageAttention forward (KernelBench 102) for Ascend NPU - Triton implementation.

Single-pass flash-attention style kernel with online softmax in the log2 domain
(exp2 trick).  Supports HND / NHD layouts (NHD via strides, no data copy),
GQA, is_causal (top-left aligned, self-attention), fp16 / bf16 inputs with
fp32 accumulation, and natural-log lse output (fp32).
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 64


def _num_ai_cores():
    """Number of AI cores (persistent-grid size) - read dynamically."""
    try:
        import torch_npu
        n = int(torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40))
        if n > 0:
            return n
    except Exception:
        pass
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(0)
        n = int(getattr(props, "num_cores", 0) or 0)
        if n > 0:
            return n
    except Exception:
        pass
    return 40


_NUM_CORES = _num_ai_cores()


@triton.jit
def _sage_attn_fwd_kernel(
    Q, K, V, O, LSE,
    sqb, sqh, sqs,
    skb, skh, sks,
    svb, svh, svs,
    sob, soh, sos,
    slb, slh, sls,
    B, H, Hkv, S, GRID_M, SM_SCALE,
    NUM_PIDS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # All tensor dims/strides for the workloads here fit comfortably in int32
    # (B*H*S*D < 2**31), so keep everything in int32 arithmetic.
    pid = tl.program_id(0).to(tl.int32)
    B32 = B.to(tl.int32)
    H32 = H.to(tl.int32)
    Hkv32 = Hkv.to(tl.int32)
    S32 = S.to(tl.int32)
    grid_m = GRID_M.to(tl.int32)

    n_units = B32 * H32 * grid_m
    group_size = H32 // Hkv32

    for unit in range(pid, n_units, NUM_PIDS):
        q0 = (unit % grid_m) * BLOCK_M
        bh = unit // grid_m
        b = bh // H32
        hh = bh % H32
        kv_h = hh // group_size

        offs_m = q0 + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        # ---- load q block (kept in input dtype for CUBE dot) ----
        q_ptrs = Q + b * sqb + hh * sqh + offs_m[:, None] * sqs + offs_d[None, :]
        if EVEN_M:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=(offs_m < S32)[:, None], other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        qk_scale = SM_SCALE * 1.4426950408889634  # scores in log2 units

        if IS_CAUSAL:
            di = q0 + BLOCK_M
            hi = tl.minimum(di, S32)
        else:
            hi = S32

        k_base = K + b * skb + kv_h * skh
        v_base = V + b * svb + kv_h * svh

        for j0 in range(0, hi, BLOCK_N):
            n_idx = j0 + offs_n
            if EVEN_N:
                kk = tl.load(k_base + n_idx[:, None] * sks + offs_d[None, :])
            else:
                kk = tl.load(k_base + n_idx[:, None] * sks + offs_d[None, :],
                             mask=(n_idx < S32)[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(kk), out_dtype=tl.float32) * qk_scale

            if IS_CAUSAL:
                scores = tl.where((n_idx[None, :] <= offs_m[:, None])
                                  & (n_idx[None, :] < S32),
                                  scores, float("-inf"))
            elif not EVEN_N:
                scores = tl.where((n_idx[None, :] < S32), scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)

            if EVEN_N:
                vv = tl.load(v_base + n_idx[:, None] * svs + offs_d[None, :])
            else:
                vv = tl.load(v_base + n_idx[:, None] * svs + offs_d[None, :],
                             mask=(n_idx < S32)[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv, out_dtype=tl.float32)
            m_i = m_new

        # ---- epilogue: normalize and store ----
        l_safe = tl.where(l_i == 0, 1.0, l_i)
        o = acc / l_safe[:, None]

        if EVEN_M:
            store_mask = None
        else:
            store_mask = offs_m < S32

        o_ptrs = O + b * sob + hh * soh + offs_m[:, None] * sos + offs_d[None, :]
        o = o.to(q.dtype)
        if EVEN_M:
            tl.store(o_ptrs, o)
        else:
            tl.store(o_ptrs, o, mask=store_mask[:, None])

        if RETURN_LSE:
            # m_i, l_i are in the log2 domain:  lse (natural log) =
            # ln(sum exp(s)) = (m_i + log2(l_i)) * ln2
            lse = (m_i + tl.math.log2(l_safe)) * 0.6931471805599453
            l_ptrs = LSE + b * slb + hh * slh + offs_m * sls
            if EVEN_M:
                tl.store(l_ptrs, lse)
            else:
                tl.store(l_ptrs, lse, mask=store_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, tensor_layout, is_causal, sm_scale, return_lse):
        device = q.device
        B = q.shape[0]
        if tensor_layout == "NHD":
            _, S, H, D = q.shape
            Hkv = k.shape[2]
            # contiguous [B, S, H, D] interpreted as (b, h, s, d):
            sqb, sqh, sqs = S * H * D, D, H * D
            skb, skh, sks = S * Hkv * D, D, Hkv * D
            svb, svh, svs = S * Hkv * D, D, Hkv * D
            sob, soh, sos = S * H * D, D, H * D
            slb, slh, sls = S * H, H, 1  # lse [B, S, H]
        else:
            _, H, S, D = q.shape
            Hkv = k.shape[1]
            sqb, sqh, sqs = H * S * D, S * D, D
            skb, skh, sks = Hkv * S * D, S * D, D
            svb, svh, svs = Hkv * S * D, S * D, D
            sob, soh, sos = H * S * D, S * D, D
            slb, slh, sls = H * S, S, 1  # lse [B, H, S]

        o = torch.empty_like(q)
        grid_m = triton.cdiv(S, BLOCK_M)
        npids = min(_NUM_CORES, B * H * grid_m)

        if return_lse:
            lse = torch.empty(
                (B, S, H) if tensor_layout == "NHD" else (B, H, S),
                dtype=torch.float32, device=device)
        else:
            lse = q  # unused placeholder

        _sage_attn_fwd_kernel[(npids,)](
            q, k, v, o, lse,
            sqb, sqh, sqs,
            skb, skh, sks,
            svb, svh, svs,
            sob, soh, sos,
            slb, slh, sls,
            B, H, Hkv, S, grid_m, float(sm_scale),
            npids,
            IS_CAUSAL=bool(is_causal),
            RETURN_LSE=bool(return_lse),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=D,
            EVEN_M=(S % BLOCK_M == 0),
            EVEN_N=(S % BLOCK_N == 0),
        )

        if return_lse:
            return o, lse
        return o
