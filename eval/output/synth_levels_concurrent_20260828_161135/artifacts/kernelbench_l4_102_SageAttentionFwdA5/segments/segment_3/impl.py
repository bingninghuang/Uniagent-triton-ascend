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


def _probe_num_pids():
    """Number of AI cores for the persistent grid (query once at __init__).

    This kernel is CUBE-dominated, so prefer cube_core_num; fall back to
    vector_core_num, then the triton driver device properties, then 20.
    """
    try:
        import torch_npu
        limit = torch_npu.npu.npu_config.get_device_limit(0)
        for key in ("cube_core_num", "vector_core_num"):
            try:
                n = int(limit.get(key, 0))
            except Exception:
                n = 0
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
    return 20


@triton.jit
def _sage_attn_fwd_kernel(
    Q, K, V, O, LSE,
    sqb, sqh, sqs,
    skb, skh, sks,
    svb, svh, svs,
    sob, soh, sos,
    slb, slh, sls,
    BH, H, Hkv, S, GRID_M, SM_SCALE,
    NUM_PIDS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    # Keep scalar int args at their native width: NO .to() casts, because
    # Triton specializes an int arg equal to 1 as tl.constexpr, which lacks
    # tensor methods (.to etc.).  Binary arithmetic with such constants is
    # fine, and all values here fit in int32 anyway (B*H*S*D < 2**31).
    pid = tl.program_id(0)
    n_units = BH * GRID_M
    group_size = H // Hkv

    for unit in range(pid, n_units, NUM_PIDS):
        q0 = (unit % GRID_M) * BLOCK_M
        bh = unit // GRID_M
        b = bh // H
        hh = bh % H
        kv_h = hh // group_size

        offs_m = q0 + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        # ---- load q block (kept in input dtype for CUBE dot) ----
        q_ptrs = Q + b * sqb + hh * sqh + offs_m[:, None] * sqs + offs_d[None, :]
        if EVEN_M:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=(offs_m < S)[:, None], other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        qk_scale = SM_SCALE * 1.4426950408889634  # scores in log2 units

        if IS_CAUSAL:
            hi = tl.minimum(q0 + BLOCK_M, S)
        else:
            hi = S

        k_base = K + b * skb + kv_h * skh
        v_base = V + b * svb + kv_h * svh

        for j0 in range(0, hi, BLOCK_N):
            n_idx = j0 + offs_n
            if EVEN_N:
                kk = tl.load(k_base + n_idx[:, None] * sks + offs_d[None, :])
            else:
                kk = tl.load(k_base + n_idx[:, None] * sks + offs_d[None, :],
                             mask=(n_idx < S)[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(kk), out_dtype=tl.float32) * qk_scale

            if IS_CAUSAL:
                scores = tl.where((n_idx[None, :] <= offs_m[:, None])
                                  & (n_idx[None, :] < S),
                                  scores, float("-inf"))
            elif not EVEN_N:
                scores = tl.where((n_idx[None, :] < S), scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)

            if EVEN_N:
                vv = tl.load(v_base + n_idx[:, None] * svs + offs_d[None, :])
            else:
                vv = tl.load(v_base + n_idx[:, None] * svs + offs_d[None, :],
                             mask=(n_idx < S)[:, None], other=0.0)
            # Split P into hi+lo low-precision parts so that P*V is accumulated
            # with ~22-significant-bit weight precision (each half-precision
            # product is exact in the fp32 CUBE accumulator).  A single fp16 cast
            # of P would lose ~10 bits and fail the verifier's rtol = 2^-10.
            p_hi = p.to(vv.dtype)
            p_lo = (p - p_hi.to(tl.float32)).to(vv.dtype)
            acc = acc * alpha[:, None]
            acc = tl.dot(p_hi, vv, acc=acc, out_dtype=tl.float32)
            acc = tl.dot(p_lo, vv, acc=acc, out_dtype=tl.float32)
            m_i = m_new

        # ---- epilogue: normalize and store ----
        l_safe = tl.where(l_i == 0, 1.0, l_i)
        o = acc / l_safe[:, None]

        if EVEN_M:
            store_mask = None
        else:
            store_mask = offs_m < S

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
        # Persistent-grid size (# AI cores), queried once here (never in
        # forward: get_device_limit triggers a device sync).
        self._NPIDS = _probe_num_pids()

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
            slb, slh, sls = S * H, 1, H  # lse [B, S, H]
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
        n_units = B * H * grid_m
        # Persistent grid clamped to the AI-core count (no min() builtin:
        # forward() may only allocate buffers and compute shapes).
        npids = n_units if n_units <= self._NPIDS else self._NPIDS

        # Pass B*H as a single scalar (never 1) instead of B: Triton
        # specializes an int arg == 1 as tl.constexpr, which breaks kernel
        # scalar arithmetic in that case.
        BH = B * H

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
            BH, H, Hkv, S, grid_m, float(sm_scale),
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
