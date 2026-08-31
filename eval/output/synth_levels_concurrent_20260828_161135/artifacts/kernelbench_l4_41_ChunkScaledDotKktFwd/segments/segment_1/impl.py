"""Triton Ascend implementation of KernelBench problem 41: ChunkScaledDotKktFwd.

Computes A[b, t, h, :] for every (batch b, head h, time t) as the gated,
row-scaled, strictly lower triangular chunk matrix  beta * (Kc @ Kc^T)  where
Kc is the chunk (size BT, default 64) of  k[b, :, h // group, :]  containing t:

    A_chunk[i, j] = beta_i * dot(Kc[i, :], Kc[j, :]) * exp(min(g_i - g_j, 0))
                    * (i > j)

Design:
  * One fused @triton.jit kernel. Grid is clamped to the number of vector
    cores (G4/T1); each program loops over its share of (batch, chunk, head)
    blocks.
  * Kc @ Kc^T is a single tl.dot of a bf16 [BT, K] tile with its transpose
    (fp32 accumulation, exactly the golden-kernel behavior).
  * varlen mode (cu_seqlens given, B == 1): a tiny single-block table kernel
    computes per-sequence chunk counts + exclusive prefix sums; the main
    kernel maps each global chunk id to (sequence, chunk offset) with pure
    vector ops - no host sync, no data-dependent branch.
"""

import torch
import triton
import triton.language as tl

FLA_CHUNK_SIZE = 64


def _get_vec_core_num(device):
    """G1: read the vector-core count dynamically (never hardcode)."""
    idx = 0
    if isinstance(device, torch.device) and device.index is not None:
        idx = device.index
    try:
        from triton.runtime.driver import driver
        props = driver.active.utils.get_device_properties(idx)
        for attr in ("num_vectorcore", "num_aicore", "ai_core_num"):
            try:
                n = int(getattr(props, attr, 0) or 0)
                if n > 0:
                    return n
            except Exception:
                pass
    except Exception:
        pass
    try:
        import torch_npu
        n = int(torch_npu.npu.npu_config.get_device_limit(idx).get(
            "vector_core_num", 0) or 0)
        if n > 0:
            return n
    except Exception:
        pass
    return 48


@triton.jit
def _varlen_chunk_table_kernel(
    cu_ptr,          # in:  int32 [n_seq + 1] cumulative sequence lens
    n_chunks_ptr,    # out: int32 [MAX_NSEQ] chunks per sequence
    cum_chunks_ptr,  # out: int32 [MAX_NSEQ] exclusive prefix sum of n_chunks
    n_seq,
    BT: tl.constexpr,
    MAX_NSEQ: tl.constexpr,
):
    # Single blocked program; MAX_NSEQ is small (constexpr, fully unrolled).
    running = 0
    for i in range(0, MAX_NSEQ):
        m = i < n_seq
        c = tl.load(cu_ptr + i, mask=m, other=0).to(tl.int32)
        d = tl.load(cu_ptr + i + 1, mask=m, other=0).to(tl.int32)
        n = tl.where(m, (d - c + BT - 1) // BT, 0)
        tl.store(n_chunks_ptr + i, n, mask=m)
        tl.store(cum_chunks_ptr + i, running, mask=m)
        running = running + n


@triton.jit
def _chunk_kkt_fwd_kernel(
    k_ptr, g_ptr, beta_ptr, out_ptr,
    cu_ptr, n_chunks_ptr, cum_chunks_ptr,
    total_units,
    T, nch_per_b,
    stride_kb, stride_kt, stride_kh,
    stride_gb, stride_gt, stride_gh,
    stride_bb, stride_bt, stride_bh,
    stride_ob, stride_ot, stride_oh,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    num_pids: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    K,
    HAS_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NEED_K_MASK: tl.constexpr,
    MAX_NSEQ: tl.constexpr,
    n_seq,
):
    pid = tl.program_id(0).to(tl.int32)
    total_blocks = H * total_units

    o = tl.arange(0, BT)          # chunk-local index, int32
    ko = tl.arange(0, BK)         # head-dim index, int32

    for blk in range(pid, total_blocks, num_pids):
        h = blk % H
        unit = blk // H           # global "chunk" id (varlen) / (b * nch + c)
        hg = h // GROUP

        if IS_VARLEN:
            idx = tl.arange(0, MAX_NSEQ)
            m = idx < n_seq
            ce = tl.load(cum_chunks_ptr + idx, mask=m, other=0).to(tl.int32)
            nv = tl.load(n_chunks_ptr + idx, mask=m, other=0).to(tl.int32)
            cu0 = tl.load(cu_ptr + idx, mask=m, other=0).to(tl.int32)
            cu1 = tl.load(cu_ptr + idx + 1, mask=m, other=0).to(tl.int32)
            # Exactly one sequence i (if the unit is valid) satisfies
            # ce[i] <= unit < ce[i] + nv[i]; invalid units match none, giving
            # t0 >= t1 below so the row mask is all-false and no store happens.
            sel = m & (ce <= unit) & (unit < ce + nv)
            ce_u = tl.max(tl.where(sel, ce, 0), axis=0)
            t0 = tl.max(tl.where(sel, cu0, 0), axis=0).to(tl.int32) \
                 + (unit - ce_u) * BT
            t1 = tl.max(tl.where(sel, cu1, 0), axis=0).to(tl.int32)
            b = 0
        else:
            b = unit // nch_per_b
            c = unit - b * nch_per_b
            t0 = c * BT
            t1 = T

        # Global time rows of this chunk block: [t0, t0 + BT)
        t_row = t0 + o
        row_ok = t_row < t1

        tl.assume(stride_kh >= 1)
        tl.assume(stride_oh >= 1)

        k_ptrs = k_ptr + b * stride_kb + t_row[:, None] * stride_kt \
            + hg * stride_kh + ko[None, :]
        if NEED_K_MASK:
            k_mask = row_ok[:, None] & (ko < K)[None, :]
            kc = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            kc = tl.load(k_ptrs, mask=row_ok[:, None], other=0.0)

        # Kc @ Kc^T : bf16/fp16 inputs, fp32 accumulation (golden behavior)
        acc = tl.dot(kc, tl.trans(kc))

        if HAS_G:
            g_val = tl.load(
                g_ptr + b * stride_gb + t_row * stride_gt + h * stride_gh,
                mask=row_ok, other=0.0).to(tl.float32)
            diff = g_val[:, None] - g_val[None, :]
            acc = acc * tl.exp(tl.minimum(diff, 0.0))

        b_val = tl.load(
            beta_ptr + b * stride_bb + t_row * stride_bt + h * stride_bh,
            mask=row_ok, other=0.0).to(tl.float32)
        acc = acc * b_val[:, None]

        # Strictly lower triangular within the chunk: keep i > j only.
        # Padding rows/cols (t >= t1) are already zero because their Kc rows
        # were loaded as 0, so no extra zeroing is needed.
        acc = tl.where(o[:, None] > o[None, :], acc, 0.0)

        out_ptrs = out_ptr + b * stride_ob + t_row[:, None] * stride_ot \
            + h * stride_oh + o[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty),
                 mask=row_ok[:, None])


class ModelNew(torch.nn.Module):
    """Triton Ascend implementation of the chunk-scaled dot KKT forward op."""

    FLA_CHUNK_SIZE = 64

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, k, g=None, beta=None, cu_seqlens=None, chunk_indices=None,
                chunk_size=None, output_dtype=torch.float32):
        if chunk_size is None:
            chunk_size = self.FLA_CHUNK_SIZE
        BT = int(chunk_size)
        assert BT == triton.next_power_of_2(BT) and BT <= 256, \
            "chunk_size must be a power of two (<= 256)"
        BK = triton.next_power_of_2(k.shape[-1])

        k = k.contiguous()
        if g is not None:
            g = g.contiguous()
        beta = beta.contiguous()

        B, T, Hg, K = k.shape
        H = beta.shape[-1]
        GROUP = H // Hg

        A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

        n_seq = 0
        cu = None
        table = None
        if cu_seqlens is not None:
            # varlen: B must be 1; no host sync - mapping is done on device.
            n_seq = cu_seqlens.numel() - 1
            cu = cu_seqlens.to(dtype=torch.int32)
            if not cu.is_contiguous():
                cu = cu.contiguous()
            # triton.next_power_of_2 already returns >= 1, and varlen always
            # has n_seq >= 1, so both are safe without min/max calls.
            max_nseq = triton.next_power_of_2(n_seq) if n_seq > 0 else 1
            table = torch.empty(2 * max_nseq, dtype=torch.int32,
                                device=k.device)
            total_units = (T + BT - 1) // BT + (n_seq - 1 if n_seq > 0 else 0)
            _varlen_chunk_table_kernel[(1,)](
                cu, table[:max_nseq], table[max_nseq:],
                n_seq, BT=BT, MAX_NSEQ=max_nseq,
            )
            is_varlen = True
        else:
            nch_per_b = (T + BT - 1) // BT
            total_units = B * nch_per_b
            if table is None:
                # Dummies; never read when IS_VARLEN is False.
                table = torch.empty(2, dtype=torch.int32, device=k.device)
                cu = table  # dummy pointer
            max_nseq = 1
            is_varlen = False

        has_g = g is not None
        g_arg = g if g is not None else beta  # unused dummy when HAS_G=False

        num_cores = _get_vec_core_num(k.device)
        total_blocks = H * total_units
        grid_n = total_blocks if total_blocks < num_cores else num_cores

        _chunk_kkt_fwd_kernel[(grid_n,)](
            k, g_arg, beta, A,
            cu, table[:max_nseq], table[max_nseq:],
            total_units,
            T, (T + BT - 1) // BT,
            k.stride(0), k.stride(1), k.stride(2),
            g_arg.stride(0), g_arg.stride(1), g_arg.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            H=H, GROUP=GROUP,
            num_pids=grid_n,
            BT=BT, BK=BK,
            K=K,
            HAS_G=has_g,
            IS_VARLEN=is_varlen,
            NEED_K_MASK=(K != BK),
            MAX_NSEQ=max_nseq,
            n_seq=n_seq,
        )
        return A
