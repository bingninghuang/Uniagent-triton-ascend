"""Triton Ascend implementation of KernelBench problem 41: ChunkScaledDotKktFwd.

Computes A[b, t, h, j] for every (batch b, head h, time t) as the gated,
row-scaled, strictly lower triangular chunk matrix  beta * (Kc @ Kc^T) where
Kc is the chunk (size BT, default 64) of  k[b, :, h // group, :]  containing t:

    A_chunk[i, j] = beta_i * dot(Kc[i, :], Kc[j, :]) * exp(min(g_i - g_j, 0))
                    * (i > j)

Design notes (Ascend 910B1 / BiShengIR):
  * A single program computing one full [BT, BT] chunk block with its
    elementwise chain (diff/min/exp/muls/where on ~9 tiles of 16KB fp32)
    overflows the 192KB AICore UB (backend demanded ~456KB with multi-buffers).
    The kernel is therefore tiled further: one program computes one [16, 16]
    block of the chunk output (grid = units * H * 4 * 4).  Per-program working
    set is then ~1-2KB of fp32 tiles plus two [16, K] bf16 k tiles, which the
    backend compiles with wide margin.
  * Kc @ Kc^T for the block is a single tl.dot (bf16 inputs, fp32 acc) of the
    16 chunk rows against 16 chunk columns (loaded directly in transposed
    layout, no tl.trans needed).
  * varlen mode (cu_seqlens given, B == 1): a tiny single-block table kernel
    computes per-sequence chunk counts + exclusive prefix sums; the main
    kernel maps each global chunk id to (sequence, chunk offset, span end)
    with a scalar loop - no host sync, no data-dependent tensor branches.
"""

import torch
import triton
import triton.language as tl

FLA_CHUNK_SIZE = 64
BLK = 16  # output block side; BT must be a multiple of BLK (power-of-two BT)


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
def _chunk_kkt_blk_kernel(
    k_ptr, g_ptr, beta_ptr, out_ptr,
    cu_ptr, n_chunks_ptr, cum_chunks_ptr,
    T, nch_per_b, n_seq,
    stride_kb, stride_kt, stride_kh,
    stride_gb, stride_gt, stride_gh,
    stride_bb, stride_bt, stride_bh,
    stride_ob, stride_ot, stride_oh,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    BT: tl.constexpr,
    BLK: tl.constexpr,
    NBL: tl.constexpr,     # BT // BLK blocks per side
    BK: tl.constexpr,      # next_pow2(K)
    K: tl.constexpr,
    HAS_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NEED_K_MASK: tl.constexpr,
):
    blk = tl.program_id(0).to(tl.int32)
    h = blk % H
    rest = blk // H
    cb = rest % (NBL * NBL)
    rb = cb // NBL
    cbk = cb % NBL
    unit = rest // (NBL * NBL)

    # ---- unit -> (b, t0, t1): chunk span in global time -----------------
    if IS_VARLEN:
        b = 0
        # scalar lookup: exactly one sequence s has cum[s] <= unit < cum[s]+n[s];
        # for invalid ids (unit beyond the total chunk count) no s matches,
        # giving t1 == 0 below so every mask is all-false and nothing is
        # read or written.
        t0 = 0
        t1 = 0
        found = 0
        for s in range(0, n_seq):
            c0 = tl.load(cu_ptr + s).to(tl.int32)
            c1 = tl.load(cu_ptr + s + 1).to(tl.int32)
            n = (c1 - c0 + BT - 1) // BT
            ce = tl.load(cum_chunks_ptr + s).to(tl.int32)
            hit = (found == 0) & (unit >= ce) & (unit < ce + n)
            t0 = tl.where(hit, c0 + (unit - ce) * BT, t0)
            t1 = tl.where(hit, c1, t1)
            found = tl.where(hit, 1, found)
    else:
        b = unit // nch_per_b
        c = unit - b * nch_per_b
        t0 = c * BT
        t1 = T

    # ---- block-local indices --------------------------------------------
    ir = tl.arange(0, BLK)
    ic = tl.arange(0, BLK)
    ko = tl.arange(0, BK)

    i_loc = rb * BLK + ir       # chunk-local row index
    j_loc = cbk * BLK + ic      # chunk-local col index
    t_r = t0 + i_loc            # global rows
    t_ct = t0 + j_loc           # global cols (chunk rows of Kc on the right)

    r_ok = t_r < t1
    c_ok = t_ct < t1

    # k[b, t, hg, q]: left operand rows + right operand rows in transposed
    # tile layout -> one tl.dot per program, no tl.trans in the kernel.
    k_row_ptrs = k_ptr + b * stride_kb + t_r[:, None] * stride_kt \
        + (h // GROUP) * stride_kh + ko[None, :]
    k_col_ptrs = k_ptr + b * stride_kb + t_ct[None, :] * stride_kt \
        + (h // GROUP) * stride_kh + ko[:, None]

    if NEED_K_MASK:
        kc_r = tl.load(k_row_ptrs,
                       mask=r_ok[:, None] & (ko < K)[None, :], other=0.0)
        kc_c = tl.load(k_col_ptrs,
                       mask=c_ok[None, :] & (ko < K)[:, None], other=0.0)
    else:
        kc_r = tl.load(k_row_ptrs, mask=r_ok[:, None], other=0.0)
        kc_c = tl.load(k_col_ptrs, mask=c_ok[None, :], other=0.0)

    acc = tl.dot(kc_r, kc_c)    # [BLK, BLK], fp32 accumulate

    if HAS_G:
        g_ptr_off = g_ptr + b * stride_gb
        g_r = tl.load(g_ptr_off + t_r * stride_gt + h * stride_gh,
                      mask=r_ok, other=0.0).to(tl.float32)
        g_c = tl.load(g_ptr_off + t_ct * stride_gt + h * stride_gh,
                      mask=c_ok, other=0.0).to(tl.float32)
        diff = g_r[:, None] - g_c[None, :]
        acc = acc * tl.exp(tl.minimum(diff, 0.0))

    beta_r = tl.load(beta_ptr + b * stride_bb + t_r * stride_bt + h * stride_bh,
                     mask=r_ok, other=0.0).to(tl.float32)
    acc = acc * beta_r[:, None]

    acc = tl.where(i_loc[:, None] > j_loc[None, :], acc, 0.0)

    o_ptrs = out_ptr + b * stride_ob + t_r[:, None] * stride_ot \
        + h * stride_oh + j_loc[None, :]
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=r_ok[:, None])


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
        assert (BT & (BT - 1)) == 0 and BT % BLK == 0 and BT >= 2 * BLK, \
            "chunk_size must be a power of two, a multiple of 16, and >= 32"
        BK = triton.next_power_of_2(k.shape[-1])

        k = k.contiguous()
        if g is not None:
            g = g.contiguous()
        beta = beta.contiguous()

        B, T, Hg, K = k.shape
        H = beta.shape[-1]
        GROUP = H // Hg

        A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)

        nch_per_b = (T + BT - 1) // BT
        if cu_seqlens is not None:
            # varlen: B must be 1; all mapping is on device (no host sync).
            n_seq = cu_seqlens.numel() - 1
            cu = cu_seqlens.to(dtype=torch.int32)
            if not cu.is_contiguous():
                cu = cu.contiguous()
            max_nseq = triton.next_power_of_2(n_seq)
            if max_nseq < 16:
                max_nseq = 16
            table = torch.empty(2 * max_nseq, dtype=torch.int32,
                                device=k.device)
            total_units = (T + BT - 1) // BT + (n_seq - 1 if n_seq > 0 else 0)
            _varlen_chunk_table_kernel[(1,)](
                cu, table[:max_nseq], table[max_nseq:],
                n_seq, BT=BT, MAX_NSEQ=max_nseq,
            )
            cu_arg, nch_arg, cum_arg = cu, table[:max_nseq], table[max_nseq:]
            is_varlen = True
        else:
            n_seq = 0
            total_units = B * nch_per_b
            # Dummies; never read when IS_VARLEN is False.
            dummy = torch.empty(4, dtype=torch.int32, device=k.device)
            cu_arg = nch_arg = cum_arg = dummy
            is_varlen = False

        has_g = g is not None
        g_arg = g if g is not None else beta

        nbl = BT // BLK
        total_blocks = total_units * H * nbl * nbl

        _chunk_kkt_blk_kernel[(total_blocks,)](
            k, g_arg, beta, A,
            cu_arg, nch_arg, cum_arg,
            T, nch_per_b, n_seq,
            k.stride(0), k.stride(1), k.stride(2),
            g_arg.stride(0), g_arg.stride(1), g_arg.stride(2),
            beta.stride(0), beta.stride(1), beta.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            H=H, GROUP=GROUP,
            BT=BT, BLK=BLK, NBL=nbl, BK=BK,
            K=K,
            HAS_G=has_g,
            IS_VARLEN=is_varlen,
            NEED_K_MASK=(K != BK),
        )
        return A
