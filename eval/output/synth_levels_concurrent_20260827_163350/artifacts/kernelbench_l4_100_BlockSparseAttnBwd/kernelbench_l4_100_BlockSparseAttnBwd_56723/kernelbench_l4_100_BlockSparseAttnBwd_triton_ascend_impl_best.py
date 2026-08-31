# KernelBench L4-100: BlockSparseAttnBwd (MIT Han Lab Block-Sparse-Attention backward)
# Ascend NPU (910B1) Triton implementation.
#
# Two-kernel FlashAttention-2 style backward (m / l are pre-given by the forward):
#   K1 (dQ):  program owns (b, qhead, i-row-block); loops over j col-blocks
#   K2 (dK/dV): program owns (b, khead, j-col-block); loops g over G q-heads x i
# delta = rowsum(dY*O) is computed on the fly inside each kernel.
#
# Masks (all in *local* per-batch coordinates, matching the reference):
#   - causal element mask:            keep col <= row + (sk - sq)      [all heads]
#   - exact streaming (elementwise):  keep col <= min(row+sk-sq, sk)
#                                     and not(sink <= col < row+sk-sq-(local-1))
#   - block streaming (128-blocks):   keep rowblk >= start and
#                                     (local window | sink blocks)
#   - block sparse (128-blocks):      keep mask bit [b, rank(h), rowblk, colblk]

import torch
import torch.nn as nn
import triton
import triton.language as tl

_NPU_CACHE = {}


def _vec_core_num(device):
    idx = 0
    if device is not None:
        try:
            idx = int(device.index) if device.index is not None else 0
        except Exception:
            idx = 0
    if idx not in _NPU_CACHE:
        n = 48
        try:
            import torch_npu
            n = torch_npu.npu.npu_config.get_device_limit(idx).get('vector_core_num', 48)
        except Exception:
            try:
                props = triton.runtime.driver.active.utils.get_device_properties(device)
                n = props.get('num_vectorcore', 48)
            except Exception:
                n = 48
        _NPU_CACHE[idx] = int(n)
    return _NPU_CACHE[idx]


@triton.jit
def _bs_rank_kernel(hmt_ptr, rank_ptr, H: tl.constexpr, num_pids):
    # rank[h] = number of blocksparse (==1) heads with index < h
    for h in range(tl.program_id(0), H, num_pids):
        c = 0
        for h2 in range(0, h):
            v = tl.load(hmt_ptr + h2)
            c += (v == 1).to(tl.int32)
        tl.store(rank_ptr + h, c)


@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, dy_ptr, o_ptr,
    m_ptr, l_ptr, dq_ptr,
    cuq_ptr, cuk_ptr,
    hmt_ptr, rank_ptr, si_ptr, mask_ptr,
    scale, H, G,
    ms0, ms1,
    msk0, msk1, msk2,
    ds0, ds1,                 # data strides (row, head)
    nrow_max,
    total_tasks,
    is_causal: tl.constexpr,
    exact_streaming: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    num_pids,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_bn = tl.arange(0, BN)
    offs_bm = tl.arange(0, BM)

    for t in range(pid, total_tasks, num_pids):
        t32 = t.to(tl.int32)
        i = t32 % nrow_max
        tmp = t32 // nrow_max
        h = tmp % H
        b = tmp // H

        qs = tl.load(cuq_ptr + b).to(tl.int32)
        qe = tl.load(cuq_ptr + b + 1).to(tl.int32)
        ks = tl.load(cuk_ptr + b).to(tl.int32)
        ke = tl.load(cuk_ptr + b + 1).to(tl.int32)
        sq = qe - qs
        sk = ke - ks

        nrow_b = (sq + BM - 1) // BM
        if i < nrow_b:
            t_offs = i * BM + offs_bm
            valid_r = t_offs < sq

            hmt_v = tl.load(hmt_ptr + h)
            rank_v = tl.load(rank_ptr + h).to(tl.int32)
            sink_v = tl.load(si_ptr + 2 * h).to(tl.int32)
            local_v = tl.load(si_ptr + 2 * h + 1).to(tl.int32)

            q_v = tl.load(q_ptr + (qs + t_offs)[:, None] * ds0 + h * ds1 + offs_d[None, :],
                          mask=valid_r[:, None], other=0.0)
            dy_v = tl.load(dy_ptr + (qs + t_offs)[:, None] * ds0 + h * ds1 + offs_d[None, :],
                           mask=valid_r[:, None], other=0.0)
            o_v = tl.load(o_ptr + (qs + t_offs)[:, None] * ds0 + h * ds1 + offs_d[None, :],
                          mask=valid_r[:, None], other=0.0)
            mrow = tl.load(m_ptr + b * ms0 + h * ms1 + qs + t_offs,
                           mask=valid_r, other=0.0)
            lrow = tl.load(l_ptr + b * ms0 + h * ms1 + qs + t_offs,
                           mask=valid_r, other=1.0)
            delta = tl.sum(dy_v.to(tl.float32) * o_v.to(tl.float32), axis=1)
            inv_l = 1.0 / lrow

            row_last = (i + 1) * BM
            if row_last > sq:
                row_last = sq
            if is_causal:
                lim = row_last - 1 + sk - sq
            else:
                lim = sk - 1
            nvalid = tl.minimum(lim, sk - 1) + 1
            if nvalid < 0:
                nvalid = 0
            j_stop = (nvalid + BN - 1) // BN

            ib128 = i // (128 // BM)

            acc = tl.zeros((BM, D), dtype=tl.float32)
            for j in range(0, j_stop):
                j_offs = j * BN + offs_bn
                col_valid = j_offs < sk
                jb128 = j // (128 // BN)
                k_v = tl.load(k_ptr + (ks + j_offs)[:, None] * ds0 + (h // G) * ds1 + offs_d[None, :],
                              mask=col_valid[:, None], other=0.0)
                v_v = tl.load(v_ptr + (ks + j_offs)[:, None] * ds0 + (h // G) * ds1 + offs_d[None, :],
                              mask=col_valid[:, None], other=0.0)

                s = tl.dot(q_v, tl.trans(k_v), out_dtype=tl.float32)
                s = s * scale
                if exact_streaming:
                    causal_keep = (offs_bn[None, :] <= (t_offs[:, None] + (sk - sq))) & valid_r[:, None]
                    edge = t_offs[:, None] + (sk - sq)
                    far = (offs_bn[None, :] >= sink_v) & \
                          (offs_bn[None, :] < (edge - (local_v - 1)))
                    st_keep = causal_keep & ~far
                    e_keep = tl.where(hmt_v == -1, st_keep, causal_keep)
                elif is_causal:
                    e_keep = (offs_bn[None, :] <= (t_offs[:, None] + (sk - sq))) & valid_r[:, None]
                else:
                    e_keep = valid_r[:, None] & col_valid[None, :]

                bit = tl.load(mask_ptr + b * msk0 + rank_v * msk1 + (ib128 * msk2 + jb128),
                              mask=(hmt_v == 1), other=1)
                ncol128 = (sk + 127) // 128
                if not exact_streaming:
                    if is_causal:
                        start_c = (sq - sk) // 128
                        if start_c < 0:
                            start_c = 0
                        dsk = sk - sq
                        if dsk < 0:
                            dsk = 0
                        mr = (dsk + 127) // 128 + 1 + (ib128 - start_c)
                    else:
                        mr = ncol128
                        start_c = 0
                    lo = mr - local_v
                    if lo < 0:
                        lo = 0
                    mr_hi = mr
                    if mr_hi > ncol128:
                        mr_hi = ncol128
                    win = (jb128 >= lo) & (jb128 < mr_hi)
                    if is_causal:
                        keep_st = (ib128 >= start_c) & (win | (jb128 < sink_v))
                    else:
                        keep_st = win | (jb128 < sink_v)
                    keep = ((hmt_v == 1).to(tl.int32) * bit.to(tl.int32) +
                            (hmt_v == -1).to(tl.int32) * keep_st.to(tl.int32) +
                            (hmt_v == 0).to(tl.int32))
                else:
                    keep = ((hmt_v == 1).to(tl.int32) * bit.to(tl.int32) +
                            (hmt_v == -1).to(tl.int32) +
                            (hmt_v == 0).to(tl.int32))

                s = tl.where(e_keep & (keep.to(tl.int1)), s, float('-inf'))
                p = tl.exp(s - mrow[:, None]) * inv_l[:, None]
                dp = tl.dot(dy_v, tl.trans(v_v), out_dtype=tl.float32)
                ds = p * (dp - delta[:, None])

                acc = tl.dot(ds.to(IN_DTYPE), k_v, acc, out_dtype=tl.float32)

            dq_base = dq_ptr + (qs + t_offs)[:, None] * ds0 + h * ds1 + offs_d[None, :]
            tl.store(dq_base, (acc * scale).to(IN_DTYPE), mask=valid_r[:, None])


@triton.jit
def _attn_bwd_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, dy_ptr, o_ptr,
    m_ptr, l_ptr, dk_ptr, dv_ptr,
    cuq_ptr, cuk_ptr,
    hmt_ptr, rank_ptr, si_ptr, mask_ptr,
    scale, H, G,
    ms0, ms1,
    msk0, msk1, msk2,
    ds0, ds1,                 # data strides (row, head)
    ncol_max,
    total_tasks,
    is_causal: tl.constexpr,
    exact_streaming: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    num_pids,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_bn = tl.arange(0, BN)
    offs_bm = tl.arange(0, BM)

    for t in range(pid, total_tasks, num_pids):
        t32 = t.to(tl.int32)
        j = t32 % ncol_max
        tmp = t32 // ncol_max
        h = tmp % H
        b = tmp // H

        qs = tl.load(cuq_ptr + b).to(tl.int32)
        qe = tl.load(cuq_ptr + b + 1).to(tl.int32)
        ks = tl.load(cuk_ptr + b).to(tl.int32)
        ke = tl.load(cuk_ptr + b + 1).to(tl.int32)
        sq = qe - qs
        sk = ke - ks

        ncol_b = (sk + BN - 1) // BN
        if ncol_b > 0:
            if j >= ncol_b:
                j = ncol_b - 1

            j_offs = j * BN + offs_bn
            col_valid = j_offs < sk
            k_v = tl.load(k_ptr + (ks + j_offs)[:, None] * ds0 + h * ds1 + offs_d[None, :],
                          mask=col_valid[:, None], other=0.0)
            v_v = tl.load(v_ptr + (ks + j_offs)[:, None] * ds0 + h * ds1 + offs_d[None, :],
                          mask=col_valid[:, None], other=0.0)

            if is_causal:
                skip = sq - sk - BM + 1
                if skip > 0:
                    i_min = (skip + BM - 1) // BM * BM
                else:
                    i_min = 0
            else:
                i_min = 0

            acc_dk = tl.zeros((BN, D), dtype=tl.float32)
            acc_dv = tl.zeros((BN, D), dtype=tl.float32)

            jb128 = j // (128 // BN)
            ncol128 = (sk + 127) // 128
            if is_causal:
                start_c = (sq - sk) // 128
                if start_c < 0:
                    start_c = 0
                dsk = sk - sq
                if dsk < 0:
                    dsk = 0
            else:
                start_c = 0
                dsk = 0

            for g in range(0, G):
                hq = h * G + g
                hmt_v = tl.load(hmt_ptr + hq)
                rank_v = tl.load(rank_ptr + hq).to(tl.int32)
                sink_v = tl.load(si_ptr + 2 * hq).to(tl.int32)
                local_v = tl.load(si_ptr + 2 * hq + 1).to(tl.int32)

                for i0 in range(i_min, sq, BM):
                    t_offs = i0 + offs_bm
                    valid_r = t_offs < sq
                    ib128 = i0 // (128 // BM)

                    mrow = tl.load(m_ptr + b * ms0 + hq * ms1 + qs + t_offs,
                                   mask=valid_r, other=0.0)
                    lrow = tl.load(l_ptr + b * ms0 + hq * ms1 + qs + t_offs,
                                   mask=valid_r, other=1.0)
                    q_v = tl.load(q_ptr + (qs + t_offs)[:, None] * ds0 + hq * ds1 + offs_d[None, :],
                                  mask=valid_r[:, None], other=0.0)
                    dy_v = tl.load(dy_ptr + (qs + t_offs)[:, None] * ds0 + hq * ds1 + offs_d[None, :],
                                   mask=valid_r[:, None], other=0.0)
                    o_v = tl.load(o_ptr + (qs + t_offs)[:, None] * ds0 + hq * ds1 + offs_d[None, :],
                                  mask=valid_r[:, None], other=0.0)
                    delta = tl.sum(dy_v.to(tl.float32) * o_v.to(tl.float32), axis=1)

                    s = tl.dot(q_v, tl.trans(k_v), out_dtype=tl.float32)
                    s = s * scale
                    if exact_streaming:
                        causal_keep = (offs_bn[None, :] <= (t_offs[:, None] + (sk - sq))) & valid_r[:, None]
                        edge = t_offs[:, None] + (sk - sq)
                        far = (offs_bn[None, :] >= sink_v) & \
                              (offs_bn[None, :] < (edge - (local_v - 1)))
                        st_keep = causal_keep & ~far
                        e_keep = tl.where(hmt_v == -1, st_keep, causal_keep)
                    elif is_causal:
                        e_keep = (offs_bn[None, :] <= (t_offs[:, None] + (sk - sq))) & valid_r[:, None]
                    else:
                        e_keep = valid_r[:, None] & col_valid[None, :]

                    bit = tl.load(mask_ptr + b * msk0 + rank_v * msk1 + (ib128 * msk2 + jb128),
                                  mask=(hmt_v == 1), other=1)
                    if not exact_streaming:
                        if is_causal:
                            mr = (dsk + 127) // 128 + 1 + (ib128 - start_c)
                        else:
                            mr = ncol128
                        lo = mr - local_v
                        if lo < 0:
                            lo = 0
                        mr_hi = mr
                        if mr_hi > ncol128:
                            mr_hi = ncol128
                        win = (jb128 >= lo) & (jb128 < mr_hi)
                        if is_causal:
                            keep_st = (ib128 >= start_c) & (win | (jb128 < sink_v))
                        else:
                            keep_st = win | (jb128 < sink_v)
                        keep = ((hmt_v == 1).to(tl.int32) * bit.to(tl.int32) +
                                (hmt_v == -1).to(tl.int32) * keep_st.to(tl.int32) +
                                (hmt_v == 0).to(tl.int32))
                    else:
                        keep = ((hmt_v == 1).to(tl.int32) * bit.to(tl.int32) +
                                (hmt_v == -1).to(tl.int32) +
                                (hmt_v == 0).to(tl.int32))

                    s = tl.where(e_keep & (keep.to(tl.int1)), s, float('-inf'))
                    p = tl.exp(s - mrow[:, None]) / lrow[:, None]
                    dp = tl.dot(dy_v, tl.trans(v_v), out_dtype=tl.float32)
                    ds = p * (dp - delta[:, None])

                    acc_dk = tl.dot(tl.trans(ds.to(IN_DTYPE)), q_v, acc_dk, out_dtype=tl.float32)
                    acc_dv = tl.dot(tl.trans(p.to(IN_DTYPE)), dy_v, acc_dv, out_dtype=tl.float32)

            dk_base = dk_ptr + (ks + j_offs)[:, None] * ds0 + h * ds1 + offs_d[None, :]
            dv_base = dv_ptr + (ks + j_offs)[:, None] * ds0 + h * ds1 + offs_d[None, :]
            tl.store(dk_base, (acc_dk * scale).to(IN_DTYPE), mask=col_valid[:, None])
            tl.store(dv_base, acc_dv.to(IN_DTYPE), mask=col_valid[:, None])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                streaming_info, base_blockmask, dout,
                softmax_max, softmax_sum, attention_in,
                softmax_scale, is_causal, exact_streaming):
        total_q, H, D = q.shape
        total_k, HK, _ = k.shape
        B = cu_seqlens_q.shape[0] - 1
        G = H // HK
        dtype = q.dtype
        scale = float(softmax_scale) if softmax_scale is not None else float(D) ** -0.5

        dev = q.device
        cores = _vec_core_num(dev)

        is_causal_f = bool(is_causal)
        exact_f = bool(exact_streaming) and is_causal_f

        BM = 32
        BN = 32

        rank = torch.empty(H, dtype=torch.int32, device=dev)
        nr = H if H < cores else cores
        grid_r = (nr,)
        _bs_rank_kernel[grid_r](head_mask_type, rank, H, grid_r[0], num_warps=1)

        dq = torch.empty_like(q)
        dk = torch.empty(total_k, HK, D, dtype=dtype, device=dev)
        dv = torch.empty(total_k, HK, D, dtype=dtype, device=dev)

        ds0 = H * D
        ds1 = D
        ms0 = H * softmax_max.shape[2]
        ms1 = softmax_max.shape[2]
        mb = base_blockmask.shape
        msk0 = mb[1] * mb[2] * mb[3]
        msk1 = mb[2] * mb[3]
        msk2 = mb[3]

        nrow_max = (total_q + BM - 1) // BM
        ncol_max = (total_k + BN - 1) // BN

        in_dt = tl.float16 if dtype == torch.float16 else tl.bfloat16

        tasks_dq = B * H * nrow_max
        gdq = tasks_dq if tasks_dq < cores else cores
        grid_dq = (gdq,)
        _attn_bwd_dq_kernel[grid_dq](
            q, k, v, dout, attention_in,
            softmax_max, softmax_sum, dq,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type, rank, streaming_info, base_blockmask,
            scale, H, G,
            ms0, ms1,
            msk0, msk1, msk2,
            ds0, ds1,
            nrow_max, tasks_dq,
            is_causal_f, exact_f,
            D, BM, BN, in_dt,
            grid_dq[0],
        )

        tasks_dk = B * HK * ncol_max
        gdk = tasks_dk if tasks_dk < cores else cores
        grid_dk = (gdk,)
        _attn_bwd_dkdv_kernel[grid_dk](
            q, k, v, dout, attention_in,
            softmax_max, softmax_sum, dk, dv,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type, rank, streaming_info, base_blockmask,
            scale, HK, G,
            ms0, ms1,
            msk0, msk1, msk2,
            ds0, ds1,
            ncol_max, tasks_dk,
            is_causal_f, exact_f,
            D, BM, BN, in_dt,
            grid_dk[0],
        )
        return dq, dk, dv
