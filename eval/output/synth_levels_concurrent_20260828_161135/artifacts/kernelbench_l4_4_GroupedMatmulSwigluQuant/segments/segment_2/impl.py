import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _k_gemm_swiglu(
    x_ptr, w_ptr, ws_ptr, xs_ptr,
    out_ptr, rowmax_ptr,
    M, N2, K,
    stride_wk,  # weight row stride (= full n)
    stride_out,  # out row stride (= N2)
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int32)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N2, BN)
    num_blocks = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n

    offs_m = tl.arange(0, BM).to(tl.int32)
    offs_n = tl.arange(0, BN).to(tl.int32)
    offs_k = tl.arange(0, BK).to(tl.int32)

    for pid in range(pid0, num_blocks, NUM_CORES):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        rows = pid_m * BM + offs_m
        cols = pid_n * BN + offs_n
        row_ok = rows < M
        col_ok = cols < N2

        x_base = x_ptr + rows[:, None] * K

        acc_gate = tl.zeros((BM, BN), dtype=tl.int32)
        acc_up = tl.zeros((BM, BN), dtype=tl.int32)

        xs_block = tl.load(xs_ptr + rows, mask=row_ok, other=0.0)

        for start in range(0, K, BK):
            k_valid = start + offs_k < K
            a = tl.load(x_base + (start + offs_k)[None, :],
                        mask=row_ok[:, None] & k_valid[None, :],
                        other=0)
            w_row = w_ptr + (start + offs_k)[:, None] * stride_wk
            b_gate = tl.load(w_row + offs_n[None, :], mask=k_valid[:, None], other=0)
            b_up = tl.load(w_row + N2 + offs_n[None, :], mask=k_valid[:, None], other=0)
            acc_gate = tl.dot(a, b_gate, acc_gate, out_dtype=tl.int32)
            acc_up = tl.dot(a, b_up, acc_up, out_dtype=tl.int32)

        ws_g = tl.load(ws_ptr + cols, mask=col_ok, other=0.0)
        ws_u = tl.load(ws_ptr + N2 + cols, mask=col_ok, other=0.0)

        vg = acc_gate.to(tl.float32) * ws_g[None, :]
        vg = vg.to(tl.float16).to(tl.float32)
        vg = vg * xs_block[:, None]
        vg = vg.to(tl.float16).to(tl.float32)

        vu = acc_up.to(tl.float32) * ws_u[None, :]
        vu = vu.to(tl.float16).to(tl.float32)
        vu = vu * xs_block[:, None]
        vu = vu.to(tl.float16).to(tl.float32)

        sig = tl.sigmoid(vg)
        e1 = (vg * sig).to(tl.float16).to(tl.float32)
        swi = (e1 * vu).to(tl.float16)

        out0 = out_ptr + rows[:, None] * stride_out + cols[None, :]
        tl.store(out0, swi, mask=row_ok[:, None] & col_ok[None, :])

        rmax = tl.max(tl.abs(swi.to(tl.float32)), axis=1)
        tl.atomic_max(rowmax_ptr + rows, rmax, mask=row_ok)


@triton.jit
def _k_quant(
    tmp_ptr, rowmax_ptr, q_ptr, qs_ptr,
    M, N2,
    stride_out,
    BC: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    for i in range(pid, M, NUM_CORES):
        rm = tl.load(rowmax_ptr + i)
        rm = tl.maximum(rm, 1.0e-9)
        qs = rm / 127.0
        tl.store(qs_ptr + i, qs)
        factor = 127.0 / rm
        row_tmp = tmp_ptr + i.to(tl.int64) * stride_out
        row_q = q_ptr + i.to(tl.int64) * stride_out
        for c0 in range(0, N2, BC):
            offs = c0 + tl.arange(0, BC).to(tl.int32)
            col_ok = offs < N2
            v = tl.load(row_tmp + offs, mask=col_ok, other=0.0).to(tl.float32)
            qv = tl.floor(v * factor + 0.5)
            qv = tl.minimum(tl.maximum(qv, -128.0), 127.0)
            tl.store(row_q + offs, qv.to(tl.int8), mask=col_ok)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cube_cores = 24
        self.vec_cores = 48
        try:
            import torch_npu
            limit = torch_npu.npu.npu_config.get_device_limit(0)
            self.cube_cores = int(limit.get("cube_core_num", 24))
            self.vec_cores = int(limit.get("vector_core_num", 48))
        except Exception:
            pass
        self.BM = 128
        self.BN = 128
        self.BK = 128
        self.GROUP_M = 8
        self.BC = 512

    def forward(self, x: torch.Tensor, weight: list, weight_scale: list,
                x_scale: torch.Tensor, group_list: torch.Tensor):
        device = x.device
        if isinstance(weight, (list, tuple)):
            w_list = [w.to(device) for w in weight]
            ws_list = [ws.to(device) for ws in weight_scale]
        else:
            w_list = [weight.to(device)]
            ws_list = [weight_scale.to(device)]
        e = len(w_list)
        x_scale = x_scale.to(device)

        if isinstance(group_list, (list, tuple)):
            groups = [int(v) for v in group_list]
        else:
            groups = [int(v) for v in group_list.to(device).tolist()]

        m = x.shape[0]
        k = int(w_list[0].shape[-2])
        n = int(w_list[0].shape[-1])
        n2 = n // 2

        if not x.is_contiguous():
            x = x.contiguous()
        if not x_scale.is_contiguous():
            x_scale = x_scale.contiguous()

        tmp = torch.empty(m, n2, dtype=torch.float16, device=device)
        rowmax = torch.zeros(m, dtype=torch.float32, device=device)
        q_out = torch.empty(m, n2, dtype=torch.int8, device=device)
        q_scale = torch.empty(m, dtype=torch.float32, device=device)

        start = 0
        for gi in range(e):
            bound = groups[gi] if gi < len(groups) else m
            m_g = bound - start
            if m_g > 0:
                w_i = w_list[gi]
                if w_i.dtype != torch.int8:
                    w_i = w_i.to(torch.int8)
                if not w_i.is_contiguous():
                    w_i = w_i.contiguous()
                ws_i = ws_list[gi]
                if not ws_i.is_contiguous():
                    ws_i = ws_i.contiguous()

                num_pid_m = triton.cdiv(m_g, self.BM)
                num_pid_n = triton.cdiv(n2, self.BN)
                grid_m = min(num_pid_m * num_pid_n, self.cube_cores)
                _k_gemm_swiglu[(grid_m,)](
                    x[start:start + m_g], w_i, ws_i, x_scale[start:start + m_g],
                    tmp[start:start + m_g], rowmax[start:start + m_g],
                    m_g, n2, k,
                    w_i.stride(0), n2,
                    BM=self.BM, BN=self.BN, BK=self.BK,
                    GROUP_M=self.GROUP_M, NUM_CORES=self.cube_cores,
                )
            start = bound

        grid_q = min(m, self.vec_cores)
        _k_quant[(grid_q,)](
            tmp, rowmax, q_out, q_scale,
            m, n2, n2,
            BC=self.BC, NUM_CORES=self.vec_cores,
        )
        return q_out, q_scale
