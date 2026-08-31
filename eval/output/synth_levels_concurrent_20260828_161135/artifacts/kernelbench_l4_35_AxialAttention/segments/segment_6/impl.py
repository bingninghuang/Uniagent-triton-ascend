import torch
import triton
import triton.language as tl


@triton.jit
def axial_qkv_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    M, S, T, HW, C, W_ROW_OFF, W_ROWS,
    num_cores: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
    E: tl.constexpr, D: tl.constexpr, DPAD: tl.constexpr,
    HSTR: tl.constexpr, WSTR: tl.constexpr,
):
    """out[r, e*D + d] = sum_c x2[row(r), c] * W[W_ROW_OFF + e*D + d, c]

    x2: (MP, C) channel-last, rows in (b, h, w) order (MP >= M).
    A-row r: g = r // S, s = r % S, b = g // T, o = g % T.
    H-axis: s = h, o = w  -> row_x = b*HW + s*W + o   (HSTR=W, WSTR=1)
    W-axis: s = w, o = h  -> row_x = b*HW + o*W + s   (HSTR=1, WSTR=W)
    """
    pid = tl.program_id(0).to(tl.int32)
    m_tiles = tl.cdiv(M, BM)
    n_blocks = m_tiles * E
    for blk in range(pid, n_blocks, num_cores):
        m_t = blk // E
        e = blk % E
        offs_m = m_t * BM + tl.arange(0, BM).to(tl.int32)
        valid_m = offs_m < M
        g = offs_m // S
        s_m = offs_m % S
        b_m = g // T
        o_m = g % T
        b_row = tl.where(valid_m, b_m, 0)
        row_x = b_row * HW + s_m * HSTR + o_m * WSTR
        d_off = tl.arange(0, DPAD).to(tl.int32)
        j_off = e * D + d_off
        acc = tl.zeros((BM, DPAD), dtype=tl.float32)
        for k0 in range(0, C, BK):
            offs_k = k0 + tl.arange(0, BK).to(tl.int32)
            a = tl.load(x_ptr + row_x[:, None] * C + offs_k[None, :],
                        mask=valid_m[:, None], other=0.0)
            if DPAD != D:
                b_t = tl.load(w_ptr + (W_ROW_OFF + j_off)[None, :] * C + offs_k[:, None],
                              mask=(j_off < W_ROWS)[None, :], other=0.0)
            else:
                b_t = tl.load(w_ptr + (W_ROW_OFF + j_off)[None, :] * C + offs_k[:, None])
            acc = tl.dot(a, b_t, acc, out_dtype=tl.float32)
        out_offs = offs_m[:, None] * (E * D) + j_off[None, :]
        if DPAD != D:
            tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty),
                     mask=(d_off < D)[None, :])
        else:
            tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def axial_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    N, S,
    qk_scale,
    num_cores: tl.constexpr,
    E: tl.constexpr, D: tl.constexpr,
    DPAD: tl.constexpr, SPAD: tl.constexpr,
):
    """For each (group n, head e): softmax(q @ k^T * scale) @ v.

    q/k/v layout: (M, E*D) row-major; group n rows r = n*S + s (r < M only).
    Emulates intermediate input-dtype rounding of the torch reference.
    """
    pid = tl.program_id(0).to(tl.int32)
    n_blocks = N * E
    for blk in range(pid, n_blocks, num_cores):
        n = blk // E
        e = blk % E
        s_off = tl.arange(0, SPAD).to(tl.int32)
        d_off = tl.arange(0, DPAD).to(tl.int32)
        row_base = (n * S + s_off) * (E * D) + e * D
        s_valid = s_off < S
        d_valid = d_off < D
        q = tl.load(q_ptr + row_base[:, None] + d_off[None, :],
                    mask=s_valid[:, None] & d_valid[None, :], other=0.0)
        # k transposed: kt[d, t] = k[n, t, e, d]
        kt = tl.load(k_ptr + row_base[None, :] + d_off[:, None],
                     mask=s_valid[None, :] & d_valid[:, None], other=0.0)
        v = tl.load(v_ptr + row_base[:, None] + d_off[None, :],
                    mask=s_valid[:, None] & d_valid[None, :], other=0.0)
        scores = tl.dot(q, kt, out_dtype=tl.float32)
        in_dt = o_ptr.dtype.element_ty
        s16 = scores.to(in_dt)
        scaled = (s16.to(tl.float32) * qk_scale).to(in_dt)
        sf = scaled.to(tl.float32)
        col_valid = s_valid[None, :]
        sf = tl.where(col_valid, sf, float("-inf"))
        mx = tl.max(sf, axis=1)[:, None]
        ex = tl.exp(sf - mx)
        ex = tl.where(col_valid, ex, 0.0)
        l = tl.sum(ex, axis=1)[:, None]
        attn = (ex / l).to(in_dt)
        o = tl.dot(attn, v, out_dtype=tl.float32)
        tl.store(o_ptr + row_base[:, None] + d_off[None, :], o.to(in_dt),
                 mask=s_valid[:, None] & d_valid[None, :])


@triton.jit
def axial_out_gemm_kernel(
    a_ptr, w_ptr, bias_ptr, out_ptr,
    M, S, T, HH, WW, C,
    num_cores: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
    E: tl.constexpr, D: tl.constexpr,
    HSTR: tl.constexpr, WSTR: tl.constexpr,
    IS_ADD: tl.constexpr,
):
    """out[b, c, h, w] = (A @ W^T)[r, c] + bias[c] (+ existing).

    A: (MP, E*D) row-major attention outputs (rows in axis spatial order;
    masked rows r >= M are never stored). out: (B, C, HH, WW) contiguous.
    """
    pid = tl.program_id(0).to(tl.int32)
    inner = E * D
    m_tiles = tl.cdiv(M, BM)
    c_tiles = tl.cdiv(C, BN)
    n_blocks = m_tiles * c_tiles
    for blk in range(pid, n_blocks, num_cores):
        m_t = blk // c_tiles
        c_t = blk % c_tiles
        offs_m = m_t * BM + tl.arange(0, BM).to(tl.int32)
        valid_m = offs_m < M
        g = offs_m // S
        s_m = offs_m % S
        b_m = g // T
        o_m = g % T
        i_pos = b_m * (HH * WW) + s_m * HSTR + o_m * WSTR
        row_out = tl.where(valid_m, b_m * (C * (HH * WW)) + i_pos, 0)
        offs_c = c_t * BN + tl.arange(0, BN).to(tl.int32)
        valid_c = offs_c < C
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, inner, BK):
            offs_k = k0 + tl.arange(0, BK).to(tl.int32)
            a = tl.load(a_ptr + offs_m[:, None] * inner + offs_k[None, :])
            b_t = tl.load(w_ptr + offs_c[None, :] * inner + offs_k[:, None],
                          mask=valid_c[None, :], other=0.0)
            acc = tl.dot(a, b_t, acc, out_dtype=tl.float32)
        bias = tl.load(bias_ptr + offs_c, mask=valid_c, other=0.0)
        res = acc + bias.to(tl.float32)[None, :]
        out_offs = row_out[:, None] + offs_c[None, :] * (HH * WW)
        store_mask = valid_m[:, None] & valid_c[None, :]
        if IS_ADD:
            cur = tl.load(out_ptr + out_offs, mask=store_mask, other=0.0)
            res = res + cur.to(tl.float32)
        tl.store(out_ptr + out_offs, res.to(out_ptr.dtype.element_ty),
                 mask=store_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch_npu
            lim = torch_npu.npu.npu_config.get_device_limit(0)
            self.vec_cores = lim.get("vector_core_num", 48)
            self.cube_cores = lim.get("cube_core_num", 24)
        except Exception:
            self.vec_cores = 48
            self.cube_cores = 24

    def forward(self, x, dim, heads, dim_heads,
                to_q_w, to_q_b, to_kv_w, to_kv_b, to_out_w, to_out_b):
        b, c, hh, ww = x.shape
        if dim is None:
            dim = c
        if heads is None:
            heads = 8
        head_dim = (dim // heads) if dim_heads is None else dim_heads
        inner_dim = head_dim * heads

        dev = x.device
        dt = x.dtype
        M = b * hh * ww
        E = int(heads)
        D = int(head_dim)
        DPAD = max(triton.next_power_of_2(D), 16)
        SPAD_h = max(triton.next_power_of_2(hh), 16)
        SPAD_w = max(triton.next_power_of_2(ww), 16)

        BM = 32
        BK = 16
        BN = 32
        MP = (M + BM - 1) // BM * BM
        qk_scale = float(D ** -0.5)

        # channel-last (b, h, w, c) view of x, padded to MP rows
        x2 = torch.empty((MP, c), device=dev, dtype=dt)
        x2[:M] = x.permute(0, 2, 3, 1).reshape(M, c)

        q_h = torch.empty((MP, inner_dim), device=dev, dtype=dt)
        k_h = torch.empty((MP, inner_dim), device=dev, dtype=dt)
        v_h = torch.empty((MP, inner_dim), device=dev, dtype=dt)
        ao_h = torch.empty((MP, inner_dim), device=dev, dtype=dt)
        q_w = torch.empty((MP, inner_dim), device=dev, dtype=dt)
        k_w = torch.empty((MP, inner_dim), device=dev, dtype=dt)
        v_w = torch.empty((MP, inner_dim), device=dev, dtype=dt)
        ao_w = torch.empty((MP, inner_dim), device=dev, dtype=dt)
        out = torch.empty((b, dim, hh, ww), device=dev, dtype=dt)

        cores = self.cube_cores
        grid = (cores,)

        n_h = b * ww
        # ---- height axis: S = hh (h), T = ww (w) ----
        axial_qkv_gemm_kernel[grid](
            x2, to_q_w, q_h, M, hh, ww, hh * ww, c, 0, inner_dim,
            cores, BM=BM, BK=BK,
            E=E, D=D, DPAD=DPAD, HSTR=ww, WSTR=1,
        )
        axial_qkv_gemm_kernel[grid](
            x2, to_kv_w, k_h, M, hh, ww, hh * ww, c, 0, 2 * inner_dim,
            cores, BM=BM, BK=BK,
            E=E, D=D, DPAD=DPAD, HSTR=ww, WSTR=1,
        )
        axial_qkv_gemm_kernel[grid](
            x2, to_kv_w, v_h, M, hh, ww, hh * ww, c, inner_dim,
            2 * inner_dim, cores, BM=BM, BK=BK,
            E=E, D=D, DPAD=DPAD, HSTR=ww, WSTR=1,
        )
        axial_attn_kernel[grid](
            q_h, k_h, v_h, ao_h, n_h, hh, qk_scale, cores,
            E=E, D=D, DPAD=DPAD, SPAD=SPAD_h,
        )

        n_w = b * hh
        # ---- width axis: S = ww (w), T = hh (h) ----
        axial_qkv_gemm_kernel[grid](
            x2, to_q_w, q_w, M, ww, hh, hh * ww, c, 0, inner_dim,
            cores, BM=BM, BK=BK,
            E=E, D=D, DPAD=DPAD, HSTR=1, WSTR=ww,
        )
        axial_qkv_gemm_kernel[grid](
            x2, to_kv_w, k_w, M, ww, hh, hh * ww, c, 0, 2 * inner_dim,
            cores, BM=BM, BK=BK,
            E=E, D=D, DPAD=DPAD, HSTR=1, WSTR=ww,
        )
        axial_qkv_gemm_kernel[grid](
            x2, to_kv_w, v_w, M, ww, hh, hh * ww, c, inner_dim,
            2 * inner_dim, cores, BM=BM, BK=BK,
            E=E, D=D, DPAD=DPAD, HSTR=1, WSTR=ww,
        )
        axial_attn_kernel[grid](
            q_w, k_w, v_w, ao_w, n_w, ww, qk_scale, cores,
            E=E, D=D, DPAD=DPAD, SPAD=SPAD_w,
        )

        # ---- out projection (+ bias); second pass adds to first ----
        axial_out_gemm_kernel[grid](
            ao_h, to_out_w, to_out_b, out, M, hh, ww, hh, ww, dim,
            cores, BM=BM, BK=BK, BN=BN, E=E, D=D,
            HSTR=ww, WSTR=1, IS_ADD=False,
        )
        axial_out_gemm_kernel[grid](
            ao_w, to_out_w, to_out_b, out, M, ww, hh, hh, ww, dim,
            cores, BM=BM, BK=BK, BN=BN, E=E, D=D,
            HSTR=1, WSTR=ww, IS_ADD=True,
        )
        return out
