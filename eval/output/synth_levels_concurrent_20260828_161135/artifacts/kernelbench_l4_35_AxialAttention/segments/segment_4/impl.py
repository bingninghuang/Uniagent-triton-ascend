import torch
import triton
import triton.language as tl


@triton.jit
def axial_qkv_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    M, S, T, HH, WW, C, W_ROW_OFF, W_ROWS,
    num_cores: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
    E: tl.constexpr, D: tl.constexpr, DPAD: tl.constexpr,
    SU: tl.constexpr, SO: tl.constexpr,
):
    """out[r, e*D+jd] = sum_c A[r, c] * W[W_ROW_OFF + e*D + jd, c]

    row r of the "sequence A matrix" maps to x[b, c, h, w] via
      g = r // S, s = r % S, b = g // T, o = g % T
      x_flat offset = b*(C*HH*WW) + s*SU + o*SO + c
    """
    pid = tl.program_id(0).to(tl.int32)
    m_tiles = tl.cdiv(M, BM)
    n_blocks = m_tiles * E
    for blk in range(pid, n_blocks, num_cores):
        m_t = blk // E
        e = blk % E
        offs_m = m_t * BM + tl.arange(0, BM)
        valid_m = offs_m < M
        g = offs_m // S
        s_m = offs_m % S
        b_m = g // T
        o_m = g % T
        base_m = b_m * (C * HH * WW) + s_m * SU + o_m * SO
        d_off = tl.arange(0, DPAD)
        j_off = e * D + d_off
        w_ok = (W_ROW_OFF + j_off) < W_ROWS
        w_base = w_ptr + (W_ROW_OFF + j_off) * C
        acc = tl.zeros((BM, DPAD), dtype=tl.float32)
        for k0 in range(0, C, BK):
            offs_k = k0 + tl.arange(0, BK)
            a = tl.load(x_ptr + base_m[:, None] + offs_k[None, :],
                        mask=valid_m[:, None], other=0.0)
            b_tile = tl.load(w_base[None, :] + offs_k[:, None],
                             mask=w_ok[None, :], other=0.0)
            acc = tl.dot(a, b_tile, acc, out_dtype=tl.float32)
        out_offs = offs_m[:, None] * (E * D) + j_off[None, :]
        res_mask = valid_m[:, None] & (d_off[None, :] < D)
        tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=res_mask)


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

    q/k/v layout: (N, S, E, D); row r = n*S+s base = r*(E*D) + e*D.
    Emulates intermediate input-dtype rounding of the torch reference.
    """
    pid = tl.program_id(0).to(tl.int32)
    n_blocks = N * E
    for blk in range(pid, n_blocks, num_cores):
        n = blk // E
        e = blk % E
        s_off = tl.arange(0, SPAD)
        d_off = tl.arange(0, DPAD)
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
        # emulate: matmul materialized in in_dtype, then scaled, materialized again
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
    SU: tl.constexpr, SO: tl.constexpr,
    IS_ADD: tl.constexpr,
):
    """out[b, c, h, w] = (A @ to_out_w^T)[row, c] + bias[c] (+ existing).

    A is (M, E*D) row-major (attention outputs concatenated per position).
    """
    pid = tl.program_id(0).to(tl.int32)
    inner = E * D
    m_tiles = tl.cdiv(M, BM)
    c_tiles = tl.cdiv(C, BN)
    n_blocks = m_tiles * c_tiles
    for blk in range(pid, n_blocks, num_cores):
        m_t = blk // c_tiles
        c_t = blk % c_tiles
        offs_m = m_t * BM + tl.arange(0, BM)
        valid_m = offs_m < M
        g = offs_m // S
        s_m = offs_m % S
        b_m = g // T
        o_m = g % T
        i_pos = b_m * (HH * WW) + s_m * SU + o_m * SO
        out_base = (i_pos // (HH * WW)) * (C * HH * WW) + (i_pos % (HH * WW))
        offs_c = c_t * BN + tl.arange(0, BN)
        valid_c = offs_c < C
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, inner, BK):
            offs_k = k0 + tl.arange(0, BK)
            a = tl.load(a_ptr + offs_m[:, None] * inner + offs_k[None, :],
                        mask=valid_m[:, None], other=0.0)
            b_tile = tl.load(w_ptr + offs_c[None, :] * inner + offs_k[:, None],
                             mask=valid_c[None, :], other=0.0)
            acc = tl.dot(a, b_tile, acc, out_dtype=tl.float32)
        bias = tl.load(bias_ptr + offs_c, mask=valid_c, other=0.0)
        res = (acc + bias.to(tl.float32)[None, :]).to(out_ptr.dtype.element_ty)
        out_offs = out_base[:, None] + offs_c[None, :] * (HH * WW)
        store_mask = valid_m[:, None] & valid_c[None, :]
        if IS_ADD:
            cur = tl.load(out_ptr + out_offs, mask=store_mask, other=0.0)
            res = (res.to(tl.float32) + cur.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + out_offs, res, mask=store_mask)


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
        DPAD = 1 << max(4, (D - 1).bit_length())
        SPAD_h = 1 << max(4, (hh - 1).bit_length())
        SPAD_w = 1 << max(4, (ww - 1).bit_length())

        BM = 64
        BK = 32
        BK_o = 32
        BN = 64
        qk_scale = float(head_dim ** -0.5)

        q_h = torch.empty((M, inner_dim), device=dev, dtype=dt)
        k_h = torch.empty((M, inner_dim), device=dev, dtype=dt)
        v_h = torch.empty((M, inner_dim), device=dev, dtype=dt)
        ao_h = torch.empty((M, inner_dim), device=dev, dtype=dt)
        q_w = torch.empty((M, inner_dim), device=dev, dtype=dt)
        k_w = torch.empty((M, inner_dim), device=dev, dtype=dt)
        v_w = torch.empty((M, inner_dim), device=dev, dtype=dt)
        ao_w = torch.empty((M, inner_dim), device=dev, dtype=dt)
        out = torch.empty((b, dim, hh, ww), device=dev, dtype=dt)

        grid = (self.cube_cores,)

        n_h = b * ww
        # ---- height axis: sequences over h (S=hh, T=ww, SU=ww, SO=1) ----
        axial_qkv_gemm_kernel[grid](
            x, to_q_w, q_h, M, hh, ww, hh, ww, c, 0, inner_dim,
            self.cube_cores, BM=BM, BK=BK,
            E=E, D=D, DPAD=DPAD, SU=ww, SO=1,
        )
        axial_qkv_gemm_kernel[grid](
            x, to_kv_w, k_h, M, hh, ww, hh, ww, c, 0, 2 * inner_dim,
            self.cube_cores, BM=BM, BK=BK,
            E=E, D=D, DPAD=DPAD, SU=ww, SO=1,
        )
        axial_qkv_gemm_kernel[grid](
            x, to_kv_w, v_h, M, hh, ww, hh, ww, c, inner_dim,
            2 * inner_dim, self.cube_cores,
            BM=BM, BK=BK, E=E, D=D, DPAD=DPAD, SU=ww, SO=1,
        )
        axial_attn_kernel[grid](
            q_h, k_h, v_h, ao_h, n_h, hh, qk_scale, self.cube_cores,
            E=E, D=D, DPAD=DPAD, SPAD=SPAD_h,
        )

        n_w = b * hh
        # ---- width axis: sequences over w (S=ww, T=hh, SU=1, SO=ww) ----
        axial_qkv_gemm_kernel[grid](
            x, to_q_w, q_w, M, ww, hh, hh, ww, c, 0, inner_dim,
            self.cube_cores,
            BM=BM, BK=BK, E=E, D=D, DPAD=DPAD, SU=1, SO=ww,
        )
        axial_qkv_gemm_kernel[grid](
            x, to_kv_w, k_w, M, ww, hh, hh, ww, c, 0, 2 * inner_dim,
            self.cube_cores,
            BM=BM, BK=BK, E=E, D=D, DPAD=DPAD, SU=1, SO=ww,
        )
        axial_qkv_gemm_kernel[grid](
            x, to_kv_w, v_w, M, ww, hh, hh, ww, c, inner_dim,
            2 * inner_dim, self.cube_cores,
            BM=BM, BK=BK, E=E, D=D, DPAD=DPAD, SU=1, SO=ww,
        )
        axial_attn_kernel[grid](
            q_w, k_w, v_w, ao_w, n_w, ww, qk_scale, self.cube_cores,
            E=E, D=D, DPAD=DPAD, SPAD=SPAD_w,
        )

        # ---- out projection (+ bias); second pass adds to first ----
        axial_out_gemm_kernel[grid](
            ao_h, to_out_w, to_out_b, out, M, hh, ww, hh, ww, dim,
            self.cube_cores, BM=BM, BK=BK_o, BN=BN, E=E, D=D, SU=ww, SO=1,
            IS_ADD=False,
        )
        axial_out_gemm_kernel[grid](
            ao_w, to_out_w, to_out_b, out, M, ww, hh, hh, ww, dim,
            self.cube_cores, BM=BM, BK=BK_o, BN=BN, E=E, D=D, SU=1, SO=ww,
            IS_ADD=True,
        )
        return out
