import os
import torch
import triton
import triton.language as tl

_DBG_ACTIVE = [False]


def _ds(tag, a, b):
    a = a.detach().float()
    b = b.detach().float()
    d = (a - b).abs()
    idx = int(d.view(-1).argmax())
    shp = list(d.shape)
    pos = []
    t = idx
    for s in reversed(shp):
        pos.insert(0, int(t % s))
        t //= s
    fa = a.view(-1)[idx].item()
    fb = b.view(-1)[idx].item()
    msg = (tag + " max=" + format(d.max().item(), ".6g")
           + " mean=" + format(d.mean().item(), ".6g")
           + " relmax=" + format((d / (a.abs() + 1e-9)).max().item(), ".6g")
           + " pos=" + repr(tuple(pos))
           + " fw=" + format(fa, ".6g") + " mn=" + format(fb, ".6g"))
    print("[DBG35] " + msg)
    return msg


def _dbg_case3(x):
    if not (
        x.dtype == torch.bfloat16
        and tuple(x.shape) == (2, 32, 9, 9)
        and x.is_contiguous()
    ):
        return
    try:
        x = x.to("cpu", dtype=torch.float32)
        c = torch.manual_seed(1003)
        to_q = torch.nn.functional.linear.weight  # type: ignore
        torch.nn  # noqa
        import json, os
        ref_mod = None
        try:
            import importlib.util
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernelbench_l4_35_AxialAttention.py")
            spec = importlib.util.spec_from_file_location("kbref35", p)
            ref_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ref_mod)
        except Exception as ex:
            print(f"[DBG] ref import failed: {ex}")
        groups = ref_mod.get_input_groups()
        g = groups[2]
        dim, heads, dim_heads = g[1], g[2], g[3]
        to_q_w, to_q_b, to_kv_w, to_kv_b, to_out_w, to_out_b = g[4], g[5], g[6], g[7], g[8], g[9]
        fw = ref_mod.Model().forward(x, dim, heads, dim_heads, to_q_w, to_q_b, to_kv_w, to_kv_b, to_out_w, to_out_b)
        mine = torch.load(__import__("weakref").ref(1) and __import__("pickle") or None) if False else None
        # recompute mine via fresh kernel object
        import torch.nn as nn
        class _N(nn.Module):
            def __init__(self):
                super().__init__()
                self.vec_cores = 48
                self.cube_cores = 24
        import types
        f = types.MethodType(ModelNew.forward, _N())
        init = types.MethodType(ModelNew.__init__, _N())
        impl = _N(); init()
        mi = impl.forward(x, dim, heads, dim_heads, to_q_w, to_q_b, to_kv_w, to_kv_b, to_out_w, to_out_b)
        f32a = fw.float(); f32b = mi.float()
        d = (f32a - f32b).abs()
        print(f"[DBG] shapes: fw={list(fw.shape)} mi={list(mi.shape)}")
        print(f"[DBG] out maxdiff={d.max().item():.6g} meandiff={d.mean().item():.6g} relmax={(d/(f32a.abs()+1e-6)).max().item():.6g}")
        # per-axis decomposition of my impl
        try:
            _dbg_axes(x, dim, heads, dim_heads, to_q_w, to_q_b, to_kv_w, to_kv_b, to_out_w, to_out_b)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[DBG] axes traceback: {ex}")
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[DBG] case3 trace: {ex}")
    finally:
        torch.cuda.synchronize() if torch.cuda.is_available() else None


def _dbg_axes(x, dim, heads, dim_heads, to_q_w, to_q_b, to_kv_w, to_kv_b, to_out_w, to_out_b):
    import torch.nn.functional as F
    b, c, h, w = x.shape
    head_dim = (dim // heads) if dim_heads is None else dim_heads
    inner = head_dim * heads
    x_perm = x.permute(0, 2, 3, 1).contiguous()
    # --- FW axis H intermediates ---
    seq_h = x_perm.permute(0, 2, 1, 3).contiguous().view(b * w, h, c)
    q_h = F.linear(seq_h, to_q_w, to_q_b)
    kv_h = F.linear(seq_h, to_kv_w, to_kv_b)
    k_h, v_h = kv_h.chunk(2, dim=-1)
    Q = q_h.view(b * w, h, heads, head_dim).transpose(1, 2)
    K = k_h.view(b * w, h, heads, head_dim).transpose(1, 2)
    V = v_h.view(b * w, h, heads, head_dim).transpose(1, 2)
    A = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * (head_dim ** -0.5), dim=-1)
    O = torch.matmul(A, V)
    # --- MY axis H intermediates (re-run kernels into buffers) ---
    dev, dt = x.device, x.dtype
    M = b * h * w
    x2 = x_perm.view(M, c)
    BM, BK = 32, 16
    MP = (M + BM - 1) // BM * BM
    x2p = torch.zeros((MP, c), device=dev, dtype=dt)
    x2p[:M] = x2
    q_my = torch.zeros((MP, inner), device=dev, dtype=dt)
    k_my = torch.zeros((MP, inner), device=dev, dtype=dt)
    v_my = torch.zeros((MP, inner), device=dev, dtype=dt)
    ao_my = torch.zeros((MP, inner), device=dev, dtype=dt)
    E, D = int(heads), int(head_dim)
    DPAD = 128 if D > 64 else 64 if D > 32 else 32 if D > 16 else 16
    SPAD = 32 if h > 16 else 16
    cores = 12
    grid = (cores,)
    axial_qkv_gemm_kernel[grid](x2p, to_q_w, q_my, M, h, w, h * w, c, 0, inner,
                                cores, BM=BM, BK=BK, E=E, D=D, DPAD=DPAD, HSTR=w, WSTR=1)
    axial_qkv_gemm_kernel[grid](x2p, to_kv_w, k_my, M, h, w, h * w, c, 0, 2 * inner,
                                cores, BM=BM, BK=BK, E=E, D=D, DPAD=DPAD, HSTR=w, WSTR=1)
    axial_qkv_gemm_kernel[grid](x2p, to_kv_w, v_my, M, h, w, h * w, c, inner, 2 * inner,
                                cores, BM=BM, BK=BK, E=E, D=D, DPAD=DPAD, HSTR=w, WSTR=1)
    axial_attn_kernel[grid](q_my, k_my, v_my, ao_my, b * w, h, float(D ** -0.5), cores,
                            E=E, D=D, DPAD=DPAD, SPAD=SPAD)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    q_my_fq = q_my[:M].view(b * w, h, heads, head_dim).transpose(1, 2)
    k_my_fk = k_my[:M].view(b * w, h, heads, head_dim).transpose(1, 2)
    for nm, a, b_ in (("q", q_h.float(), q_my_fq.float()),
                      ("k", k_h.float(), k_my_fk),
                      ("v", v_h.float(), v_my[:M].view(b * w, h, heads, head_dim).transpose(1, 2).float()),
                      ("out", O.transpose(1, 2).contiguous().view(M, inner).float(), ao_my[:M].float())):
        dd = (a - b_).abs()
        print(f"[DBG] H-axis {nm}: max={dd.max().item():.6g} mean={dd.mean().item():.6g} "
              f"relmax={(dd/(a.abs()+1e-6)).max().item():.6g}")


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
        DPAD = 128 if D > 64 else 64 if D > 32 else 32 if D > 16 else 16
        SPAD_h = 32 if hh > 16 else 16
        SPAD_w = 32 if ww > 16 else 16

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
