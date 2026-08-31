import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Hardware discovery (G1: dynamic core count). Queried once in __init__
# (never in forward - get_device_limit triggers device sync); robust
# fallback if the query is unavailable.
# ---------------------------------------------------------------------------
def _get_num_cube_cores(device):
    try:
        import torch_npu
        limit = torch_npu.npu.npu_config.get_device_limit(device)
        if isinstance(limit, dict):
            for key in ("cube_core_num", "cube_core", "aicore_num",
                        "ai_core_num", "core_num"):
                v = limit.get(key)
                if v:
                    return max(1, int(v))
            v = limit.get("vector_core_num")
            if v:
                return max(1, int(v) // 2)
    except Exception:
        pass
    return 24


# ---------------------------------------------------------------------------
# Kernel 1: timestep embedding + GEMM1 + bias + SiLU (all fp32 math),
# stores the SiLU output as a hi/lo pair in the input 16-bit dtype so that
# kernel-2 GEMMs can run on the fast CUBE (fp16/bf16) path while keeping
# ~double-float precision against the fp32 reference.
#
#   t_emb[m, j]   = sin(1000*t[m]*f[j])      j in [0, 384)
#   t_emb[m, j+384] = cos(1000*t[m]*f[j])
#   h1 = t_emb @ W1^T + b1      (fp32)
#   h  = silu(h1)               (fp32)
# ---------------------------------------------------------------------------
@triton.jit
def _k1_temb_gemm1_silu(
    ts_ptr, freqs_ptr, w1_ptr, b1_ptr, hh_ptr, hl_ptr,
    B, n_tiles,
    num_cores: tl.constexpr,
    IS_BF16: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    K1: tl.constexpr = 768
    N1: tl.constexpr = 3072
    KH: tl.constexpr = 384
    NUM_N: tl.constexpr = N1 // BN
    KPH: tl.constexpr = KH // BK  # k chunks per sin/cos half (= 3)

    pid = tl.program_id(0).to(tl.int32)
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    for tile_id in range(pid, n_tiles, num_cores):
        tile_m = tile_id // NUM_N
        n0 = (tile_id - tile_m * NUM_N) * BN
        m0 = tile_m * BM
        row_mask = (m0 + offs_m) < B

        # s[m] = 1000 * timestep[m]  (fp32, matches reference)
        s = tl.load(ts_ptr + m0 + offs_m, mask=row_mask, other=0.0)
        s = s.to(tl.float32) * 1000.0

        # Weight tile pointers: W1 is [3072, 768]; load [BN, BK] blocks
        # (K-contiguous => 256B aligned) and transpose for the dot.
        w1_base = w1_ptr + (n0 + offs_n)[:, None] * K1 + offs_k[None, :]

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        # ---- sin half (k in [0, 384)) ----
        for i in range(KPH):
            f = tl.load(freqs_ptr + i * BK + offs_k).to(tl.float32)
            x = s[:, None] * f[None, :]
            v = tl.sin(x)
            if IS_BF16:
                hi = v.to(tl.bfloat16)
                lo = (v - hi.to(tl.float32)).to(tl.bfloat16)
            else:
                hi = v.to(tl.float16)
                lo = (v - hi.to(tl.float32)).to(tl.float16)
            w = tl.load(w1_base)
            w1_base += BK
            acc = tl.dot(hi, tl.trans(w), acc)
            acc = tl.dot(lo, tl.trans(w), acc)

        # ---- cos half (k in [384, 768)); freqs indices re-use [0, 384) ----
        for i in range(KPH):
            f = tl.load(freqs_ptr + i * BK + offs_k).to(tl.float32)
            x = s[:, None] * f[None, :]
            v = tl.cos(x)
            if IS_BF16:
                hi = v.to(tl.bfloat16)
                lo = (v - hi.to(tl.float32)).to(tl.bfloat16)
            else:
                hi = v.to(tl.float16)
                lo = (v - hi.to(tl.float32)).to(tl.float16)
            w = tl.load(w1_base)
            w1_base += BK
            acc = tl.dot(hi, tl.trans(w), acc)
            acc = tl.dot(lo, tl.trans(w), acc)

        # bias + SiLU in fp32 (matches reference)
        b1 = tl.load(b1_ptr + n0 + offs_n).to(tl.float32)
        h1 = acc + b1[None, :]
        h = h1 * tl.sigmoid(h1)

        # split into hi/lo pair in the 16-bit dtype for the CUBE path
        if IS_BF16:
            hh = h.to(tl.bfloat16)
            hl = (h - hh.to(tl.float32)).to(tl.bfloat16)
        else:
            hh = h.to(tl.float16)
            hl = (h - hh.to(tl.float32)).to(tl.float16)

        out_offs = (m0 + offs_m)[:, None] * N1 + (n0 + offs_n)[None, :]
        tl.store(hh_ptr + out_offs, hh, mask=row_mask[:, None])
        tl.store(hl_ptr + out_offs, hl, mask=row_mask[:, None])


# ---------------------------------------------------------------------------
# Kernel 2: fused GEMM2 (t-side) + GEMM3 (pooled-side) + biases + the
# reference's intermediate rounding to the bias dtypes + final add/cast.
#
#   t = h @ W2^T + b2   -> round to dtype(b2) -> back to fp32
#   p = pooled @ W3^T + b3 -> round to dtype(b3) -> back to fp32
#   out = (t + p).to(result_dtype)
# ---------------------------------------------------------------------------
@triton.jit
def _k2_gemm23_add(
    hh_ptr, hl_ptr, pooled_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
    B, n_tiles,
    num_cores: tl.constexpr,
    IS_BF16: tl.constexpr,
    B2_IS_F32: tl.constexpr,
    B3_IS_F32: tl.constexpr,
    OUT_IS_F32: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    K2: tl.constexpr = 3072
    N2: tl.constexpr = 768
    NUM_N: tl.constexpr = N2 // BN

    pid = tl.program_id(0).to(tl.int32)
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    for tile_id in range(pid, n_tiles, num_cores):
        tile_m = tile_id // NUM_N
        n0 = (tile_id - tile_m * NUM_N) * BN
        m0 = tile_m * BM
        row_mask = (m0 + offs_m) < B

        a_base = (m0 + offs_m)[:, None] * K2 + offs_k[None, :]
        w2_base = w2_ptr + (n0 + offs_n)[:, None] * K2 + offs_k[None, :]
        w3_base = w3_ptr + (n0 + offs_n)[:, None] * K2 + offs_k[None, :]

        acc_t = tl.zeros((BM, BN), dtype=tl.float32)
        acc_p = tl.zeros((BM, BN), dtype=tl.float32)

        for k in range(0, K2, BK):
            hh = tl.load(hh_ptr + a_base, mask=row_mask[:, None], other=0.0)
            hl = tl.load(hl_ptr + a_base, mask=row_mask[:, None], other=0.0)
            pp = tl.load(pooled_ptr + a_base, mask=row_mask[:, None], other=0.0)

            w2 = tl.load(w2_base)
            w3 = tl.load(w3_base)

            acc_t = tl.dot(hh, tl.trans(w2), acc_t)
            acc_t = tl.dot(hl, tl.trans(w2), acc_t)
            acc_p = tl.dot(pp, tl.trans(w3), acc_p)

            a_base += BK
            w2_base += BK
            w3_base += BK

        b2 = tl.load(b2_ptr + n0 + offs_n).to(tl.float32)
        b3 = tl.load(b3_ptr + n0 + offs_n).to(tl.float32)

        t = acc_t + b2[None, :]
        p = acc_p + b3[None, :]

        # intermediate rounding to the bias dtypes, as in the reference
        if not B2_IS_F32:
            if IS_BF16:
                t = t.to(tl.bfloat16).to(tl.float32)
            else:
                t = t.to(tl.float16).to(tl.float32)
        if not B3_IS_F32:
            if IS_BF16:
                p = p.to(tl.bfloat16).to(tl.float32)
            else:
                p = p.to(tl.float16).to(tl.float32)

        total = t + p
        if OUT_IS_F32:
            out = total
        elif IS_BF16:
            out = total.to(tl.bfloat16)
        else:
            out = total.to(tl.float16)

        out_offs = (m0 + offs_m)[:, None] * N2 + (n0 + offs_n)[None, :]
        tl.store(out_ptr + out_offs, out, mask=row_mask[:, None])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = 3072
        self.time_embed_dim = 768
        try:
            self.CUBE_CORE_NUM = _get_num_cube_cores(0)
        except Exception:
            self.CUBE_CORE_NUM = 24

    def forward(self, timestep, pooled_projections, freqs,
                timestep_linear1_weight, timestep_linear1_bias,
                timestep_linear2_weight, timestep_linear2_bias,
                text_embedder_weight, text_embedder_bias):
        B = timestep.shape[0]
        dt = timestep.dtype
        is_bf16 = (dt == torch.bfloat16)
        b2_f32 = (timestep_linear2_bias.dtype == torch.float32)
        b3_f32 = (text_embedder_bias.dtype == torch.float32)
        out_dtype = torch.float32 if (b2_f32 or b3_f32) else dt

        # G7: ensure contiguity
        timestep = timestep.contiguous()
        pooled_projections = pooled_projections.contiguous()
        freqs = freqs.contiguous()
        w1 = timestep_linear1_weight.contiguous()
        b1 = timestep_linear1_bias.contiguous()
        w2 = timestep_linear2_weight.contiguous()
        b2 = timestep_linear2_bias.contiguous()
        w3 = text_embedder_weight.contiguous()
        b3 = text_embedder_bias.contiguous()

        dev = timestep.device
        hh = torch.empty((B, self.inner_dim), dtype=dt, device=dev)
        hl = torch.empty((B, self.inner_dim), dtype=dt, device=dev)
        out = torch.empty((B, self.time_embed_dim), dtype=out_dtype, device=dev)

        num_cores = self.CUBE_CORE_NUM

        # ---------------- kernel 1 dispatch ----------------
        if B <= 16:
            bm1 = 16
        elif B <= 32:
            bm1 = 32
        else:
            bm1 = 64
        bn1 = 128
        m_tiles1 = (B + bm1 - 1) // bm1
        n_tiles1 = m_tiles1 * (3072 // bn1)
        grid1 = n_tiles1 if n_tiles1 < num_cores else num_cores
        _k1_temb_gemm1_silu[(grid1,)](
            timestep, freqs, w1, b1, hh, hl,
            B, n_tiles1,
            num_cores=grid1,
            IS_BF16=is_bf16,
            BM=bm1, BN=bn1, BK=128,
        )

        # ---------------- kernel 2 dispatch ----------------
        bm2 = 64
        bn2 = 64
        m_tiles2 = (B + bm2 - 1) // bm2
        n_tiles2 = m_tiles2 * (768 // bn2)
        grid2 = n_tiles2 if n_tiles2 < num_cores else num_cores
        _k2_gemm23_add[(grid2,)](
            hh, hl, pooled_projections, w2, b2, w3, b3, out,
            B, n_tiles2,
            num_cores=grid2,
            IS_BF16=is_bf16,
            B2_IS_F32=b2_f32,
            B3_IS_F32=b3_f32,
            OUT_IS_F32=(out_dtype == torch.float32),
            BM=bm2, BN=bn2, BK=128,
        )

        return out
