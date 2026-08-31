import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except Exception:  # pragma: no cover
    torch_npu = None


@triton.jit
def _sinus_kernel(t_ptr, f_ptr, out_ptr, M,
                  FREQ: tl.constexpr, FREQ_BLOCK: tl.constexpr,
                  num_cores: tl.constexpr, OUT_DT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    offs = tl.arange(0, FREQ_BLOCK)
    mask = offs < FREQ
    f = tl.load(f_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    for b in range(pid, M, num_cores):
        t = tl.load(t_ptr + b).to(tl.float32)
        x = (t * 1000.0) * f
        s = tl.sin(x).to(OUT_DT)
        c = tl.cos(x).to(OUT_DT)
        tl.store(out_ptr + b * (2 * FREQ) + offs, s, mask=mask)
        tl.store(out_ptr + b * (2 * FREQ) + FREQ + offs, c, mask=mask)


@triton.jit
def _gemm1_silu_kernel(a_ptr, w_ptr, b_ptr, h_ptr, M,
                       N: tl.constexpr, K: tl.constexpr,
                       num_cores: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       OUT_DT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    PM = tl.cdiv(M, BM)
    PN = N // BN
    NB = PM * PN
    for bi in range(pid, NB, num_cores):
        bm = bi // PN
        bn = bi % PN
        offs_m = (bm * BM + tl.arange(0, BM)).to(tl.int32)
        offs_n = (bn * BN + tl.arange(0, BN)).to(tl.int32)
        offs_k = tl.arange(0, BK)
        mmask = offs_m < M
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=mmask[:, None], other=0.0)
            w = tl.load(w_ptrs)
            acc = tl.dot(a, tl.trans(w), acc)
            a_ptrs += BK
            w_ptrs += BK
        bias = tl.load(b_ptr + offs_n).to(tl.float32)
        acc = acc + bias[None, :]
        h = acc * tl.sigmoid(acc)
        tl.store(h_ptr + offs_m[:, None] * N + offs_n[None, :], h.to(OUT_DT),
                 mask=mmask[:, None])


@triton.jit
def _gemm23_kernel(a_t_ptr, a_x_ptr, w_t_ptr, w_x_ptr,
                   b_t_ptr, b_x_ptr, out_t_ptr, out_x_ptr, M,
                   N: tl.constexpr, K: tl.constexpr,
                   num_cores: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   OUT_T_DT: tl.constexpr, OUT_X_DT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    PM = tl.cdiv(M, BM)
    PN = N // BN
    GNB = PM * PN
    NB = 2 * GNB
    for bi in range(pid, NB, num_cores):
        g = bi // GNB
        idx = bi % GNB
        bm = idx // PN
        bn = idx % PN
        offs_m = (bm * BM + tl.arange(0, BM)).to(tl.int32)
        offs_n = (bn * BN + tl.arange(0, BN)).to(tl.int32)
        offs_k = tl.arange(0, BK)
        mmask = offs_m < M
        if g == 0:
            a_ptrs = a_t_ptr + offs_m[:, None] * K + offs_k[None, :]
            w_ptrs = w_t_ptr + offs_n[:, None] * K + offs_k[None, :]
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in range(0, K, BK):
                a = tl.load(a_ptrs, mask=mmask[:, None], other=0.0)
                w = tl.load(w_ptrs)
                acc = tl.dot(a, tl.trans(w), acc)
                a_ptrs += BK
                w_ptrs += BK
            bias = tl.load(b_t_ptr + offs_n).to(tl.float32)
            acc = acc + bias[None, :]
            tl.store(out_t_ptr + offs_m[:, None] * N + offs_n[None, :],
                     acc.to(OUT_T_DT), mask=mmask[:, None])
        else:
            a_ptrs = a_x_ptr + offs_m[:, None] * K + offs_k[None, :]
            w_ptrs = w_x_ptr + offs_n[:, None] * K + offs_k[None, :]
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k in range(0, K, BK):
                a = tl.load(a_ptrs, mask=mmask[:, None], other=0.0)
                w = tl.load(w_ptrs)
                acc = tl.dot(a, tl.trans(w), acc)
                a_ptrs += BK
                w_ptrs += BK
            bias = tl.load(b_x_ptr + offs_n).to(tl.float32)
            acc = acc + bias[None, :]
            tl.store(out_x_ptr + offs_m[:, None] * N + offs_n[None, :],
                     acc.to(OUT_X_DT), mask=mmask[:, None])


@triton.jit
def _add_kernel(o1_ptr, o2_ptr, out_ptr, n_elements,
                BLOCK: tl.constexpr, num_cores: tl.constexpr,
                OUT_DT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int32)
    NB = tl.cdiv(n_elements, BLOCK)
    for bi in range(pid, NB, num_cores):
        offs = (bi * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
        mask = offs < n_elements
        a = tl.load(o1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(o2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, (a + b).to(OUT_DT), mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = 3072
        self.time_embed_dim = 768
        try:
            limits = torch_npu.npu.npu_config.get_device_limit(0)
            self.VEC_CORE_NUM = int(limits.get("vector_core_num", 48))
            self.CUBE_CORE_NUM = int(limits.get("cube_core_num", 24))
        except Exception:
            self.VEC_CORE_NUM = 48
            self.CUBE_CORE_NUM = 24
        self._tl_dts = {
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
            torch.float32: tl.float32,
        }

    def forward(self, timestep: torch.Tensor, pooled_projections: torch.Tensor,
                freqs: torch.Tensor,
                timestep_linear1_weight: torch.Tensor, timestep_linear1_bias: torch.Tensor,
                timestep_linear2_weight: torch.Tensor, timestep_linear2_bias: torch.Tensor,
                text_embedder_weight: torch.Tensor, text_embedder_bias: torch.Tensor):
        B = timestep.shape[0]
        device = timestep.device
        if not timestep.is_contiguous():
            timestep = timestep.contiguous()
        if not pooled_projections.is_contiguous():
            pooled_projections = pooled_projections.contiguous()
        if not freqs.is_contiguous():
            freqs = freqs.contiguous()
        if not timestep_linear1_weight.is_contiguous():
            timestep_linear1_weight = timestep_linear1_weight.contiguous()
        if not timestep_linear1_bias.is_contiguous():
            timestep_linear1_bias = timestep_linear1_bias.contiguous()
        if not timestep_linear2_weight.is_contiguous():
            timestep_linear2_weight = timestep_linear2_weight.contiguous()
        if not timestep_linear2_bias.is_contiguous():
            timestep_linear2_bias = timestep_linear2_bias.contiguous()
        if not text_embedder_weight.is_contiguous():
            text_embedder_weight = text_embedder_weight.contiguous()
        if not text_embedder_bias.is_contiguous():
            text_embedder_bias = text_embedder_bias.contiguous()

        low = self._tl_dts[timestep.dtype]
        b2_dt = self._tl_dts[timestep_linear2_bias.dtype]
        b3_dt = self._tl_dts[text_embedder_bias.dtype]
        out_dtype = torch.result_type(timestep_linear2_bias.dtype,
                                      text_embedder_bias.dtype)
        out_dt = self._tl_dts[out_dtype]

        h1 = torch.empty((B, 768), device=device, dtype=timestep.dtype)
        h2 = torch.empty((B, 3072), device=device, dtype=timestep.dtype)
        t2 = torch.empty((B, 768), device=device,
                         dtype=timestep_linear2_bias.dtype)
        x2 = torch.empty((B, 768), device=device,
                         dtype=text_embedder_bias.dtype)
        out = torch.empty((B, 768), device=device, dtype=out_dtype)

        # K1: sinusoidal timestep embedding -> [B, 768]
        _sinus_kernel[(min(B, self.VEC_CORE_NUM),)](
            timestep, freqs, h1, B,
            FREQ=384, FREQ_BLOCK=512,
            num_cores=self.VEC_CORE_NUM, OUT_DT=low,
        )

        # K2: H2 = silu(H1 @ W1^T + b1)  -> [B, 3072]
        _gemm1_silu_kernel[(self.CUBE_CORE_NUM,)](
            h1, timestep_linear1_weight, timestep_linear1_bias, h2, B,
            N=3072, K=768,
            num_cores=self.CUBE_CORE_NUM,
            BM=64, BN=128, BK=128, OUT_DT=low,
        )

        # K3: t2 = H2 @ W2^T + b2 ; x2 = pooled @ W3^T + b3 (one kernel)
        _gemm23_kernel[(self.CUBE_CORE_NUM,)](
            h2, pooled_projections,
            timestep_linear2_weight, text_embedder_weight,
            timestep_linear2_bias, text_embedder_bias,
            t2, x2, B,
            N=768, K=3072,
            num_cores=self.CUBE_CORE_NUM,
            BM=64, BN=64, BK=128,
            OUT_T_DT=b2_dt, OUT_X_DT=b3_dt,
        )

        # K4: out = (t2 + x2) in fp32, cast to out dtype
        _add_kernel[(min(triton.cdiv(B * 768, 1024), self.VEC_CORE_NUM),)](
            t2, x2, out, B * 768,
            BLOCK=1024, num_cores=self.VEC_CORE_NUM, OUT_DT=out_dt,
        )

        return out
