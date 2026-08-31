import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Model that performs chunk KDA forward with fused gate computation
    (golden for chunk_kda_with_fused_gate_fwd).

    torch 原生小算子拼接实现，语义与 vLLM fused kernel 逐条对齐：

    1. fused gate（对应 kda_gate_fwd_kernel，全程 fp32）：
           g = -exp(A_log) * softplus(raw_g + g_bias)
       raw_g: [B, T, H*K] -> reshape [B, T, H, K]（channel-wise 衰减，
       每个 k 通道独立）；softplus 取默认 beta=1, threshold=20；
       g <= 0，每步衰减因子 alpha_t = exp(g_t) ∈ (0, 1]。

    2. KDA 递推（论文 Eq.1，fp32 逐时间步，按头 batch 化）：
           h_t = Diag(exp(g_t)) · h_{t-1}                       # [H, K, V]
           delta_t = (v_t - h_t^T k_t) * beta_t                 # [H, V]
           h_t = h_t + k_t ⊗ delta_t
           o_t = h_t^T · (scale * q_t)                          # [H, V]
       即 S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T。

    3. varlen（cu_seqlens 不为 None，约定 B == 1）：按 cu_seqlens 切序列，
       每条序列独立递推，initial_state 按序列取 [N, H, K, V]。

    说明：
    - use_qk_l2norm_in_kernel 是上层 wrapper（chunk_kda_with_fused_gate）
      在调用 fwd 之前完成的，不属于本算子语义，这里不包含。
    - 精度：全程 fp32 递推；输出 o 转回 v.dtype，final_state 保持 fp32。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, raw_g, beta, A_log, g_bias=None, scale=None,
                initial_state=None, output_final_state=False, cu_seqlens=None):
        """
        Args:
            q (Tensor): Query tensor of shape [B, T, H, K].
            k (Tensor): Key tensor of shape [B, T, H, K].
            v (Tensor): Value tensor of shape [B, T, H, V].
            raw_g (Tensor): Raw gate projection of shape [B, T, H*K]（或已reshape好的 [B, T, H, K]）.
            beta (Tensor): Beta tensor of shape [B, T, H].
            A_log (Tensor): Log decay parameter of shape [H]（或 [1, 1, H, 1]）.
            g_bias (Tensor, optional): Gate bias of shape [H*K]（或 [1, 1, H, K]）.
            scale (float, optional): Attention scale. Default: K ** -0.5.
            initial_state (Tensor, optional): Initial state of shape [B, H, K, V]
                （varlen 时为 [N, H, K, V]，N 为序列条数）.
            output_final_state (bool): Whether to return the final state.
            cu_seqlens (Tensor, optional): Cumulative sequence lengths（varlen，约定 B == 1）.

        Returns:
            o (Tensor): Output of shape [B, T, H, V]，dtype 与 v 一致.
            final_state (Tensor or None): 最终状态 [B, H, K, V]（varlen 为 [N, H, K, V]），fp32.
        """
        B, T, H, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K ** -0.5

        # ---- 1. fused gate: g = -exp(A_log) * softplus(raw_g + g_bias)，fp32 ----
        rg = raw_g.float()
        if rg.dim() == 3:          # [B, T, H*K] -> [B, T, H, K]
            rg = rg.reshape(B, T, H, K)
        if g_bias is not None:
            rg = rg + g_bias.float().reshape(H, K)
        a = -torch.exp(A_log.float().reshape(H))                    # [H]
        g = a.view(1, 1, H, 1) * F.softplus(rg)                     # [B, T, H, K]，<= 0

        # ---- 2. 统一转置为 [B, H, T, ...] 并升 fp32，q 乘 scale ----
        qf = q.transpose(1, 2).float() * scale                      # [B, H, T, K]
        kf = k.transpose(1, 2).float()                              # [B, H, T, K]
        vf = v.transpose(1, 2).float()                              # [B, H, T, V]
        betaf = beta.transpose(1, 2).float()                        # [B, H, T]
        gf = g.transpose(1, 2)                                      # [B, H, T, K]

        # ---- 3. 构造 (batch_idx, state_idx, bos, eos) 列表 ----
        if cu_seqlens is not None:
            # varlen：约定 B == 1，q/k/v 为 [1, total_T, H, ...] 拉直序列
            cu = cu_seqlens.cpu().tolist()
            spans = [(0, i, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
        else:
            spans = [(b, b, 0, T) for b in range(B)]

        o = torch.zeros(B, H, T, V, device=q.device, dtype=torch.float32)
        final_states = []

        for b, si, bos, eos in spans:
            h = torch.zeros(H, K, V, device=q.device, dtype=torch.float32)
            if initial_state is not None:
                h = initial_state[si].float().clone()

            for t in range(bos, eos):
                qt = qf[b, :, t]                                    # [H, K]
                kt = kf[b, :, t]                                    # [H, K]
                vt = vf[b, :, t]                                    # [H, V]
                gt = gf[b, :, t]                                    # [H, K]
                bt = betaf[b, :, t]                                 # [H]

                # 衰减 -> delta 残差 -> beta 缩放 -> 状态更新 -> 输出
                h = h * gt.exp()[..., None]                         # Diag(alpha) · h
                delta = vt - (h * kt[..., None]).sum(-2)            # v - h^T k，[H, V]
                delta = delta * bt[..., None]
                h = h + kt.unsqueeze(-1) * delta.unsqueeze(-2)      # h + k ⊗ delta
                o[b, :, t] = torch.einsum("hk,hkv->hv", qt, h)      # o = h^T q

            final_states.append(h)

        o = o.transpose(1, 2).contiguous().to(v.dtype)              # [B, T, H, V]
        final_state = torch.stack(final_states, dim=0) if output_final_state else None
        return o, final_state


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "40_ChunkKdaWithFusedGateFwd.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        # KDA 递推对幅值极其敏感，进一步收紧输入分布以保持长序列数值稳定
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-0.1, 0.1).item())
            sigma = float(torch.empty(1).uniform_(0.01, 0.05).item())
            if dtype is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-0.1, 0.1)

    def beta_tensor(shape, dtype):
        # beta 必须为正且较小，防止状态爆炸
        t = torch.empty(shape, dtype=torch.float32).uniform_(0.01, 0.1)
        return t.to(dtype)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tensor_names = ("q", "k", "v", "raw_g", "beta", "A_log", "g_bias", "initial_state")

    input_groups = []
    for case in cases:
        tensors = {}
        scale = None
        output_final_state = False
        cu_seqlens = None

        for inp in case["inputs"]:
            name = inp.get("name", "")
            if name in tensor_names:
                dtype = dtype_map[inp.get("dtype", "bfloat16")]
                if name == "beta":
                    tensors[name] = beta_tensor(inp["shape"], dtype)
                else:
                    tensors[name] = random_tensor(inp["shape"], dtype)
            elif name == "scale":
                scale = inp["value"]
            elif name == "output_final_state":
                output_final_state = bool(inp["value"])
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)

        input_groups.append([
            tensors["q"], tensors["k"], tensors["v"], tensors["raw_g"],
            tensors["beta"], tensors["A_log"], tensors.get("g_bias"),
            scale, tensors.get("initial_state"), output_final_state, cu_seqlens,
        ])
    return input_groups


def get_init_inputs():
    return []