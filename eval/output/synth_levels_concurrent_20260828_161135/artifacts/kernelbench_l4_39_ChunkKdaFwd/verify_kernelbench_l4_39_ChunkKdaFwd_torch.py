import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Model that performs chunk KDA forward with pre-computed log-decay gate
    (golden for chunk_kda_fwd).

    torch 原生小算子拼接实现，语义与 vLLM chunk_kda_fwd 对齐。
    kernel 内部为分块流水线（KKT -> solve_tril -> recompute_w_u -> fwd_h
    -> gla_fwd_o），与下面的逐步递推数学等价：

    1. 门控：g 直接输入（log 衰减，channel-wise，[B, T, H, K]，g <= 0），
       每步衰减因子 alpha_t = exp(g_t) ∈ (0, 1]。
       （kernel 内部的 chunk-local cumsum 与 RCP_LN2 转换只是 exp2 实现
       细节，逐步递推天然等价，golden 不涉及。）

    2. KDA 递推（fp32 逐时间步，按头 batch 化）：
           S_t = Diag(exp(g_t)) · S_{t-1}                     # [H, K, V]
           delta_t = (v_t - S_t^T k_t) * beta_t               # [H, V]
           S_t = S_t + k_t ⊗ delta_t
           o_t = S_t^T · (scale * q_t)                        # [H, V]
       即 S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T。

    3. state 布局约定（与 vLLM kernel 一致）：initial_state / final_state
       均为转置布局 [N, H, V, K]；本实现内部转置为 [H, K, V] 递推，
       返回前再转置回 [N, H, V, K]。

    4. varlen（cu_seqlens 不为 None，约定 B == 1）：按 cu_seqlens 切序列，
       每条序列独立递推，initial_state 按序列取 [N, H, V, K]。

    说明：
    - use_qk_l2norm_in_kernel 是上层 wrapper（chunk_kda）在调用 fwd 之前
      完成的，不属于本算子语义，这里不包含。
    - 精度：全程 fp32 递推；输出 o 转回 v.dtype，final_state 保持 fp32。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, g, beta, scale=None, initial_state=None,
                output_final_state=False, cu_seqlens=None):
        """
        Args:
            q (Tensor): Query tensor of shape [B, T, H, K].
            k (Tensor): Key tensor of shape [B, T, H, K].
            v (Tensor): Value tensor of shape [B, T, H, V].
            g (Tensor): Log-decay gate of shape [B, T, H, K]（channel-wise，<= 0）.
            beta (Tensor): Beta tensor of shape [B, T, H].
            scale (float, optional): Attention scale. Default: K ** -0.5.
            initial_state (Tensor, optional): Initial state，转置布局 [N, H, V, K]
                （非 varlen 时 N == B）.
            output_final_state (bool): Whether to return the final state.
            cu_seqlens (Tensor, optional): Cumulative sequence lengths（varlen，约定 B == 1）.

        Returns:
            o (Tensor): Output of shape [B, T, H, V]，dtype 与 v 一致.
            final_state (Tensor or None): 最终状态，转置布局 [N, H, V, K]，fp32.
        """
        B, T, H, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K ** -0.5

        # 统一转置为 [B, H, T, ...] 并升 fp32，q 乘 scale
        qf = q.transpose(1, 2).float() * scale                      # [B, H, T, K]
        kf = k.transpose(1, 2).float()                              # [B, H, T, K]
        vf = v.transpose(1, 2).float()                              # [B, H, T, V]
        gf = g.transpose(1, 2).float()                              # [B, H, T, K]
        betaf = beta.transpose(1, 2).float()                        # [B, H, T]

        if cu_seqlens is None:
            # 批量递推：S 带 batch 维 [B, H, K, V]，仅保留时间维循环
            # （逐时间步递推是 KDA 的数学语义本身，非分核逻辑）
            S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
            if initial_state is not None:
                # state 布局转换：kernel 约定 [N, H, V, K] -> 递推用 [B, H, K, V]
                S = initial_state.float().transpose(-1, -2)
            o = torch.zeros(B, H, T, V, device=q.device, dtype=torch.float32)
            for t in range(T):
                # 衰减 -> delta 残差 -> beta 缩放 -> 状态更新 -> 输出
                S = S * gf[:, :, t].exp().unsqueeze(-1)             # Diag(alpha) · S
                delta = vf[:, :, t] - (S * kf[:, :, t].unsqueeze(-1)).sum(-2)
                delta = delta * betaf[:, :, t].unsqueeze(-1)        # [B, H, V]
                S = S + kf[:, :, t].unsqueeze(-1) * delta.unsqueeze(-2)
                o[:, :, t] = torch.einsum("bhk,bhkv->bhv", qf[:, :, t], S)
            # 递推布局 [B, H, K, V] -> kernel 约定 [N, H, V, K]
            final_state = S.transpose(-1, -2).contiguous()  # STAGE-MARKER-7
        else:
            # varlen（约定 B == 1）：按 cu_seqlens 切序列，每条序列独立递推，
            # initial_state 按序列取 [N, H, V, K]；分段是 varlen 的语义边界
            cu = cu_seqlens.cpu().tolist()
            o = torch.zeros(B, H, T, V, device=q.device, dtype=torch.float32)
            final_states = []
            for i in range(len(cu) - 1):
                bos, eos = cu[i], cu[i + 1]
                S = torch.zeros(H, K, V, device=q.device, dtype=torch.float32)
                if initial_state is not None:
                    S = initial_state[i].float().transpose(-1, -2).contiguous()
                for t in range(bos, eos):
                    S = S * gf[0, :, t].exp().unsqueeze(-1)
                    delta = vf[0, :, t] - (S * kf[0, :, t].unsqueeze(-1)).sum(-2)
                    delta = delta * betaf[0, :, t].unsqueeze(-1)
                    S = S + kf[0, :, t].unsqueeze(-1) * delta.unsqueeze(-2)
                    o[0, :, t] = torch.einsum("hk,hkv->hv", qf[0, :, t], S)
                final_states.append(S.transpose(-1, -2).contiguous())
            final_state = torch.stack(final_states, dim=0)

        o = o.transpose(1, 2).contiguous().to(v.dtype)              # [B, T, H, V]
        return o, final_state


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "39_ChunkKdaFwd.json")
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

    def gate_tensor(shape, dtype):
        # g 为 log 衰减，语义上必须 <= 0，硬编码走 logsigmoid(randn)，
        # 对齐 vLLM 官方测试 test_kda.py 的 g 生成方式。
        # 统一 fp32 生成（randn 对 bf16 支持差，且 g 本身就是 fp32）。
        return F.logsigmoid(torch.randn(shape, dtype=torch.float32)).to(dtype)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    input_groups = []
    for case in cases:
        tensors = {}
        scale = None
        output_final_state = False
        cu_seqlens = None

        for inp in case["inputs"]:
            name = inp.get("name", "")
            if name in ("q", "k", "v", "initial_state"):
                dtype = dtype_map[inp.get("dtype", "bfloat16")]
                tensors[name] = random_tensor(inp["shape"], dtype)
            elif name == "beta":
                dtype = dtype_map[inp.get("dtype", "bfloat16")]
                tensors[name] = beta_tensor(inp["shape"], dtype)
            elif name == "g":
                # g 硬编码 logsigmoid 约束，不读 case 的分布字段
                dtype = dtype_map[inp.get("dtype", "float32")]
                tensors[name] = gate_tensor(inp["shape"], dtype)
            elif name == "scale":
                scale = inp["value"]
            elif name == "output_final_state":
                output_final_state = bool(inp["value"])
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)

        input_groups.append([
            tensors["q"], tensors["k"], tensors["v"], tensors["g"],
            tensors["beta"], scale, tensors.get("initial_state"),
            output_final_state, cu_seqlens,
        ])
    return input_groups


def get_init_inputs():
    return []