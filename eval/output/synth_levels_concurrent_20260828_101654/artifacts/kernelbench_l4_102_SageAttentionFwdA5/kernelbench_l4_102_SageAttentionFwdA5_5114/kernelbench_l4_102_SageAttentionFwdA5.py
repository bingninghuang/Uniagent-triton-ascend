import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    SageAttention 前向标杆 (精确数学参考)。
    对标 thu-ml/SageAttention @ main 的高层入口 sageattn
    (sageattention/core.py:79, 按 GPU 架构在 :139-160 分发到
    qk_int8_pv_fp16_cuda / qk_int8_pv_fp16_triton / qk_int8_pv_fp8_cuda /
    qk_int8_pv_fp8_cuda_sm90, 覆盖 sm80/86/89/90/120/121;
    sm100 (Blackwell) 不在 sageattn() 路由内, 需使用独立模块
    sageattention3_blackwell 的自有入口)。

    数学定义 (self-attention, qo_len == kv_len = S; GQA: 头 h 用组
    g = h // (H/Hkv) 的 k/v):
        scores[i, j] = sm_scale * q_i · k_j
        causal=1 时仅保留 j <= i (左上对齐; kernel 要求 qo_len==kv_len)
        p[i, :] = softmax(scores[i, :])
        out[i]  = p[i, :] @ v
        lse[i]  = log( sum_j exp(scores[i, j]) )    (自然对数域)
    sm_scale 缺省时 kernel 取 1/sqrt(head_dim); 标杆 case 中显式给出。

    关键语义核对 (core.py):
        - smooth_k (默认开): k <- k - mean(k, dim=seq)。对每行 logits 只
          增加常数 q_i·mean(k), softmax 输出不变, 故标杆省略该步
          (lse 经 kernel 的 lse_correction = q·mean(k)*sm_scale 修正后
          与未平滑的精确 lse 一致, 标杆直接输出精确 lse)。
          注: 官方 lse_correction 的 q·km matmul 在输入 dtype 下计算
          (bf16 级误差), 标杆的精确 lse 更准, kernel 对比容差需覆盖。
        - kernel 返回的 lse 以 2 为底存储, 出口处 /1.44269504 (乘 ln2)
          换回自然对数, 与标杆的 logsumexp 一致; lse 恒 fp32。
          注: 官方 triton 路径 lse 恒按 [B,H,S] 分配 (NHD 不转置),
          本标杆按算子规格 NHD 时输出 [B,S,H]。
        - GQA: H % Hkv == 0, kernel 内 off_h // num_kv_groups 选 kv 头,
          与 repeat_interleave(R, dim=1) 的连续分组一致。
        - head_dim < 64 / (64,128) 时 q/k/v 右侧补零到 64/128, 输出切回
          原 D; 补零维对 logits/output 无贡献, 数学不变。D > 128 不支持。
        - bf16 输入时 v 先转 fp16 再做 PV (精度差异, 见下)。

    布局约定:
        q [B, H, S, D] (HND) 或 [B, S, H, D] (NHD), fp16/bf16
        k, v [B, Hkv, S, D] 或 [B, S, Hkv, D], 与 q 同 dtype
        tensor_layout 字符串 attr: "HND" / "NHD"
        is_causal int (0/1); sm_scale float; return_lse int (0/1)
        输出 out 与 q 同形状同 dtype; lse fp32,
        HND 为 [B, H, S], NHD 为 [B, S, H]


    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, tensor_layout, is_causal, sm_scale, return_lse):
        device = q.device

        if tensor_layout == "NHD":
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        Bsz, H, S, D = q.shape
        Hkv = k.shape[1]
        R = H // Hkv

        qf = q.float()
        kf = k.float()
        vf = v.float()
        if R > 1:                                   # GQA: 广播到 q 头数
            kf = kf.repeat_interleave(R, dim=1)
            vf = vf.repeat_interleave(R, dim=1)

        o = torch.empty(Bsz, H, S, D, dtype=torch.float32, device=device)
        lse = torch.empty(Bsz, H, S, dtype=torch.float32, device=device)

        # 分块在线 softmax: b/h 维批入张量, 仅保留显存控制分块。
        # query 维 QBLK 分块; bh 维自适应分块使 scores 块 ≤ 2**26 个 fp32
        # (256MiB, 与原版单片占用相当), 常规形状下 BHBLK >= BH 即单迭代。
        QBLK = 2048
        BH = Bsz * H
        BHBLK = max(1, min(BH, (1 << 26) // (QBLK * S)))
        qf2 = qf.reshape(BH, S, D)
        kf2 = kf.reshape(BH, S, D)
        vf2 = vf.reshape(BH, S, D)
        o2 = o.reshape(BH, S, D)
        lse2 = lse.reshape(BH, S)
        kj = torch.arange(S, device=device)
        for bh0 in range(0, BH, BHBLK):
            bh1 = min(bh0 + BHBLK, BH)
            for q0 in range(0, S, QBLK):
                q1 = min(q0 + QBLK, S)
                scores = (qf2[bh0:bh1, q0:q1]
                          @ kf2[bh0:bh1].transpose(-1, -2)) * sm_scale
                if is_causal:
                    qi = torch.arange(q0, q1, device=device).unsqueeze(1)
                    scores = scores.masked_fill(kj.unsqueeze(0) > qi,
                                                float("-inf"))
                m = scores.amax(dim=-1)                   # [BHb, Qb]
                p = (scores - m.unsqueeze(-1)).exp()
                s = p.sum(dim=-1)
                lse2[bh0:bh1, q0:q1] = m + s.log()
                o2[bh0:bh1, q0:q1] = (p @ vf2[bh0:bh1]) / s.unsqueeze(-1)

        o = o.to(q.dtype)
        if tensor_layout == "NHD":
            o = o.transpose(1, 2)
            lse = lse.transpose(1, 2)               # [B, S, H]
        if return_lse:
            return o, lse
        return o


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "102_SageAttentionFwdA5.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        # 标准输入分布: 50% 均匀 [-5, 5] + 50% 正态 (mu ∈ [-5, 5], sigma ∈ [0.1, 2])
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)
        shapes, dtypes, attrs = {}, {}, {}
        for inp in case["inputs"]:
            if inp.get("type") == "tensor":
                shapes[inp["name"]] = inp["shape"]
                dtypes[inp["name"]] = dtype_map[inp["dtype"]]
            else:
                attrs[inp["name"]] = inp["value"]

        tensor_layout = str(attrs["tensor_layout"])
        is_causal = int(attrs["is_causal"])
        sm_scale = float(attrs["sm_scale"])
        return_lse = int(attrs["return_lse"])
        sq, sk = shapes["q"], shapes["k"]
        assert shapes["k"] == shapes["v"]
        assert sq[-1] == sk[-1]                      # head_dim 一致
        if tensor_layout == "HND":
            Bsz, H, S, D = sq
            assert sk[1] <= H and H % sk[1] == 0     # GQA: Hkv 整除 H
            assert sk[2] == S                        # self-attn: qo_len==kv_len
        else:
            Bsz, S, H, D = sq
            assert sk[2] <= H and H % sk[2] == 0
            assert sk[1] == S
        assert D in (64, 128)                        # kernel 支持范围
        assert sm_scale > 0

        q = random_tensor(sq, dtypes["q"])
        k = random_tensor(sk, dtypes["k"])
        v = random_tensor(shapes["v"], dtypes["v"])

        input_groups.append([q, k, v, tensor_layout, is_causal,
                             sm_scale, return_lse])
    return input_groups


def get_init_inputs():
    return []