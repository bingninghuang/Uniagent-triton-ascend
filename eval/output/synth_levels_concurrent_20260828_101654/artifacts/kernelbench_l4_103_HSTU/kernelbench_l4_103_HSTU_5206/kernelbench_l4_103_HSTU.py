import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os


class Model(nn.Module):
    """
    HSTU 注意力前向 —— NVIDIA recsys-examples HSTU（Hierarchical
    Sequential Transduction Unit，生成式推荐）jagged attention golden
    reference。

    算子出处：
      NVIDIA/recsys-examples（main 分支）examples/hstu/ops
      - 语义依据:  ops/pt_ops/pt_hstu_attention.py
                  pytorch_hstu_mha + _get_valid_attn_mask（官方 PyTorch
                  参考实现，仓上 CUTLASS/Triton kernel 的数值对标对象）
      - kernel:    ops/triton_ops/triton_hstu_attention.py triton_hstu_mha
      - 测试:      examples/hstu/test/test_hstu_op.py
      数学语义逐行对齐 pytorch_hstu_mha / _get_valid_attn_mask。

    数学语义（jagged 布局：q/k [L,H,D]，v [L,H,V]，seq_offsets [B+1]；
    L = 各 batch 变长序列总长，alpha 为打分缩放）：
      1) 打分：scores = SiLU(q @ k^T * alpha) / scaling_seqlen
         —— HSTU 用 SiLU 取代 softmax（无归一化），逐点激活后除
         scaling_seqlen；scaling_seqlen = -1 时取本批最大序列长
         max_seq_len（与官方一致）。
      2) 掩码（逐 batch 在 [n,n] 上构建，乘法置零，不用 -inf）：
         记 ids = arange(n)。若 num_contextuals = nc > 0：
           ids = clamp(ids - nc + 1, min=0)   （前 nc 个 contextual
           token 的 id 全归为 0；历史 token id 从 1 起）
           max_ids = n - nc + 1               （否则 max_ids = n）
         row/col = ids 的行列展开，dist = row - col；
         causal: 保留 dist > 0 及对角线（row >= col）；
         非因果: dist 取 |row - col|（> 0 加对角线即全互看）。
         若给定 num_targets = nt（候选/目标 token 位于序列末尾 nt 个）：
           tg = clamp(ids - max_ids + nt, min=-1) // target_group_size
           （历史 token 恒为 -1；末尾 nt 个目标按 target_group_size 分组）
           目标掩码 = (tg_row == tg_col) | (tg_row < 0) | (tg_col < 0)
           —— 保留目标组内注意力与"目标↔历史"注意力，屏蔽跨目标组
           注意力；随后 max_ids -= nt（max_ids 变为历史长度）。
         若 max_attn_len > 0：再要求 dist <= max_attn_len（局部窗口，
         非因果时用 |dist|，与官方一致）。
         若 nc 张量给定：再放开 (row_ids == 0) & (col_ids < max_ids)
         —— contextual 行可以看到所有历史（不含目标）token。
      3) 输出：out = (scores * mask) @ v，回铸 q.dtype。

    变体说明：遵循 pt_hstu_mha 张量路径（num_targets/num_contextuals
      以 [B] int32 张量传入；仓上测试 test_hstu_attn 即此用法）。全 0
      张量与官方 None（不启用）严格等价：nt=0 时所有 tg = -1，目标
      掩码恒真、max_ids 不变；nc=0 时 ids 整体 +1（距离差不变）、
      row_ids==0 恒假，contextual 放开项 vacuous。min_full_attn_seq_len
      / contextual_seq_len / sort_by_length / enable_tma / dropout_pr /
      training 属未测变体或训练/调度参数，未纳入（dropout_pr=0 时
      dropout 为恒等；官方测试亦恒为 0）。

    dtype 约定：内部全程 fp32 计算（官方 CUTLASS/Triton kernel 中间量
      有 bf16/fp16 舍入，golden 定义 fp32 理想语义，差异属已知精度带，
      以 fp64 基线刻画）；输入 q/k/v 为 bf16/fp16（仓上测试 dtype），
      输出回铸 q.dtype。掩码为精确布尔逻辑，无数值误差。
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, seq_offsets, num_targets, num_contextuals,
                alpha, causal, max_attn_len, target_group_size,
                scaling_seqlen):
        device = q.device
        causal = bool(causal)
        L, H, D = q.shape
        V = v.shape[2]
        lens = seq_offsets[1:] - seq_offsets[:-1]
        if scaling_seqlen == -1:
            scaling_seqlen = int(lens.max().item())

        q32 = q.detach().float()
        k32 = k.detach().float()
        v32 = v.detach().float()
        out = torch.zeros(L, H, V, dtype=torch.float32, device=device)

        offs = seq_offsets.tolist()
        nt_list = num_targets.tolist()
        nc_list = num_contextuals.tolist()

        for b in range(len(offs) - 1):
            s, e = offs[b], offs[b + 1]
            n = e - s
            nt = nt_list[b]
            nc = nc_list[b]

            # ---- 掩码（逐行对齐 _get_valid_attn_mask 张量路径）----
            ids = torch.arange(n, dtype=torch.int64, device=device)
            ids = torch.clamp(ids - nc + 1, min=0)
            max_ids = n - nc + 1
            row_ids = ids.view(-1, 1).expand(n, n)
            col_ids = ids.view(1, -1).expand(n, n)
            dist = row_ids - col_ids
            if not causal:
                dist = torch.where(dist > 0, dist, -dist)
            valid = torch.eye(n, dtype=torch.bool, device=device) | (dist > 0)

            tg_row = torch.clamp(row_ids - max_ids + nt, min=-1) \
                // target_group_size
            tg_col = tg_row.transpose(0, 1)
            tg_dist = tg_row - tg_col
            tg_mask = (tg_dist == 0) | (tg_row < 0) | (tg_col < 0)
            valid = valid & tg_mask
            max_ids = max_ids - nt

            if max_attn_len > 0:
                valid = valid & (dist <= max_attn_len)

            valid = valid | ((row_ids == 0) & (col_ids < max_ids))

            # ---- SiLU 注意力（逐行对齐 pytorch_hstu_mha）----
            qb = q32[s:e].permute(1, 0, 2)  # [H,n,D]
            kb = k32[s:e].permute(1, 0, 2)
            vb = v32[s:e].permute(1, 0, 2)  # [H,n,V]
            scores = torch.einsum("hxd,hyd->hxy", qb, kb) * alpha
            scores = F.silu(scores) / scaling_seqlen
            scores = scores * valid.unsqueeze(0)
            out[s:e] = torch.einsum("hxy,hyv->hxv", scores, vb) \
                .permute(1, 0, 2)

        return out.to(q.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "103_HSTU.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    input_groups = []
    for case_idx, case in enumerate(cases):
        inputs = case["inputs"]

        def info(name):
            return next(i for i in inputs if i["name"] == name)

        dt = dtype_map[info("q")["dtype"]]
        q = random_tensor(info("q")["shape"], dt)
        k = random_tensor(info("k")["shape"], dt)
        v = random_tensor(info("v")["shape"], dt)
        seq_offsets = torch.tensor(info("seq_offsets")["value"],
                                   dtype=torch.int64)
        num_targets = torch.tensor(info("num_targets")["value"],
                                   dtype=torch.int32)
        num_contextuals = torch.tensor(info("num_contextuals")["value"],
                                       dtype=torch.int32)

        alpha = info("alpha")["value"]
        causal = info("causal")["value"]
        max_attn_len = info("max_attn_len")["value"]
        target_group_size = info("target_group_size")["value"]
        scaling_seqlen = info("scaling_seqlen")["value"]

        input_groups.append([q, k, v, seq_offsets, num_targets,
                             num_contextuals, alpha, causal, max_attn_len,
                             target_group_size, scaling_seqlen])
    return input_groups


def get_init_inputs():
    return []
