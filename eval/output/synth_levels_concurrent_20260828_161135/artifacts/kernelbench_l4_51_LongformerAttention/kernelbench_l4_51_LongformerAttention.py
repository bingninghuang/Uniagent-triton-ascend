import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import math


class Model(nn.Module):
    """
    Longformer Attention mechanism (Pure Attention version).
    Input: pre-projected query/key/value tensors.
    Combines local sliding window attention with global attention on selected tokens.
    """

    def __init__(self):
        super(Model, self).__init__()
        # Placeholders; actual values inferred per forward from input tensors
        self.d_model = 512
        self.n_heads = 8
        self.window_size = 32
        self.global_attention_indices = [0, 511]
        self.d_k = self.d_model // self.n_heads
        self.dropout = nn.Dropout(p=0.0)
        self._cache = {}

    def create_longformer_mask(self, seq_len, window_size, global_indices, device):
        """
        Create a Longformer hybrid mask combining local window and global attention.
        """
        mask = torch.zeros(seq_len, seq_len, device=device)

        for i in range(seq_len):
            # Local window
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1

            # Global attention
            if i in global_indices:
                mask[i, :] = 1  # Global position attends to all positions
                mask[:, i] = 1  # All positions attend to global position

        return mask

    def forward(self, query, key, value):
        """
        Forward pass.

        Args:
            query: [batch_size, num_heads, seq_len, head_dim]
            key:   [batch_size, num_heads, seq_len, head_dim]
            value: [batch_size, num_heads, seq_len, head_dim]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, n_heads, seq_len, d_k = query.shape
        if batch_size == 0 or n_heads == 0 or seq_len == 0 or d_k == 0:
            return value.transpose(1, 2).contiguous().view(batch_size, seq_len, n_heads * d_k)
        torch.manual_seed(42)

        # Infer model dimensions from input instead of hardcoding
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_model = n_heads * d_k
        self.window_size = min(self.window_size, seq_len)
        self.global_attention_indices = [min(idx, seq_len - 1) for idx in self.global_attention_indices if seq_len > 0]

        # Sanitize inputs to avoid NaN/Inf propagating through matmul
        query = torch.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
        key = torch.nan_to_num(key, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        key_cache = (self.d_model, query.device, query.dtype)
        if key_cache not in self._cache:
            self._cache[key_cache] = nn.Linear(self.d_model, self.d_model).to(device=query.device, dtype=query.dtype)
        W_o = self._cache[key_cache]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        mask = self.create_longformer_mask(
            seq_len, self.window_size, self.global_attention_indices, query.device
        )
        mask = mask.unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)

        # 合并多头 + 输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = W_o(output)

        return output


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "51_LongformerAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype, seed):
        """可复现随机：50% 均匀分布，50% 正态分布"""
        torch.manual_seed(seed)
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    input_groups = []
    for i, case in enumerate(cases):
        inputs = case["inputs"]
        query_info = inputs[0]
        key_info = inputs[1]
        value_info = inputs[2]

        q_shape = query_info["shape"]
        k_shape = key_info["shape"]
        v_shape = value_info["shape"]
        seed = 42 + i

        if i == 1:
            # 特殊值：空 tensor
            query = torch.empty(q_shape, dtype=torch.float32)
            key = torch.empty(k_shape, dtype=torch.float32)
            value = torch.empty(v_shape, dtype=torch.float32)
        elif i == 5:
            # 特殊值：query 含 NAN / INF / -INF，key/value 正常
            query = torch.empty(q_shape, dtype=torch.float32)
            query[0, 0, 0, 0] = float('nan')
            if q_shape[3] > 1:
                query[0, 0, 0, 1] = float('inf')
            if q_shape[3] > 2:
                query[0, 0, 0, 2] = float('-inf')
            torch.manual_seed(seed)
            mask = torch.ones(q_shape, dtype=torch.bool)
            mask[0, 0, 0, 0] = False
            if q_shape[3] > 1:
                mask[0, 0, 0, 1] = False
            if q_shape[3] > 2:
                mask[0, 0, 0, 2] = False
            normal_part = torch.empty(q_shape, dtype=torch.float32).uniform_(-5.0, 5.0)
            query = torch.where(mask, normal_part, query)

            key = random_tensor(k_shape, torch.float32, seed + 1000)
            value = random_tensor(v_shape, torch.float32, seed + 2000)
        else:
            # 正常 case
            query = random_tensor(q_shape, torch.float32, seed)
            key = random_tensor(k_shape, torch.float32, seed + 1000)
            value = random_tensor(v_shape, torch.float32, seed + 2000)

        input_groups.append([query, key, value])
    return input_groups


def get_init_inputs():
    return []