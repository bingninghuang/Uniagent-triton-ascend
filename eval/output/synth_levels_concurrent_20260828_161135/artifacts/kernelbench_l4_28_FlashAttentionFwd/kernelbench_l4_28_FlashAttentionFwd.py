import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, causal, window_left, window_right, softcap):
        torch.manual_seed(42)
        batch, query_length, n_heads, head_dim = q.shape
        key_length, n_kv_heads = k.shape[1:3]
        if n_heads % n_kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        q_heads = q.transpose(1, 2)
        k_heads = k.transpose(1, 2)
        v_heads = v.transpose(1, 2)
        repeats = n_heads // n_kv_heads
        k_heads = k_heads.repeat_interleave(repeats, dim=1)
        v_heads = v_heads.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))
        scores = scores / math.sqrt(head_dim)
        if softcap > 0.0:
            scores = softcap * torch.tanh(scores / softcap)
        row = torch.arange(query_length, device=q.device).unsqueeze(1)
        column = torch.arange(key_length, device=q.device).unsqueeze(0)
        relative = column - (row + key_length - query_length)
        mask = torch.zeros(
            (query_length, key_length), dtype=torch.bool, device=q.device
        )
        if causal:
            mask = mask | (relative > 0)
        if window_left >= 0:
            mask = mask | (relative < -window_left)
        if window_right >= 0:
            mask = mask | (relative > window_right)
        if causal or window_left >= 0 or window_right >= 0:
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v_heads)
        return output.transpose(1, 2)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    dtype = _DTYPE_MAP[spec["dtype"]]
    if torch.rand(1, generator=generator).item() < 0.5:
        mean = torch.rand(1, generator=generator).item() * 10.0 - 5.0
        std = torch.rand(1, generator=generator).item() * 1.9 + 0.1
        tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        tensor = tensor * std + mean
    else:
        tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
        tensor = tensor * 10.0 - 5.0
    return tensor.to(dtype=dtype).npu()


def _load_cases():
    json_path = os.path.splitext(__file__)[0] + ".json"
    with open(json_path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def _case_specs(case):
    return {item["name"]: item for item in case["inputs"]}


def get_input_groups():
    torch.manual_seed(42)
    input_groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = _case_specs(case)
        input_groups.append([
            _random_tensor(specs["q"], 42 + case_index * 3),
            _random_tensor(specs["k"], 43 + case_index * 3),
            _random_tensor(specs["v"], 44 + case_index * 3),
            specs["causal"]["value"],
            specs["window_left"]["value"],
            specs["window_right"]["value"],
            specs["softcap"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
