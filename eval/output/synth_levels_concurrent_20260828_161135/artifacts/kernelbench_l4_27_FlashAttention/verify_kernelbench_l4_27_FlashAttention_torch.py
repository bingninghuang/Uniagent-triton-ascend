import json
import math
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def _layers(self, x, n_heads):
        d_model = x.shape[-1]
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        key = (d_model, n_heads, x.device, x.dtype)
        if key not in self._cache:
            self._cache.clear()
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            self._cache[key] = tuple(
                nn.Linear(d_model, d_model, bias=False).to(
                    device=x.device, dtype=x.dtype
                )
                for _ in range(4)
            )
            torch.set_rng_state(rng_state)
        return self._cache[key]

    def _tiled_attention(self, q, k, v, block_size_q, block_size_kv):
        scale = 1.0 / math.sqrt(q.shape[-1])
        query_length = q.shape[-2]
        key_length = k.shape[-2]
        chunks = []
        for query_start in range(0, query_length, block_size_q):
            query_block = q[
                :, :, query_start : query_start + block_size_q
            ]
            block_shape = query_block.shape[:-1]
            running_max = torch.full(
                block_shape,
                float("-inf"),
                device=q.device,
                dtype=torch.float32,
            )
            normalizer = torch.zeros(
                block_shape, device=q.device, dtype=torch.float32
            )
            accumulator = torch.zeros_like(query_block)
            for key_start in range(0, key_length, block_size_kv):
                key_block = k[:, :, key_start : key_start + block_size_kv]
                value_block = v[
                    :, :, key_start : key_start + block_size_kv
                ]
                scores = torch.matmul(
                    query_block, key_block.transpose(-2, -1)
                ) * scale
                block_max = scores.max(dim=-1).values
                next_max = torch.maximum(running_max, block_max)
                correction = torch.exp(running_max - next_max)
                probabilities = torch.exp(scores - next_max.unsqueeze(-1))
                accumulator = (
                    accumulator * correction.unsqueeze(-1)
                    + torch.matmul(
                        probabilities.to(dtype=value_block.dtype), value_block
                    )
                )
                normalizer = (
                    normalizer * correction + probabilities.sum(dim=-1)
                )
                running_max = next_max
            chunks.append(accumulator / normalizer.clamp_min(1e-6).unsqueeze(-1))
        return torch.cat(chunks, dim=-2)

    def forward(self, x, n_heads, block_size_q, block_size_kv):
        torch.manual_seed(42)
        if block_size_q <= 0 or block_size_kv <= 0:
            raise ValueError("block sizes must be positive")
        q_proj, k_proj, v_proj, out_proj = self._layers(x, n_heads)
        batch, sequence, d_model = x.shape
        head_dim = d_model // n_heads
        q = q_proj(x).view(batch, sequence, n_heads, head_dim).transpose(1, 2)
        k = k_proj(x).view(batch, sequence, n_heads, head_dim).transpose(1, 2)
        v = v_proj(x).view(batch, sequence, n_heads, head_dim).transpose(1, 2)
        output = self._tiled_attention(
            q, k, v, block_size_q, block_size_kv
        ).to(dtype=x.dtype)
        output = output.transpose(1, 2).contiguous().view(batch, sequence, d_model)
        return out_proj(output)


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
            _random_tensor(specs["x"], 42 + case_index),
            specs["n_heads"]["value"],
            specs["block_size_q"]["value"],
            specs["block_size_kv"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
