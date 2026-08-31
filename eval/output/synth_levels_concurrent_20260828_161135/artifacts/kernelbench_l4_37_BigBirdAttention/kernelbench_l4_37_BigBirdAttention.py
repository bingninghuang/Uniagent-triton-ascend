import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    BigBird Attention (pure function, no trainable params).
    Accepts Q/K/V in BNSD layout and applies hybrid sparse attention mask.
    """

    def __init__(self):
        super(Model, self).__init__()

    def _create_bigbird_mask(self, seq_len_q: int, seq_len_k: int,
                             window_size: int, num_random_blocks: int,
                             device: torch.device) -> torch.Tensor:
        """
        Create BigBird mask for cross-attention (seq_len_q x seq_len_k).
        Returns: mask with 1 for allowed, 0 for masked (shape: [seq_len_q, seq_len_k]).
        """
        mask = torch.zeros(seq_len_q, seq_len_k, device=device)

        # 1. Local window (assuming square attention for simplicity)
        # For simplicity, we keep the same window for Q and K.
        # If seq_len_q != seq_len_k, we adapt window using q positions.
        for i in range(seq_len_q):
            start = max(0, i - window_size // 2)
            end = min(seq_len_k, i + window_size // 2 + 1)
            mask[i, start:end] = 1

        # 2. Global tokens: first and last tokens of Q attend to all K,
        # and all Q attend to first and last K.
        if seq_len_q > 0:
            mask[0, :] = 1
            mask[-1, :] = 1
        if seq_len_k > 0:
            mask[:, 0] = 1
            mask[:, -1] = 1

        # 3. Random attention for each Q position (except global ones)
        # Use a deterministic generator seeded from parameters so the reference
        # output is consistent across repeated forward calls.
        if num_random_blocks > 0:
            seed = seq_len_q + seq_len_k * 10007 + window_size * 131 + num_random_blocks * 17
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            for i in range(1, seq_len_q - 1):
                # sample random K positions (without replacement)
                if seq_len_k <= num_random_blocks:
                    # If K is short, attend to all
                    mask[i, :] = 1
                else:
                    # Ensure we don't include global tokens? Original does not exclude them.
                    # We'll just sample from all positions.
                    rand_indices = torch.randperm(seq_len_k, device=device, generator=generator)[:num_random_blocks]
                    mask[i, rand_indices] = 1

        return mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                window_size: int, num_random_blocks: int,
                scale: float = None, pse: torch.Tensor = None,
                sink: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: [batch, heads, seq_len_q, d_k]
            key:   [batch, heads, seq_len_k, d_k]
            value: [batch, heads, seq_len_k, d_v]
            window_size: size of local window (centered)
            num_random_blocks: number of random tokens per query position
            scale: optional scaling factor (default 1/sqrt(d_k))
            pse: optional positional encoding (broadcastable to scores)
            sink: optional per-head bias [heads]

        Returns:
            output: [batch, heads, seq_len_q, d_v]
        """
        b, n, sq, d_k = query.shape
        _, _, skv, _ = key.shape
        _, _, _, dv = value.shape
        # Compute scores in float32 to avoid low-precision accumulation drift.
        query_f = query.float()
        key_f = key.float()
        value_f = value.float()
        scores = torch.matmul(query_f, key_f.transpose(-2, -1))
        scale = scale or (1.0 / (d_k ** 0.5))
        scores = scores * scale

        # Create BigBird mask (1 = keep, 0 = mask)
        mask = self._create_bigbird_mask(sq, skv, window_size, num_random_blocks, query.device)
        # Expand to [1, 1, sq, skv] for broadcasting
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Optional pse / sink (same as before)
        if pse is not None:
            pse = pse.float()
            if pse.dim() == 4:
                scores = scores + pse
            elif pse.dim() == 2:
                scores = scores + pse.view(1, 1, 1, -1)
            else:
                scores = scores + pse

        if sink is not None:
            sink = sink.float()
            scores = scores + sink.view(1, -1, 1, 1)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value_f)
        return output.to(query.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "37_BigBirdAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        query_info = next(inp for inp in inputs if inp["name"] == "query")
        key_info = next(inp for inp in inputs if inp["name"] == "key")
        value_info = next(inp for inp in inputs if inp["name"] == "value")
        dtype = dtype_map[query_info["dtype"]]

        query = random_tensor(query_info["shape"], dtype)
        key = random_tensor(key_info["shape"], dtype)
        value = random_tensor(value_info["shape"], dtype)

        # Mandatory attributes
        window_size = None
        num_random_blocks = None
        scale = None
        pse = None
        sink = None

        for inp in inputs[3:]:
            name = inp.get("name", "")
            if name == "window_size":
                window_size = inp["value"]
            elif name == "num_random_blocks":
                num_random_blocks = inp["value"]
            elif name == "scale":
                scale = inp["value"]
            elif name == "pse":
                pse = random_tensor(inp["shape"], dtype)
            elif name == "sink":
                sink = random_tensor(inp["shape"], torch.float32)

        # Required params (assumed valid)

        input_groups.append([query, key, value, window_size, num_random_blocks,
                             scale, pse, sink])
    return input_groups


def get_init_inputs():
    return []
