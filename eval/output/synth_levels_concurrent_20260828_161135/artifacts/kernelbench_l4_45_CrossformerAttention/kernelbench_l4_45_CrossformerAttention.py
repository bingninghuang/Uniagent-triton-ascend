import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    CrossFormer Group Attention – pure function, no trainable params.
    Supports two modes:
      1. Single group: input x is already flattened into a single group (N = gh*gw).
      2. Multi-group: provide feature_shape (H, W) to reshape x into a 2D grid,
         split into groups of size (gh, gw), and run attention independently.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor,
                group_size,
                q: torch.Tensor = None,
                k: torch.Tensor = None,
                v: torch.Tensor = None,
                num_heads: int = 1,
                mask: torch.Tensor = None,
                scale: float = None,
                pos_bias: torch.Tensor = None,
                feature_shape: tuple = None) -> torch.Tensor:
        """
        Args:
            x: [batch, N, channels] (if feature_shape=None) or [batch, H*W, channels] (if provided)
            group_size: int or tuple (gh, gw)
            q,k,v: optional custom Q/K/V tensors (same shape as x)
            num_heads: number of heads
            mask: optional mask, broadcastable to (B, num_heads, N_group, N_group)
            scale: scaling factor (default: 1/sqrt(head_dim))
            pos_bias: optional position bias, shape (N_group, N_group) or (num_heads, N_group, N_group)
            feature_shape: (H, W) if x is a flattened feature map, else None
        Returns:
            out: [batch, N_original, channels]
        """
        B, N, C = x.shape

        if isinstance(group_size, int):
            gh = gw = group_size
        else:
            gh, gw = group_size
        if feature_shape is not None:
            H, W = feature_shape
            # 确保 H*W == N
            # 将 x 重塑为 (B, H, W, C)
            x_2d = x.view(B, H, W, C)
            # 计算组数
            num_h = H // gh
            num_w = W // gw
            x_groups = x_2d.view(B, num_h, gh, num_w, gw, C)
            # 交换维度为 (B, num_h, num_w, gh*gw, C)
            x_groups = x_groups.permute(0, 1, 3, 2, 4, 5).contiguous()
            x_groups = x_groups.view(B, num_h * num_w, gh * gw, C)  # (B, total_groups, group_N, C)
            B_flat = B * num_h * num_w
            x_flat = x_groups.view(B_flat, gh * gw, C)

            if q is not None:
                q_2d = q.view(B, H, W, C)
                q_groups = q_2d.view(B, num_h, gh, num_w, gw, C).permute(0,1,3,2,4,5).contiguous()
                q_flat = q_groups.view(B_flat, gh*gw, C)
            else:
                q_flat = x_flat
            if k is not None:
                k_2d = k.view(B, H, W, C)
                k_groups = k_2d.view(B, num_h, gh, num_w, gw, C).permute(0,1,3,2,4,5).contiguous()
                k_flat = k_groups.view(B_flat, gh*gw, C)
            else:
                k_flat = x_flat
            if v is not None:
                v_2d = v.view(B, H, W, C)
                v_groups = v_2d.view(B, num_h, gh, num_w, gw, C).permute(0,1,3,2,4,5).contiguous()
                v_flat = v_groups.view(B_flat, gh*gw, C)
            else:
                v_flat = x_flat
            out_flat = self._attention(x_flat, q_flat, k_flat, v_flat, num_heads, mask, scale, pos_bias)
            # 恢复形状：out_flat (B_flat, group_N, C) -> (B, total_groups, group_N, C) -> (B, H, W, C) -> (B, N, C)
            out_groups = out_flat.view(B, num_h * num_w, gh*gw, C)
            out_groups = out_groups.view(B, num_h, num_w, gh, gw, C)
            out_groups = out_groups.permute(0, 1, 3, 2, 4, 5).contiguous()
            out = out_groups.view(B, H, W, C)
            out = out.view(B, H*W, C)
            return out
        else:
            # 单组模式
            q_in = x if q is None else q
            k_in = x if k is None else k
            v_in = x if v is None else v
            out = self._attention(x, q_in, k_in, v_in, num_heads, mask, scale, pos_bias)
            return out

    def _attention(self, x, q, k, v, num_heads, mask, scale, pos_bias):
        # NPU matmul/softmax reduced precision drifts; compute accurate fp32 reference on NPU.
        orig_dtype = x.dtype
        x = x.float()
        q = q.float()
        k = k.float()
        v = v.float()
        if mask is not None:
            mask = mask.float()
        if pos_bias is not None:
            pos_bias = pos_bias.float()

        B, N, C = x.shape
        head_dim = C // num_heads

        def to_heads(t):
            return t.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        qh = to_heads(q)
        kh = to_heads(k)
        vh = to_heads(v)

        scale = scale if scale is not None else (head_dim ** -0.5)
        qh = qh * scale

        attn = torch.matmul(qh, kh.transpose(-2, -1))

        if pos_bias is not None:
            if pos_bias.dim() == 2:
                attn = attn + pos_bias.unsqueeze(0).unsqueeze(0)
            else:
                attn = attn + pos_bias.unsqueeze(0)
        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        out_heads = torch.matmul(attn, vh)
        out = out_heads.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return out.to(orig_dtype)
def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "45_CrossformerAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = dtype_map[x_info["dtype"]]
        x = random_tensor(x_info["shape"], dtype)
        group_size_info = next((inp for inp in inputs if inp["name"] == "group_size"), None)
        group_size = group_size_info["value"] if group_size_info else None
        q_info = next((inp for inp in inputs if inp["name"] == "q"), None)
        q = random_tensor(q_info["shape"], dtype) if q_info else None
        k_info = next((inp for inp in inputs if inp["name"] == "k"), None)
        k = random_tensor(k_info["shape"], dtype) if k_info else None
        v_info = next((inp for inp in inputs if inp["name"] == "v"), None)
        v = random_tensor(v_info["shape"], dtype) if v_info else None
        input_groups.append([x, group_size, q, k, v])
    return input_groups

def get_init_inputs():
    return []