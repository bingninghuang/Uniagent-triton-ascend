import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    """CrossFormer Group Attention (Triton Ascend implementation).

    Mirrors the reference Model: same constructor (no args) and same
    forward(x, group_size, q, k, v, num_heads, mask, scale, pos_bias,
    feature_shape) signature.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

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
        device = x.device
        if isinstance(group_size, int):
            gh = gw = int(group_size)
        else:
            gh, gw = int(group_size[0]), int(group_size[1])

        B, N, C = x.shape
        H = int(num_heads)
        head_dim = C // H

        grouped = feature_shape is not None
        if grouped:
            Hf, Wf = int(feature_shape[0]), int(feature_shape[1])
            num_h = Hf // gh
            num_w = Wf // gw
            ng = gh * gw

            def _regroup(t):
                t2 = t.view(B, Hf, Wf, C)
                t_groups = t2.view(B, num_h, gh, num_w, gw, C)
                t_groups = t_groups.permute(0, 1, 3, 2, 4, 5).contiguous()
                return t_groups.view(B * num_h * num_w, ng, C)

            x_flat = _regroup(x)
            q_flat = _regroup(q) if q is not None else x_flat
            k_flat = _regroup(k) if k is not None else x_flat
            v_flat = _regroup(v) if v is not None else x_flat
        else:
            ng = N
            x_flat = x
            q_flat = q if q is not None else x
            k_flat = k if k is not None else x
            v_flat = v if v is not None else x

        out_flat = self._attention(
            x_flat, q_flat, k_flat, v_flat, H, head_dim, mask, scale, pos_bias)

        if grouped:
            out_groups = out_flat.view(B, num_h * num_w, ng, C)
            out_groups = out_groups.view(B, num_h, num_w, gh, gw, C)
            out_groups = out_groups.permute(0, 1, 3, 2, 4, 5).contiguous()
            out = out_groups.view(B, Hf, Wf, C)
            out = out.view(B, Hf * Wf, C)
            return out
        return out_flat

    def _attention(self, x, q, k, v, num_heads, head_dim, mask, scale, pos_bias):
        orig_dtype = x.dtype
        dev = x.device

        Bf, N, C = x.shape
        H = num_heads

        x32 = x.float()
        q32 = q.float()
        k32 = k.float()
        v32 = v.float()

        scale_val = float(scale) if scale is not None else float(head_dim) ** -0.5

        has_bias = pos_bias is not None
        has_mask = mask is not None
        if has_bias:
            pb = pos_bias.float().contiguous()
            bias_3d = pb.dim() == 3
            pb = pb.view(1, N, N) if pb.dim() == 2 else pb
            pb_h = pb.shape[0]
            bias_ptr = pb
        else:
            bias_3d = False
            pb_h = 1
            bias_ptr = x32  # dummy

        if has_mask:
            mk = mask.float().contiguous()
            mask_3d = mk.dim() == 3
            mk = mk.view(1, N, N) if mk.dim() == 2 else mk
            mask_ptr = mk
        else:
            mask_3d = False
            mask_ptr = x32  # dummy

        out = torch.empty(Bf, N, C, dtype=orig_dtype, device=dev)
        out32 = out.view(torch.uint8) if False else out  # placeholder (unused)

        # Launch the fp32 fused-attention kernel and write directly into a
        # preallocated fp32 buffer, then cast back to orig dtype.
        out_fp32 = torch.empty(Bf, N, C, dtype=torch.float32, device=dev)

        q2 = q32.view(Bf, N, H, head_dim)
        k2 = k32.view(Bf, N, H, head_dim)
        v2 = v32.view(Bf, N, H, head_dim)

        BLOCK_D = max(16, _next_pow2(head_dim))
        # Tile sizes
        BLOCK_M = 64
        BLOCK_N = 64
        if N <= 128:
            BLOCK_M = min(BLOCK_M, N)
            BLOCK_N = min(BLOCK_N, _next_pow2(N))
            BLOCK_M = min(BLOCK_M, _next_pow2(N))

        num_m = triton.cdiv(N, BLOCK_M)
        grid = (num_m, Bf * H)

        attn_kernel[grid](
            q2, k2, v2, out_fp32,
            mask_ptr,
            bias_ptr,
            N,
            N,
            scale_val,
            HEADS=H,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            HAS_MASK=has_mask,
            MASK_3D=mask_3d,
            HAS_BIAS=has_bias,
            BIAS_3D=bias_3d,
            NUM_STAGES=2,
            NUM_WARPS=4,
        )

        out.copy_(out_fp32.to(orig_dtype))
        return out


@triton.jit
def attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                msk_ptr,
                bias_ptr,
                N,
                n_tokens,
                scale,
                HEADS: tl.constexpr,
                HEAD_DIM: tl.constexpr,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
                BLOCK_D: tl.constexpr,
                HAS_MASK: tl.constexpr,
                MASK_3D: tl.constexpr,
                HAS_BIAS: tl.constexpr,
                BIAS_3D: tl.constexpr,
                NUM_STAGES: tl.constexpr,
                NUM_WARPS: tl.constexpr):
    # q_ptr/k_ptr/v_ptr/o_ptr have logical shape [Bf, N, HEADS, HEAD_DIM]
    # (contiguous fp32). msk_ptr/bias_ptr: [pb_h, N, N] fp32, broadcast over
    # the batch dimension.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // HEADS
    h = pid_bh % HEADS

    base = b * N * HEADS * HEAD_DIM

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N
    mask_d = offs_d < HEAD_DIM

    q = tl.load(q_ptr + base + offs_m[:, None] * (HEADS * HEAD_DIM)
                + h * HEAD_DIM + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    msk_base = b * 0  # mask does not vary with batch
    bias_base = b * 0

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        k = tl.load(k_ptr + base + offs_n[:, None] * (HEADS * HEAD_DIM)
                    + h * HEAD_DIM + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        s = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N] fp32
        s = s * scale

        mh = h if MASK_3D else 0
        bh = h if BIAS_3D else 0

        if HAS_MASK:
            mval = tl.load(msk_ptr + (mh * N + 0) * N + offs_m[:, None] * N
                           + offs_n[None, :],
                           mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            s = s + mval
        if HAS_BIAS:
            bval = tl.load(bias_ptr + (bh * N + 0) * N + offs_m[:, None] * N
                           + offs_n[None, :],
                           mask=mask_m[:, None] & mask_n[None, :], other=0.0)
            s = s + bval

        s = tl.where(mask_n[None, :], s, -1e30)

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p,
                                            tl.load(v_ptr + base
                                                    + offs_n[:, None] * (HEADS * HEAD_DIM)
                                                    + h * HEAD_DIM + offs_d[None, :],
                                                    mask=mask_n[:, None] & mask_d[None, :],
                                                    other=0.0))
        m_i = m_new

    acc = acc / l_i[:, None]

    tl.store(o_ptr + base + offs_m[:, None] * (HEADS * HEAD_DIM)
             + h * HEAD_DIM + offs_d[None, :],
             acc, mask=mask_m[:, None] & mask_d[None, :])
