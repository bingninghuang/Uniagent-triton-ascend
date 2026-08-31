import torch
import triton
import triton.language as tl


@triton.jit
def router_topk_kernel(
    x_ptr,        # (T, E) gate logits, in dtype
    ids_ptr,      # (T, K) topk ids, int64
    vals_ptr,     # (T, K) topk values, in dtype
    T,
    E,
    K: tl.constexpr,
    N: tl.constexpr,  # next_power_of_2(E)
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, N)
    col_mask = cols < E
    # masked lanes -> -inf so they never win the max/argmax selection
    x = tl.load(x_ptr + pid * E + cols, mask=col_mask, other=float("-inf")).to(tl.float32)

    # softmax statistics in fp32 (max + normalizer)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = x  # selection space = logits (monotone transform of softmax probs)

    out_base = pid * K
    for r in range(K):
        v, i = tl.max(p, axis=0, return_indices=True, return_indices_tie_break_left=True)
        i32 = i.to(tl.int32)
        val = tl.exp(v - m) / s
        tl.store(ids_ptr + out_base + r, i32.to(tl.int64))
        tl.store(vals_ptr + out_base + r, val.to(x_ptr.dtype.element_ty))
        p = tl.where(cols == i32, float("-inf"), p)


class ModelNew:
    def __init__(self):
        self._cache = {}

    def forward(self, gate_logits: torch.Tensor, topk: int):
        gate_logits = gate_logits.contiguous()
        T, E = gate_logits.shape
        K = int(topk)
        N = triton.next_power_of_2(E)

        ids = torch.empty((T, K), dtype=torch.int64, device=gate_logits.device)
        vals = torch.empty((T, K), dtype=gate_logits.dtype, device=gate_logits.device)

        router_topk_kernel[(T,)](
            gate_logits, ids, vals,
            T, E,
            K=K, N=N,
        )
        return ids, vals
