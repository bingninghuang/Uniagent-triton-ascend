# 归约操作 (Reduction Operations)

## tl.sum

计算输入 tensor 沿指定轴的元素和。

```python
triton.language.sum(input, axis=None, keep_dims=False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor` | 输入 tensor |
| `axis` | `int` 或 `None` | 沿着哪个维度进行求和。如果为 None，则对所有维度求和 |
| `keep_dims` | `bool` | 如果为 True，保持被求和的维度为长度 1 |

返回值：`tensor`，求和结果。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64。`dtype` 参数当前版本暂未支持（社区 Triton 3.5.0 引入）。`keep_dims=True` 已测 3D dim=2 情况下支持。

```python
@triton.jit
def sum_kernel(in_ptr, out_ptr, xnumel: tl.constexpr, ynumel: tl.constexpr,
               XB: tl.constexpr, YB: tl.constexpr, dim: tl.constexpr):
    xidx = tl.arange(0, XB) + tl.program_id(0) * XB
    yidx = tl.arange(0, YB) + tl.program_id(1) * YB
    idx = xidx[:, None] * ynumel + yidx[None, :]
    x = tl.load(in_ptr + idx)
    ret = tl.sum(x, dim)
    tl.store(out_ptr + (yidx if dim == 0 else xidx), ret)
```

---

## tl.max

在指定维度上返回最大值。

```python
triton.language.max(input, axis=None, return_indices=False,
                     return_indices_tie_break_left=True, keep_dims=False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入的张量数据 |
| `axis` | `int` | 指定在哪个维度上进行规约；axis=None 时在所有轴进行规约 |
| `keep_dims` | `bool` | 保持规约轴规约后的维度 |
| `return_indices` | `bool` | 是否返回最大值所在下标 |
| `return_indices_tie_break_left` | `bool` | 如果多个元素有相同的最大值，返回最左侧最大值的下标 |

返回值：`tl.tensor`。`return_indices=true` 时返回的 index 下标类型是 fp32 类型。

**参数组合支持**：当 `axis=None` 且 `return_indices=True` 时不支持。其余组合均支持。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64。

```python
@triton.jit
def max_kernel(in_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr + xoffset)
    tmp4 = tl.max(tmp0, 0)
    tl.store(out_ptr, tmp4)
```

---

## tl.min

在指定维度上返回最小值。

```python
triton.language.min(input, axis=None, return_indices=False,
                    return_indices_tie_break_left=True, keep_dims=False)
```

参数与 `tl.max` 相同。

**DataType 支持 (Ascend)**：同 `tl.max`。

---

## tl.argmax

在指定维度上返回最大值所在的下标。

```python
triton.language.argmax(input, axis, tie_break_left=True, keep_dims=False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 张量数据 |
| `axis` | `int` | 指定在哪个维度上进行规约 |
| `keep_dims` | `bool` | 保持规约轴规约后的维度 |
| `tie_break_left` | `bool` | 如果多个元素有相同的最大值，返回最左侧最大值的下标 |

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64。

**特殊取值**：对于 `tensor[nan, inf]` 的情况，返回 inf 所在的下标。

```python
@triton.jit
def argmax_kernel(in_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) + tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr + xoffset)
    tmp4 = tl.argmax(tmp0, 0)
    tl.store(out_ptr, tmp4)
```

---

## tl.argmin

在指定维度上返回最小值所在的下标。

```python
triton.language.argmin(input, axis, tie_break_left=True, keep_dims=False)
```

参数与 `tl.argmax` 相同。

**DataType 支持 (Ascend)**：同 `tl.argmax`。

---

## tl.reduce

沿指定轴 `axis` 对输入 tensor 应用 `combine_fn` 进行规约。

```python
triton.language.reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor` 或 `tuple of Tensor` | 输入 tensor，可以是单个 tensor 或 tensor 元组 |
| `axis` | `int` 或 `None` | 沿着哪个维度进行 reduce 操作。如果为 None，则 reduce 所有维度 |
| `combine_fn` | `Callable` | 用于组合两个标量 tensor 组的函数（必须用 @triton.jit 标记） |
| `keep_dims` | `bool` | 如果为 True，保持被 reduce 的维度为长度 1 |

返回值：`tensor`，规约后的结果张量。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16, bool。

```python
@triton.jit
def _reduce_combine(a, b):
    return a + b

@triton.jit
def reduce_kernel(in_ptr, out_ptr, dim: tl.constexpr, XB: tl.constexpr, YB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    idx = xidx[:, None] * YB + yidx[None, :]
    x = tl.load(in_ptr + idx)
    ret = tl.reduce(x, dim, _reduce_combine)
    tl.store(out_ptr + (yidx if dim == 0 else xidx), ret)
```

---

## tl.xor_sum

计算输入 tensor 沿指定轴的异或和。

```python
triton.language.xor_sum(input, axis=None, keep_dims=False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor` | 输入 tensor |
| `axis` | `int` 或 `None` | 沿着哪个维度进行异或和操作 |
| `keep_dims` | `bool` | 如果为 True，保持被操作的维度为长度 1 |

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp16/fp32/bf16, fp64。

```python
@triton.jit
def xor_sum_kernel(in_ptr, out_ptr, dim: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    idx = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    x = tl.load(in_ptr + idx)
    tmp4 = tl.xor_sum(x, axis=dim)
    tl.store(out_ptr + tl.arange(0, N), tmp4)
```

---

## Ascend 通用限制总结

- **不支持 fp64**：所有归约操作在 Ascend 上均不支持 fp64。
- **不支持 uint16/uint32/uint64**：所有归约操作在 Ascend 上不支持这些类型。
- **keep_dims=True**：需要测试更多规格来确定全面支持。目前已测 3D dim=2 情况下支持。
- **return_indices 与 axis=None**：当 axis=None 且 return_indices=True 时不支持。
