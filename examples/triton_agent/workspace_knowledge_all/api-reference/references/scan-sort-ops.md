# 扫描/排序操作 (Scan & Sort Operations)

## tl.associative_scan

对输入 tensor 沿指定轴应用关联扫描操作，使用 combine_fn 函数组合元素并更新进位值。

```python
triton.language.associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor` 或 `tuple of Tensor` | 输入 tensor，可以是单个 tensor 或 tensor 元组 |
| `axis` | `int` | 沿着哪个维度进行关联扫描操作 |
| `combine_fn` | `Callable` | 用于组合两个标量 tensor 组的函数（必须用 @triton.jit 标记） |
| `reverse` | `bool` | 是否沿轴的反方向应用关联扫描 |

返回值：`tensor`，扫描操作后的结果。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64。

**reverse=True 限制**：需要 tl.load 加载数据时对齐，即不使用 mask 过滤掉多余数据索引。

```python
@triton.jit
def add_fn(a, b):
    return a + b

@triton.jit
def scan_kernel(in_ptr, out_ptr, dim: tl.constexpr, XB: tl.constexpr, RB: tl.constexpr):
    idx = tl.arange(0, XB)[:, None] * RB + tl.arange(0, RB)[None, :]
    x = tl.load(in_ptr + idx)
    ret = tl.associative_scan(x, axis=dim, combine_fn=add_fn)
    tl.store(out_ptr + idx, ret)
```

---

## tl.cumsum

计算输入 tensor 沿指定轴的累积和（前缀和）。

```python
triton.language.cumsum(input, axis=0, reverse=False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor` | 输入 tensor |
| `axis` | `int` | 沿着哪个维度进行累积和操作，默认为 0 |
| `reverse` | `bool` | 如果为 True，沿反方向进行累积和操作 |

对于输入 `[a, b, c, d]`，累积和结果为 `[a, a+b, a+b+c, a+b+c+d]`。当 `reverse=True` 时：`[a+b+c+d, b+c+d, c+d, d]`。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16, bool。

```python
@triton.jit
def cumsum_kernel(in_ptr, out_ptr, dim: tl.constexpr, reverse: tl.constexpr,
                  XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    idx = tl.arange(0, XBLOCK)[:, None] * RBLOCK + tl.arange(0, RBLOCK)[None, :]
    x = tl.load(in_ptr + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr + idx, ret)
```

---

## tl.cumprod

计算输入 tensor 沿指定轴的累积乘积（前缀乘积）。

```python
triton.language.cumprod(input, axis=0, reverse=False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor` | 输入 tensor |
| `axis` | `int` | 沿着哪个维度进行累积乘积操作，默认为 0 |
| `reverse` | `bool` | 如果为 True，沿反方向进行累积乘积操作 |

对于输入 `[a, b, c, d]`，累积乘积结果为 `[a, a*b, a*b*c, a*b*c*d]`。

**DataType 支持 (Ascend)**：同 cumsum。

**注意**：cumprod 没有 `dtype` 参数，使用时需注意数据类型的溢出问题。

---

## tl.histogram

基于 input 计算具有 num_bins 个 bin 的直方图，每个 bin 宽度为 1，起始于 0。

```python
triton.language.histogram(input, num_bins, mask=None, _semantic=None, _generator=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入数据，包含需要统计分布的所有数值点 |
| `num_bins` | `int` | 定义要将整个数据范围划分成多少个等宽的区间 |
| `mask` | `int1` 或 `tensor<int1>` | 可选，指定数据范围，防止访问越界 |

返回值：用 tensor 表示的直方图。

**DataType 支持 (Ascend)**：int32, uint32, uint64, int64。
**Shape 限制**：目前仅支持一维。
**输入范围**：当前限制在 [0, num_bins-1] 中，待版本更新后支持全范围。
**mask 限制**：当前 triton3.2 版本暂未支持 mask。

```python
@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)
```

---

## tl.sort

对输入张量 x 按维度进行升序或降序的排序。

```python
triton.language.sort(x, dim: constexpr | None = None, descending: constexpr = False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据 |
| `dim` | `int` | 排序维度 |
| `descending` | `bool` | 是否降序 |

返回值：与输入 x 相同 shape 的张量。

**DataType 支持 (Ascend)**：int8, int16, fp16, fp32, bf16。
**Ascend 限制**：不支持 int32, uint8, int64, fp64, bool（毕升编译器限制）。

```python
@triton.jit
def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    pid = tl.program_id(0)
    offx = tl.arange(0, M)
    off2d = offx + pid * M
    x = tl.load(X + off2d)
    x = tl.sort(x, dim=0, descending=descending)
    tl.store(Z + off2d, x)
```

---

## tl.topk

返回输入张量 x 沿指定维度的前 k 个最大元素，返回结果按从大到小排序。

```python
triton.language.topk(x, k, dim: constexpr | None = None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 输入张量 |
| `k` | `int` | 要返回的 top 元素数量，**必须是 2 的幂** |
| `dim` | `constexpr int` 或 `None` | 要查找 top k 元素的维度。如果为 None，则使用最后一个维度。**当前仅支持最后一个维度** |

返回值：输出张量的 shape 与输入张量一致，但指定维度长度变为 k。

**DataType 支持 (Ascend)**：int8, int16, fp16, fp32, bf16。
**Ascend 限制**：不支持 int32, uint8, int64, fp64, bool。当前仅返回最大值，不支持返回最小值。`dim` 仅支持最后一个维度。`k` 必须为 2 的幂。

```python
@triton.jit
def topk_kernel(X, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, N)
    offs = pid * N + offs_n
    x = tl.load(X + offs)
    z = tl.topk(x, K, dim=0)
    tl.store(Z + pid * K + tl.arange(0, K), z)
```

---

## Ascend 通用限制总结

- **不支持 uint16/uint32/uint64**：所有扫描排序操作在 Ascend 上不支持这些类型。
- **sort/topk 限制**：不支持 int32, uint8, int64, fp64, bool（毕升编译器限制）。
- **topk 限制**：仅返回最大值，k 必须为 2 的幂，dim 仅支持最后一个维度。
- **reverse=True**：associative_scan 的 reverse 功能需要 tl.load 加载数据时对齐。
