# 形状操作 (Shape Manipulation Operations)

## tl.broadcast

将两个张量广播到共同兼容的形状，使它们可以进行逐元素操作。

```python
triton.language.broadcast(input, other)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 第一个输入张量 |
| `other` | `tensor` | 第二个输入张量 |

返回值：`tensor`，两个 tensor 共同兼容的目标形状。每个返回的张量保持其输入的原始数据类型。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64, fp8。

```python
@triton.jit
def broadcast_kernel(output_ptr, BLOCK_SIZE: tl.constexpr):
    scalar = 5.0
    vector = tl.arange(0, BLOCK_SIZE) * 1.0
    broadcasted_scalar = tl.broadcast(scalar, vector)
    result = vector + broadcasted_scalar
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + offsets, result)
```

---

## tl.broadcast_to

将张量广播到目标形状，自动处理维度对齐。

```python
# 函数调用形式
triton.language.broadcast_to(input, shape)

# 成员函数形式
input.broadcast_to(shape)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `shape` | `List[int]` | 目标形状 |

返回值：`tensor`，与 shape 参数指定的目标形状相同。

**约束**：输入张量的维度数必须等于目标形状的维度数。所有维度必须满足广播规则。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。

```python
@triton.jit
def matrix_add_bias_kernel(x_ptr, bias_ptr, output_ptr, M, N,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr)
    bias_broadcast = bias.broadcast_to([BLOCK_M, BLOCK_N])
    output = x + bias_broadcast
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## tl.expand_dims

在指定轴位置插入大小为1的维度，不改变张量的数据，仅增加维度数。支持负索引。

```python
# 函数调用形式
triton.language.expand_dims(input, axis)

# 成员函数形式
input.expand_dims(axis)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `axis` | `int` 或 `Tuple[int]` | 插入维度的位置，支持负索引 |

返回值：`tensor`，在指定 axis 位置插入大小为1的维度。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。

```python
@triton.jit
def expand_dims_example(out_ptr):
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.expand_dims(x, axis=1)  # 变成 2x1x3
```

---

## tl.interleave

将两个相同形状的输入张量在最后一个维度上交错排列，输出张量的最后一个维度大小为输入张量的2倍。

```python
# 函数调用形式
triton.language.interleave(x, y)

# 成员函数形式
x.interleave(y)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 第一个输入张量 |
| `y` | `tensor` | 第二个输入张量，形状必须与 x 相同 |

返回值：`tensor`，输入形状的最后一个维度乘以2。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。

```python
@triton.jit
def interleave_example():
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.ones([2, 3], dtype=tl.float32)
    z = tl.interleave(x, y)  # 变成 2x6
```

---

## tl.join

将两个相同形状的输入张量沿着新的最小维度连接，输出张量比输入张量多一个维度，大小为2。

```python
# 函数调用形式
triton.language.join(x, y)

# 成员函数形式
x.join(y)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 第一个输入张量 |
| `y` | `tensor` | 第二个输入张量，形状必须可广播到与 x 相同 |

返回值：`tensor`，输入 tensor 广播后的形状加上一个大小为2的维度。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。

```python
@triton.jit
def join_example(out_ptr):
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.full([2, 3], 1.0, dtype=tl.float32)
    z = tl.join(x, y)  # 变成 2x2x3
```

---

## tl.permute

根据 dims 参数重新排列张量的维度，不改变张量的数据，仅改变维度的顺序。

```python
# 函数调用形式
triton.language.permute(input, dims)

# 成员函数形式
input.permute(dims)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `dims` | `List[int]` | 新的维度顺序 |

返回值：`tensor`，按照 dims 参数重新排列的维度。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持维度高于8的转置。

```python
@triton.jit
def permute_example(out_ptr):
    x = tl.zeros([2, 3, 4], dtype=tl.float32)
    y = tl.permute(x, [2, 0, 1])  # 变成 4x2x3
```

---

## tl.ravel

将输入张量展平为一维张量，保持元素在内存中的顺序。

```python
# 函数调用形式
triton.language.ravel(input)

# 成员函数形式
input.ravel()
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |

返回值：`tensor`，一维张量，包含输入张量的所有元素。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。

```python
@triton.jit
def flatten_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask)
    x_flat = x.ravel()
    tl.store(output_ptr + offsets, x_flat, mask=mask)
```

---

## tl.reshape

将张量重新解释为新的形状。

```python
# 函数调用形式
triton.language.reshape(input, shape, can_reorder=False)

# 成员函数形式
input.reshape(shape, can_reorder=False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `shape` | `List[int]` | 目标形状 |
| `can_reorder` | `bool` | 是否允许重新排序元素，默认 False |

返回值：`tensor`，与 shape 参数指定的目标形状相同。

**约束**：输入和输出张量的总元素数必须相等。所有 tensor 不允许某个 shape 的 size 小于1。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**Ascend 限制**：`can_reorder` 参数仅支持 False。

```python
@triton.jit
def reshape_example(out_ptr):
    x = tl.zeros([2, 3, 4], dtype=tl.float32)
    y = tl.reshape(x, [6, 4])  # 变成 6x4
```

---

## tl.split

将输入张量沿着最后一个维度分割成两个张量。

```python
# 函数调用形式
triton.language.split(input)

# 成员函数形式
input.split()
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量，最后一个维度大小必须为偶数 |

返回值：`Tuple[tensor, tensor]`，两个张量，形状相同，最后一个维度为输入的一半。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。

```python
@triton.jit
def complex_split_kernel(complex_ptr, real_ptr, imag_ptr, M, N,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    complex_data = tl.load(complex_ptr + offsets, mask=mask)
    real_part, imag_part = complex_data.split()
    tl.store(real_ptr + offsets, real_part, mask=mask)
    tl.store(imag_ptr + offsets, imag_part, mask=mask)
```

---

## tl.trans

根据 dims 参数转置张量的维度，不改变张量的数据，仅改变维度的顺序。专门优化的转置操作。

```python
# 函数调用形式
triton.language.trans(input, dims)

# 成员函数形式
input.trans(dims)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `dims` | `List[int]` | 转置后的维度顺序 |

返回值：`tensor`，按照 dims 参数重新排列的维度。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持维度高于8的转置。

```python
@triton.jit
def trans_example():
    x = tl.zeros([2, 3, 4], dtype=tl.float32)
    y = tl.trans(x, [2, 0, 1])  # 变成 4x2x3
```

---

## tl.view

创建张量的视图，改变形状但不复制数据，类似于 reshape，但更强调视图的概念。

```python
# 函数调用形式
triton.language.view(input, shape)

# 成员函数形式
input.view(shape)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `shape` | `List[int]` | 目标形状 |

返回值：`tensor`，与 shape 参数指定的目标形状相同。

**约束**：输入和输出张量的总元素数必须相等。输出张量必须与输入张量在内存中连续。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。

```python
@triton.jit
def view_example(out_ptr):
    x = tl.zeros([2, 3, 4], dtype=tl.float32)
    y = tl.view(x, [6, 4])  # 变成 6x4
```

---

## Ascend 通用限制总结

- **不支持 uint16/uint32/uint64, fp64, fp8**：所有形状操作在 Ascend 上不支持这些类型。
- **reshape can_reorder**：仅支持 False。
- **permute/trans 维度限制**：不支持维度高于8的转置。
- **broadcast_to 约束**：输入张量的维度数必须等于目标形状的维度数。
