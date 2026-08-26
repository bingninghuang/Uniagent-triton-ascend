# 创建操作 (Creation Operations)

## tl.arange

生成一个从 start 到 end（不包括 end）的连续整数序列。

```python
triton.language.arange(start, end, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `start` | `scalar` (constexpr) | 创建连续整数序列的起始数值，必须是编译时常量 |
| `end` | `scalar` | 创建连续整数序列的结束数值 |

返回值：`tensor`，连续整数序列的 tensor。

**DataType 支持**：start 和 end 必须是 constant 常量，支持 uint、int 类型的小于 1048576 的数值。int64 不支持。
**Shape 限制**：`0 <= (end - start) < 1048576`，`end >= 0, start >= 0`。

**特殊说明**：
1. CUDA 要求 range=(end-start) 必须为 2 的幂次方，Triton-Ascend 无此要求。
2. NV 和 Triton-Ascend 都限制 end 的最大值 TRITON_MAX_TENSOR_NUMEL = 1048576。

```python
@triton.jit
def arange_kernel(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(0, BLOCK)
    val = tl.arange(START, END)
    tl.store(z + off, val)
```

---

## tl.zeros

返回用给定形状和 dtype 的标量值 0 填充的张量。

```python
triton.language.zeros(shape, dtype)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `shape` | `tuple of ints` | 新数组的形状，例如 (8, 16) 或 (8,) |
| `dtype` | `tl.dtype` | 新数组的数据类型，例如 tl.float16 |

返回值：`tensor`，用 0 填充的张量。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64, bool。

```python
ret = tl.zeros((XB, YB, ZB), dtype=tl.float32)
```

---

## tl.zeros_like

返回与给定张量具有相同形状和类型的零的张量。

```python
triton.language.zeros_like(input)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor` | 输入 tensor |

返回值：与 `input` 相同 shape 和 dtype 的零张量。

**DataType 支持 (Ascend)**：同 `tl.zeros`。

---

## tl.full

返回一个填充了给定形状和数据类型的标量值的张量。

```python
triton.language.full(shape, value, dtype, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `shape` | `tuple of ints` | 新数组的形状 |
| `value` | `scalar` | 用于填充数组的标量值 |
| `dtype` | `tl.dtype` | 新数组的数据类型 |

返回值：`tensor`，完成填充之后的 tensor。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16, bool。

```python
ret = tl.full((XB, YB, ZB), value=100, dtype=tl.float32)
```

---

## tl.cast

将张量转换为指定的数据类型。

```python
# 函数调用形式
triton.language.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)

# 成员函数形式
input.cast(dtype, fp_downcast_rounding=None, bitcast=False)
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `input` | `tensor` | 是 | 输入张量 |
| `dtype` | `tl.dtype` | 是 | 目标数据类型 |
| `fp_downcast_rounding` | `str` | 否 | 仅对浮点降精度有效，`rtne`（默认，四舍六入五成双）或 `rtz`（向零） |
| `bitcast` | `bool` | 否 | 是否执行位级别重解释，默认 False |
| `overflow_mode` | `str` | 否 | Ascend 扩展：整数溢出处理，`trunc`（截断，默认）或 `saturate`（饱和） |

返回值：`tensor`，与输入张量相同 shape，dtype 由参数指定。

**功能**：
- 数值类型转换：整型 <-> 整型、浮点 <-> 浮点、整型 <-> 浮点
- 位级别重解释（bitcast）：不改变比特，只改变解释类型
- 浮点降精度支持舍入模式：`rtne`（默认）、`rtz`
- 整数转换（Ascend 扩展）支持溢出模式：`trunc`（默认）、`saturate`

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64, float8e4/float8e5。

```python
# 基本类型转换
y = tl.cast(x, tl.int32)

# 位级别重解释
y = x.cast(tl.int32, bitcast=True)

# 浮点降精度，向零舍入
z = x.cast(tl.float16, fp_downcast_rounding="rtz")

# float32 -> int8，启用饱和模式（Ascend 扩展）
w = x.cast(tl.int8, overflow_mode="saturate")
```

---

## tl.cat

将指定的 tensor 进行拼接。

```python
triton.language.cat(input, other, can_reorder=False, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `Tensor` | 拼接的第一个 tensor |
| `other` | `Tensor` | 拼接的第二个 tensor |
| `can_reorder` | `Bool` | 重新排序编译器提示。如果为真，编译器在连接输入时允许重新排序元素。**仅支持 can_reorder=True** |
| `_semantic` | `Optional[str]` | 保留参数 |

返回值：`tensor`，完成拼接之后的 tensor。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64。

**特殊说明**：
1. ASCEND 和 CUDA 都只支持 `can_reorder=True`。
2. cat 只支持 1D shape 的拼接。

```python
@triton.jit
def cat_kernel(output_ptr, x_ptr, y_ptr, XB: tl.constexpr):
    idx = tl.arange(0, XB)
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)
    ret = tl.cat(X, Y, can_reorder=True)
    oidx = tl.arange(0, XB * 2)
    tl.store(output_ptr + oidx, ret)
```
