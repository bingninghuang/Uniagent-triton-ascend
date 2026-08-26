# 数学操作 (Math Operations)

## 四则运算

### tl.add / `+`

加法运算，与四则运算 `+` 等价。

```python
triton.language.add(x, y, sanitize_overflow: constexpr = True, _builder=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor or Number` | 第一个入参 |
| `y` | `tensor or Number` | 第二个入参 |
| `sanitize_overflow` | `bool` | 是否对整数加法做溢出检查，默认True |

返回值：`tl.tensor`，加法结果。

**DataType 支持**：uint8, int8, uint16, int16, uint32, int32, uint64, int64, fp16, fp32, bf16, bool 均支持。
**Ascend 限制**：不支持 fp64。

```python
output = x + y  # 等价于 output = tl.add(x, y)
```

### tl.sub / `-`

减法运算，四则运算 `-`，无 `tl.sub` 调用方法。

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor or Number` | 第一个入参 |
| `y` | `tensor or Number` | 第二个入参 |

**DataType 支持**：uint8, int8, uint16, int16, uint32, int32, uint64, int64, fp16, fp32, bf16, bool 均支持。
**Ascend 限制**：不支持 fp64。

### tl.mul / `*`

乘法运算，四则运算 `*`，无 `tl.mul` 调用方法。

**DataType 支持**：同 add。
**Ascend 限制**：不支持 fp64。

### `/` (div)

除法运算，四则运算 `/`，无 `tl.div` 方法。底层实现与 `fdiv` 相同，但 `/` 会将非浮点型转换为浮点型再计算。

返回结果类型：总是返回浮点型。`int / int` 会将两个都转成 `float32`。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

### `//` (floordiv)

取整除法，返回向零取整后的除法结果，四则运算 `//`，无 `tl.floordiv` 方法。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, bool。
**Ascend 限制**：不支持 uint 类型, fp16/fp32/bf16。

### `%` (mod)

取模运算，四则运算 `%`。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint 类型, fp64。

---

## 逐元素数学函数

### tl.abs

计算张量中每个元素的绝对值。

```python
triton.language.abs(x, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据 |

返回值：与 `x` 相同 shape 的张量。

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, uint64, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 fp64。

### tl.neg / `-x`

将 tensor 的值取负。

```python
triton.language.neg(x)
```

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, int64, fp16, fp32, bf16。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64, bool。

### tl.ceil

计算张量中每个元素的向上取整值。

```python
triton.language.ceil(x, _semantic=None)
```

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, uint64, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 和整型输入的支持。

### tl.floor

计算 x 的逐元素向下取整。

```python
triton.language.floor(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.sqrt

计算 x 的逐元素快速平方根。

```python
triton.language.sqrt(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.sqrt_rn

计算 x 的逐元素精确平方根（根据 IEEE 标准四舍五入到最近的值）。

```python
triton.language.sqrt_rn(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：比 GPU 多了 fp16、bf16 的支持。

### tl.rsqrt

计算 x 的逐元素平方根倒数。

```python
triton.language.rsqrt(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.exp

计算 x 的逐元素指数（以 e 为底）。

```python
triton.language.exp(x)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.exp2

计算 x 的逐元素指数（以 2 为底）。

```python
triton.language.exp2(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.log

计算 x 的逐元素自然对数。

```python
triton.language.log(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.log2

计算 x 的逐元素对数（以 2 为底）。

```python
triton.language.log2(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.sin

计算 x 的逐元素正弦值。

```python
triton.language.sin(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.cos

计算 x 的逐元素余弦值。

```python
triton.language.cos(x, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、fp32 的支持。

### tl.sigmoid

计算 x 的逐元素 sigmoid 函数值。

```python
triton.language.sigmoid(x)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.erf

计算 x 的逐元素误差函数。

```python
triton.language.erf(x)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

### tl.softmax

计算 x 的逐元素 softmax。

```python
triton.language.softmax(x, dim=None, keep_dims=False, ieee_rounding=False)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据 |
| `dim` | `int` | 指定在哪个维度上计算 softmax |
| `keep_dims` | `bool` | 控制计算后是否保留原维度的形状 |
| `ieee_rounding` | `bool` | 控制浮点数运算是否遵循 IEEE 754 标准的舍入规则 |

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。

```python
ret = tl.softmax(a, dim=2)
```

---

## 融合运算

### tl.fma

计算 x、y 和 z 的逐元素融合乘加运算（`x * y + z`）。

```python
triton.language.fma(x, y, z, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据 |
| `y` | `tensor` | 张量数据 |
| `z` | `tensor` | 张量数据 |

返回值：与 `z` 相同 shape 的张量。

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。

### tl.fdiv

计算 x 和 y 的逐元素快速除法。

```python
triton.language.fdiv(x, y, ieee_rounding=False, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据 |
| `y` | `tensor` | 张量数据 |
| `ieee_rounding` | `bool` | 控制是否遵循 IEEE 754 标准的舍入行为 |

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。

### tl.div_rn

计算 x 和 y 的逐元素精确除法（根据 IEEE 标准四舍五入到最近的值）。

```python
triton.language.div_rn(x, y, _semantic=None)
```

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。但比 GPU 多了 fp16、bf16 的支持。

---

## 取值范围操作

### tl.clamp

将输入张量 x 的值限制在 [min, max] 范围内。

```python
triton.language.clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据 |
| `min` | `tensor` | 下界（可为张量或标量，会广播到 x 的 shape） |
| `max` | `tensor` | 上界（可为张量或标量，会广播到 x 的 shape） |
| `propagate_nan` | `constexpr` | 是否对 min 或 max 做 NaN 的传播 |

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 fp64。

**注意**：当 `propagate_nan=tl.PropagateNAN.NONE` 时，系统会自动添加 NaN 值处理逻辑，可能导致 UB 空间使用增加和性能下降。在 UB 空间紧张的场景下需特别注意。

### tl.maximum

计算 x 和 y 的逐元素最大值。

```python
triton.language.maximum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None)
```

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 fp64。

**注意**：`propagate_nan` 参数有与 `clamp` 相同的 UB 空间限制。

### tl.minimum

计算 x 和 y 的逐元素最小值。

```python
triton.language.minimum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None)
```

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 fp64。

### tl.cdiv

计算 x 除以 div 的向上取整除法。

```python
triton.language.cdiv(x, div)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据，被除数 |
| `div` | `tensor` | 张量数据，除数 |

**DataType 支持 (Ascend)**：int8, int16, int32, int64。
**Ascend 限制**：不支持 uint 类型, bool, 浮点类型。输入范围限制：0~16777216。

### tl.umulhi

计算 x 和 y 的 2N 位乘积中每个元素的最高有效 N 位。

```python
triton.language.umulhi(x, y, _semantic=None)
```

**DataType 支持 (Ascend)**：int32。
**Ascend 限制**：不支持 int64（GPU 支持）。

---

## 使用示例

```python
@triton.jit
def math_kernel(output_ptr, x_ptr,
                XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
                XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XB) + tl.program_id(0) * XB
    yidx = tl.arange(0, YB) + tl.program_id(1) * YB
    zidx = tl.arange(0, ZB) + tl.program_id(2) * ZB
    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    ret = tl.abs(X)       # 绝对值
    ret = tl.exp(ret)     # 指数
    ret = tl.sqrt(ret)    # 平方根
    ret = tl.sigmoid(ret) # sigmoid
    tl.store(output_ptr + idx, ret)
```

## Ascend 通用限制总结

- **fp64 不支持**：所有数学操作在 Ascend 上均不支持 fp64 类型。
- **比 GPU 多支持**：多数数学函数（ceil, floor, sqrt, exp, log, sin, cos, erf, sigmoid, rsqrt, sqrt_rn, div_rn, fdiv）在 Ascend 上额外支持 fp16 和 bf16。
- **Shape 限制**：多数逐元素操作支持 1~5 维 tensor。
- **四则运算**：add, sub, mul 支持无限制维度；div, floordiv, mod 也支持无限制维度。
