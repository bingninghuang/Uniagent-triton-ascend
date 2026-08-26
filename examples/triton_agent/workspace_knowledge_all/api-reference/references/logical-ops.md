# 逻辑操作 (Logical Operations)

## 按位与 (&)

计算两个元素的与值。

```python
# 运算符形式
x & y

# dunder 方法
x.__and__(y)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据 |
| `y` | `tensor` | 张量数据 |

返回值：`tensor`，同 x、y 的 shape。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, bool。
**Shape 限制**：仅支持 1~5 维 tensor。

---

## 按位或 (|)

计算两个元素的或运算。

```python
# 运算符形式
x | y

# dunder 方法
x.__or__(y)
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, bool。

---

## 按位异或 (^)

计算两个元素的异或值。

```python
# 运算符形式
x ^ y

# dunder 方法
x.__xor__(y)
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, bool。

---

## 逻辑非 (not)

对 tensor 做逐元素逻辑非（0 变 1，非零变 0）。与按位取反 `~X` 不同：逻辑非将非零值变为0，零变为1。

```python
# 通过 not 关键字（Triton AST 拦截处理）
not x

# dunder 方法
x.__not__()
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, bool。
**特殊说明**：Ascend 相比 GPU 额外支持非 bool 类型（GPU 仅支持 bool）。

---

## 按位取反 (~)

将 tensor 每个值按比特位进行翻转。

```python
# 运算符形式
~x

# dunder 方法
x.__invert__()
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, bool。

---

## logical_and

用于对两个张量进行逐元素逻辑与运算。

```python
x.logical_and(y)
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, uint16, uint32, uint64, fp16, fp32, bf16, bool。
**特殊说明**：Ascend 相比 GPU 额外增加了对整型、浮点型（除 fp64, fp8）的支持。GPU 仅支持 bool。

---

## logical_or

用于对两个张量进行逐元素逻辑或运算。

```python
x.logical_or(y)
```

**DataType 支持 (Ascend)**：同 logical_and。

---

## 左移位 (<<)

根据给定值将 tensor 张量进行左移位。

```python
# 运算符形式
x << y
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 左操作数，要进行移位的主数据 |
| `other` | `tensor` 或 `scalar` | 右操作数，进行移位的数值 |

**DataType 支持 (Ascend)**：int8, int16, int32, int64, bool。
**Ascend 限制**：不支持 uint 类型。右操作数 `other` 仅支持标量，不支持 tensor（即 `x << 2` 合法，`x << y`（y 为 tensor）暂不支持）。

```python
@triton.jit
def lshift_kernel(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    x0 = tl.load(in_ptr0 + idx)
    ret = x0 << 2  # 左移2位
    tl.store(out_ptr0 + odx, ret)
```

---

## 右移位 (>>)

根据给定值将 tensor 张量进行右移位。

```python
# 运算符形式
x >> y
```

**DataType 支持 (Ascend)**：同左移位。
**Ascend 限制**：同左移位。右操作数 `other` 仅支持标量。

---

## 取负 (-)

将 tensor 的值取负。

```python
# 运算符形式
-x

# dunder 方法
x.__neg__()
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64, bool。

---

## Ascend 通用限制总结

| 操作 | int8 | int16 | int32 | int64 | uint8 | fp16 | fp32 | bf16 | bool |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| and (&) | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ |
| or (\|) | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ |
| xor (^) | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ |
| not | ✓ | ✓ | ✓ | ✓ | × | × | × | × | ✓ |
| invert (~) | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ |
| logical_and | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| logical_or | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| lshift (<<) | ✓ | ✓ | ✓ | ✓ | × | × | × | × | ✓ |
| rshift (>>) | ✓ | ✓ | ✓ | ✓ | × | × | × | × | ✓ |
| neg (-) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |

- **logical_and/logical_or**：Ascend 相比 GPU 额外支持整型和浮点型（GPU 仅支持 bool）。
- **lshift/rshift**：右操作数仅支持标量，不支持 tensor。
- **Shape 限制**：所有操作仅支持 1~5 维 tensor。
