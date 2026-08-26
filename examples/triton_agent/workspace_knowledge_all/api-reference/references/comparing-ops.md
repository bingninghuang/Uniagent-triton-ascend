# 比较操作 (Comparing Operations)

所有比较操作均通过运算符重载使用，返回与输入 shape 相同的 bool 类型张量。

## 相等比较 (==)

```python
# 运算符形式
x == y
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

---

## 不等比较 (!=)

```python
# 运算符形式
x != y
```

**DataType 支持 (Ascend)**：同相等比较。

---

## 大于比较 (>)

```python
# 运算符形式
x > y
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64。

---

## 大于等于比较 (>=)

```python
# 运算符形式
x >= y
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

---

## 小于比较 (<)

```python
# 运算符形式
x < y
```

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

---

## 小于等于比较 (<=)

```python
# 运算符形式
x <= y
```

**DataType 支持 (Ascend)**：同小于比较。

---

## 通用参数说明

所有比较操作具有相同的参数结构：

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 左操作数，代表要进行比较的主数据 |
| `other` | `tensor` | 右操作数，与 input 逐元素进行比较 |

返回值：`tl.tensor`，与 input 的 shape 相同的 bool 类型张量。

---

## 使用示例

```python
@triton.jit
def compare_kernel(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr,
                   XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, mask=x_index < N)
        tmp1 = tl.load(in_ptr1 + x_index, mask=x_index < N)
        tmp2 = tmp0 == tmp1  # 可替换为 !=, >, >=, <, <=
        tl.store(out_ptr0 + x_index, tmp2, mask=x_index < N)
```

---

## Ascend 通用限制总结

| 操作 | int8 | int16 | int32 | int64 | uint8 | fp16 | fp32 | bf16 | bool |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| eq (==) | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ |
| ne (!=) | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ |
| gt (>) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ge (>=) | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ |
| lt (<) | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ |
| le (<=) | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ |

- **不支持 fp64**：所有比较操作在 Ascend 上均不支持 fp64。
- **uint 类型**：eq/ne/ge/lt/le 不支持 uint8，gt 额外支持 uint8。所有操作不支持 uint16/uint32/uint64。
- **Shape**：GPU 与 Ascend 平台无差异，均无维度限制。
