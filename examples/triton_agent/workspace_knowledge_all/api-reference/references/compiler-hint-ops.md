# 编译器提示操作 (Compiler Hint Operations)

编译器提示操作用于向编译器提供额外的信息，帮助生成更高效的代码。这些操作不会改变计算语义。

## tl.assume

向编译器提供条件假设信息，允许编译器基于已知为真的条件进行优化。不会在运行时检查条件。

```python
triton.language.assume(cond, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `cond` | `bool` | 编译器可以假设为真的条件表达式 |

返回值：无返回值。

**DataType 支持**：仅 bool。

```python
@triton.jit
def basic_assume_example(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # 假设 BLOCK_SIZE 是2的幂次，编译器可以优化除法为移位
    tl.assume((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    result = x // BLOCK_SIZE + y % BLOCK_SIZE
    tl.store(y_ptr + offsets, result)
```

---

## tl.debug_barrier

插入屏障指令，用于在调试时同步块中的所有线程。在同一块中的所有其他线程也到达该点之前，任何线程都不会继续执行。

```python
triton.language.debug_barrier(_semantic=None)
```

返回值：无返回值。

**注意**：主要用于调试，不应在性能关键的生产代码中使用。

```python
@triton.jit
def debug_barrier_basic(A, B, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + offsets)
    # 确保所有线程都完成了数据加载
    tl.debug_barrier()
    b = a * 2
    # 确保所有线程都完成了计算
    tl.debug_barrier()
    tl.store(C + offsets, b)
```

---

## tl.max_constancy

向编译器声明输入张量中值的常量性模式，告知编译器输入数据中每组连续的值都是相等的。

```python
triton.language.max_constancy(input, values, _builder=None, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量，其值具有特定的常量性模式 |
| `values` | `constexpr[int]` 或 `list[constexpr[int]]` | 描述每个维度的恒定性特征，维度需与 input 相同 |

返回值：`tensor`，与输入相同的张量（编译器提示，不改变值）。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

**注意**：`values` 的维度要与 `input` 的维度相同。当 shape 的最后一维为1时注意降维情况。如二维 input 对应通用 values 入参为 [1, 1]。

```python
@triton.jit
def basic_constancy_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)
    # 声明每4个连续的值都是相等的
    input_data = tl.max_constancy(input_data, [4])
    result = input_data * 2
    tl.store(B + offsets, result)
```

---

## tl.max_contiguous

向编译器声明输入张量中的连续性模式，告知编译器输入张量的前 value 个数是连续的。

```python
triton.language.max_contiguous(input, values, _builder=None, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `values` | `constexpr[int]` 或 `list[constexpr[int]]` | 描述每个维度的连续性特征，维度需与 input 相同 |

返回值：`tensor`，与输入相同的张量（编译器提示，不改变值）。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

**注意**：`values` 的维度要与 `input` 的维度相同。

```python
@triton.jit
def triton_max_contiguous(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    val = tl.load(A + offsets)
    # 声明 offset 里的前 BLOCK_SIZE 个数是连续的
    input_data = tl.max_contiguous(val, [BLOCK_SIZE])
    result = input_data * 2
    tl.store(B + offsets, result)
```

---

## tl.multiple_of

向编译器声明输入张量中的第一个值是某个数的倍数。

```python
triton.language.multiple_of(input, values, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `values` | `constexpr[int]` 或 `list[constexpr[int]]` | 声明输入值是这些数的倍数，维度需与 input 相同 |

返回值：`tensor`，与输入相同的张量（编译器提示，不改变值）。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

**注意**：`values` 描述每个维度第一个值的可除性特征，维度要与 `input` 相同。

```python
@triton.jit
def basic_multiple_of_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)
    # 声明输入张量的第一个值是 BLOCK_SIZE 的倍数
    input_data = tl.multiple_of(input_data, [BLOCK_SIZE])
```

---

## Ascend 通用限制总结

- **assume/debug_barrier**：不涉及 DataType 限制。
- **max_constancy/max_contiguous/multiple_of**：不支持 uint8/uint16/uint32/uint64, fp64（硬件限制）。
- **values 维度**：所有提示操作的 values 参数维度必须与 input 维度相同，注意最后一维为1时的降维情况。
