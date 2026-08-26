# 随机数生成 (Random Number Generation)

所有随机数生成操作基于 Philox 伪随机数生成算法。

## tl.rand

给定 seed 标量和 offset 块，返回一个在 U(0,1) 中的 float32 类型的随机块。

```python
triton.language.rand(seed, offset, n_rounds: constexpr = 10)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `seed` | `int` 或 `tensor` | 用于生成随机数的种子 |
| `offset` | `int` 或 `tensor` | 用于生成随机数的偏移量 |
| `n_rounds` | `constexpr` (默认10) | Philox 算法的迭代轮数 |

返回值：`tensor`，float32 类型的随机块，shape 与 offset 相同，值在 [0.0, 1.0) 区间内均匀分布。

**seed 类型支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, uint64, int64, bool。

```python
@triton.jit
def kernel_rand(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.rand(5, 10 + global_offset, n_rounds)
        tl.store(x_ptr + global_offset, rand_vals)
```

---

## tl.randint

给定 seed 标量和 offset 块，返回一个 int32 类型的随机块。

```python
triton.language.randint(seed, offset, n_rounds: constexpr = 10)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `seed` | `int` 或 `tensor` | 用于生成随机数的种子 |
| `offset` | `int` 或 `tensor` | 用于生成随机数的偏移量 |
| `n_rounds` | `constexpr` (默认10) | Philox 算法的迭代轮数 |

返回值：`tensor`，int32 类型的随机块，shape 与 offset 相同。

**seed 类型支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, uint64, int64, bool。

**提示**：如果需要多个随机数流，使用 `randint4x` 可能比连续调用4次 `randint` 更快。

```python
@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds)
        tl.store(x_ptr + global_offset, rand_vals)
```

---

## tl.randn

给定 seed 标量和 offset 块，返回一个服从标准正态分布 N(0,1) 的 float32 类型的随机块。

```python
triton.language.randn(seed, offset, n_rounds: constexpr = 10)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `seed` | `int` 或 `tensor` | 用于生成随机数的种子 |
| `offset` | `int` 或 `tensor` | 用于生成随机数的偏移量 |
| `n_rounds` | `constexpr` (默认10) | Philox 算法的迭代轮数 |

返回值：`tensor`，float32 类型的随机块，shape 与 offset 相同，值服从标准正态分布 N(0, 1)。

**seed 类型支持 (Ascend)**：同 `tl.rand`。

```python
@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    offsets = block_offset + tl.arange(0, XBLOCK)
    mask = offsets < N
    rand_vals = tl.randn(5, 10 + offsets, n_rounds)  # 一次生成一整块随机数
    tl.store(x_ptr + offsets, rand_vals, mask=mask)
```

---

## tl.randint4x

给定 seed 标量和 offset 块，返回4个 int32 类型的随机块。是 Philox 伪随机数生成器的最高效入口点。

```python
triton.language.randint4x(seed, offset, n_rounds: constexpr = 10)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `seed` | `int` 或 `tensor` | 用于生成随机数的种子 |
| `offset` | `int` 或 `tensor` | 用于生成随机数的偏移量 |
| `n_rounds` | `constexpr` (默认10) | Philox 算法的迭代轮数 |

返回值：4 个 int32 类型的随机块，每个块的 shape 与 offset 相同。

**seed 类型支持 (Ascend)**：同 `tl.rand`。

```python
# offset 为标量时
@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds)
        mask = (global_offset + indices) < N
        tl.store(x_ptr + global_offset + indices, rand_vals, mask)

# offset 为张量时（存储 tensor 大小是 offset 的 4 倍）
@triton.jit
def triton_randint4x1d(out_ptr, seed, L: tl.constexpr):
    idx = tl.arange(0, L)
    rnd0, rnd1, rnd2, rnd3 = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd0)
    tl.store(out_ptr + L + idx, rnd1)
    tl.store(out_ptr + 2 * L + idx, rnd2)
    tl.store(out_ptr + 3 * L + idx, rnd3)
```

---

## Ascend 通用限制总结

| 操作 | 返回类型 | 分布 |
|------|---------|------|
| rand | float32 | U(0, 1) 均匀分布 |
| randint | int32 | 均匀分布 |
| randn | float32 | N(0, 1) 标准正态分布 |
| randint4x | 4x int32 | 均匀分布 |

- **seed 类型**：支持 int8, int16, int32, uint8, uint16, uint32, uint64, int64, bool。不支持 fp16/fp32/bf16/fp64。
- **n_rounds**：默认为 10，Philox 算法的迭代轮数。
- **randint4x**：一次生成4个随机块，效率最高。当 offset 为张量时，存储 tensor 大小是 offset 的 4 倍。
