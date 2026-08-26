# 索引操作 (Indexing Operations)

## tl.flip

将 tensor 沿某一维度进行翻转。

```python
triton.language.flip(x, dim=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 张量数据 |
| `dim` | `int` | 翻转的维度 |

返回值：`tensor`，输出张量的 shape 与输入 x 的 shape 相同。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64。
**Shape 限制**：仅支持 1~5 维 tensor。

```python
@triton.jit
def flip_kernel(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
                XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XB) + tl.program_id(0) * XB
    yidx = tl.arange(0, YB) + tl.program_id(1) * YB
    zidx = tl.arange(0, ZB) + tl.program_id(2) * ZB
    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    ret = tl.flip(X, 2)
    tl.store(output_ptr + idx, ret)
```

---

## tl.gather

对 src tensor 沿 axis 维度按照 index 执行 gather 操作。

```python
triton.language.gather(src, index, axis, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor` | 被 gather 操作的 tensor |
| `index` | `tensor` | 需要 gather 的索引 |
| `axis` | `int` | 执行 gather 操作的维度 |

返回值：`tensor`，gather 后的结果。

**DataType 支持 (Ascend)**：fp16, fp32, bf16。
**Ascend 限制**：不支持 int 类型、uint 类型、fp64、bool（GPU 仅支持 fp16/fp32/bf16/fp64）。
**Shape 限制**：仅支持 1~5 维 tensor。

```python
@triton.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr,
                  src_dim0: tl.constexpr, src_dim1: tl.constexpr,
                  src_stride0: tl.constexpr, src_stride1: tl.constexpr,
                  idx_dim0: tl.constexpr, idx_dim1: tl.constexpr,
                  idx_stride0: tl.constexpr, idx_stride1: tl.constexpr,
                  out_dim0: tl.constexpr, out_dim1: tl.constexpr,
                  out_stride0: tl.constexpr, out_stride1: tl.constexpr):
    src_offs = tl.arange(0, src_dim0)[:, None] * src_stride0 + tl.arange(0, src_dim1)[None, :] * src_stride1
    src = tl.load(src_ptr + src_offs)
    idx_offs = tl.arange(0, idx_dim0)[:, None] * idx_stride0 + tl.arange(0, idx_dim1)[None, :] * idx_stride1
    idx = tl.load(idx_ptr + idx_offs)
    out = tl.gather(src, idx, axis)
    out_offs = tl.arange(0, out_dim0)[:, None] * out_stride0 + tl.arange(0, out_dim1)[None, :] * out_stride1
    tl.store(out_ptr + out_offs, out)
```

---

## tl.swizzle2d

将一个大小为 size_i x size_j 的行优先矩阵的索引，按每 size_g 行一组，分别转换为列优先矩阵的索引。

```python
triton.language.swizzle2d(i, j, size_i, size_j, size_g)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `i` | `tensor` | index 索引值，最大值为 size(i)-1 |
| `j` | `tensor` | index 索引值，最大值为 size(j)-1 |
| `size_i` | `int` | 索引值 i 的长度 |
| `size_j` | `int` | 索引值 j 的长度 |
| `size_g` | `int` | 分组大小 |

返回值：`out0, out1`，同 i, j shape 的张量。

**DataType 支持 (Ascend)**：int32, int64。
**Shape 限制**：仅支持 2 维 tensor。

```python
@triton.jit
def swizzle_kernel(out0, out1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    i = tl.arange(0, XB)[:, None]
    j = tl.arange(0, YB)[None, :]
    ij = i * YB + j
    xx, yy = tl.swizzle2d(i, j, size_i=XB, size_j=YB, size_g=ZB)
    ptr = tl.load(out0)
    xx = tl.cast(xx, dtype=ptr.dtype)
    yy = tl.cast(yy, dtype=ptr.dtype)
    tl.store(out0 + ij, xx)
    tl.store(out1 + ij, yy)
```

---

## tl.where

根据条件进行判断返回张量 x 还是 y 的值。条件为真时返回 x 的值，否则返回 y 的值。

```python
triton.language.where(condition, x, y, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `condition` | `tensor(bool)` | 条件张量 |
| `x` | `tensor` | 条件为真时返回的值 |
| `y` | `tensor` | 条件为假时返回的值 |

返回值：`tensor`，输出张量的 shape 与输入 x 的 shape 相同。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint16/uint32/uint64, fp64。
**Shape 限制**：仅支持 1~5 维 tensor。

```python
@triton.jit
def where_kernel(output_ptr, x_ptr, y_ptr,
                 XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
                 XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xidx = tl.arange(0, XB) + tl.program_id(0) * XB
    yidx = tl.arange(0, YB) + tl.program_id(1) * YB
    zidx = tl.arange(0, ZB) + tl.program_id(2) * ZB
    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]
    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)
    tmp2 = X < Y
    ret = tl.where(tmp2, X, 1)
    tl.store(output_ptr + idx, ret)
```

---

## Ascend 通用限制总结

- **gather**：Ascend 仅支持 fp16/fp32/bf16，不支持 int 类型（GPU 也仅支持浮点类型）。
- **swizzle2d**：仅支持 int32/int64，仅支持 2D tensor。
- **flip/where**：不支持 uint16/uint32/uint64, fp64。
- **Shape 限制**：所有操作仅支持 1~5 维 tensor（swizzle2d 仅支持 2D）。
