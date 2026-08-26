# 原子操作 (Atomic Operations)

所有原子操作具有相同的参数结构：

```python
triton.language.atomic_xxx(pointer, val/cmp, mask=None, sem=None, scope=None, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pointer` | `triton.PointerDType` | 要操作的内存位置 |
| `val` | `pointer.dtype.element_ty` | 操作的值（右操作数） |
| `mask` | `int1` 或 `tensor<int1>` | 可选，指定数据范围，防止访问越界 |
| `sem` | `str` | 可选，内存语义。Ascend 仅支持 "acq_rel"（默认） |
| `scope` | `str` | 可选，同步范围。Ascend 仅支持 "gpu"（默认） |

返回值：`tensor`，执行操作之前的旧值。

**通用限制**：
- `sem` 参数社区支持 "acquire", "release", "acq_rel", "relaxed"，Ascend 仅支持 "acq_rel"。
- `scope` 参数社区支持 "gpu", "cta", "sys"，Ascend 仅支持 "gpu"。

---

## tl.atomic_add

原子性加法操作，执行 `*pointer + val` 后写回。

```python
triton.language.atomic_add(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, fp16, fp32, bf16。
**Ascend 限制**：不支持 int64, fp64, bool。

```python
@triton.jit
def atomic_add_kernel(in_ptr, out_ptr, old_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)
    xmask = xindex < n_elements
    tmp0 = tl.load(in_ptr + xindex, xmask)
    tmp1 = tl.atomic_add(out_ptr + xindex, tmp0, xmask)
    tl.store(old_ptr + xindex, tmp1, xmask)
```

---

## tl.atomic_and

原子性逻辑与操作，执行 `*pointer & val` 后写回。

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, uint64, int64。
**Ascend 限制**：不支持浮点类型, bool。

---

## tl.atomic_or

原子性逻辑或操作，执行 `*pointer | val` 后写回。

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, uint64, int64。
**Ascend 限制**：不支持浮点类型, bool。

---

## tl.atomic_xor

原子性逻辑异或操作，执行 `*pointer ^ val` 后写回。

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, uint64, int64。
**Ascend 限制**：不支持浮点类型, bool。

---

## tl.atomic_xchg

原子性交换操作，将 `*pointer` 更新为 `val`。

**DataType 支持 (Ascend)**：int8, int16, int32, uint8, uint16, uint32, uint64, int64, fp16, fp32。
**Ascend 限制**：不支持 fp64, bf16, bool。

---

## tl.atomic_max

原子性取最大值操作，执行 `max(*pointer, val)` 后写回。

**DataType 支持 (Ascend)**：int8, int16, int32, fp16, fp32, bf16。
**Ascend 限制**：不支持 int64, uint 类型, fp64, bool。

---

## tl.atomic_min

原子性取最小值操作，执行 `min(*pointer, val)` 后写回。

**DataType 支持 (Ascend)**：同 atomic_max。

---

## tl.atomic_cas

原子性比较和交换操作。若 `*pointer == cmp`，则将 `*pointer` 更新为 `val`，否则不变。

```python
triton.language.atomic_cas(pointer, cmp, val, sem=None, scope=None, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pointer` | `triton.PointerDType` | 要操作的内存位置 |
| `cmp` | `pointer.dtype.element_ty` | 用于与目标内存进行比较的值 |
| `val` | `pointer.dtype.element_ty` | 用于更新的目标值 |

**DataType 支持 (Ascend)**：int16, int32, uint16, uint32, uint64, int64, fp16, fp32。
**Ascend 限制**：不支持 int8, uint8, fp64, bf16, bool。

```python
@triton.jit
def atomic_cas_kernel(in_ptr, cmp_ptr, out_ptr, old_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    xindex = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    xmask = xindex < n_elements
    val = tl.load(in_ptr + xindex, xmask)
    cmp = tl.load(cmp_ptr + xindex, xmask)
    old = tl.atomic_cas(out_ptr + xindex, cmp, val)
    tl.store(old_ptr + xindex, old, xmask)
```

---

## Ascend 通用限制总结

| 操作 | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| atomic_add | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ |
| atomic_and | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × |
| atomic_or | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × |
| atomic_xor | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × |
| atomic_xchg | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| atomic_max | ✓ | ✓ | ✓ | × | × | × | × | × | ✓ | ✓ | ✓ |
| atomic_min | ✓ | ✓ | ✓ | × | × | × | × | × | ✓ | ✓ | ✓ |
| atomic_cas | × | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |

- **sem 参数**：仅支持 "acq_rel"（默认值）。
- **scope 参数**：仅支持 "gpu"（默认值）。
- **fp64**：所有原子操作均不支持。
