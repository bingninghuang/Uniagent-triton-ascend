# 内存/指针操作 (Memory & Pointer Operations)

## tl.load

从全局内存（GlobalMemory）中加载张量数据。

```python
triton.language.load(
    pointer,
    mask=None,
    other=None,
    boundary_check=(),
    padding_option='',
    cache_modifier='',
    eviction_policy='',
    volatile=False,
    _semantic=None
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pointer` | `triton.PointerType` 或 `tensor<PointerType>` 或 `PointerType<tensor>`（来源于 `tl.make_block_ptr`） | 指向 GM 上待读取数据的指针 |
| `mask` | `int1` 或 `tensor<int1>` | 可选，当 mask[i]==False 时不读取该位置数据。仅当 pointer 不来源于 make_block_ptr 时可传入 |
| `other` | `tensor` 或 `scalar` | 可选，当 mask[i]==False 时设置返回值。仅当 mask!=None 时可传入 |
| `boundary_check` | `tuple(int)` | 可选，仅当 pointer 来源于 make_block_ptr 时可传入，指示需要做边界检查的维度 |
| `padding_option` | `""` 或 `"zero"` 或 `"nan"` | 可选，仅当 boundary_check 不为空时可传入，表示越界时填充的值 |
| `cache_modifier` | `""` 或 `"ca"` 或 `"cg"` | 可选，控制 NVIDIA PTX 上的 cache 选项，**对 Ascend 硬件无效** |
| `eviction_policy` | `str` | 控制 NVIDIA PTX 的 eviction 策略，**对 Ascend 硬件无效** |
| `volatile` | `str` | 控制 NVIDIA PTX 的 volatile 选项，**对 Ascend 硬件无效** |

返回值：从 pointer 指向位置加载的 Tensor/Scalar。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。`padding_option` 参数不支持。

### 社区约束

1. 若 pointer 是单指针：返回标量，mask 和 other 必须是标量，不允许传入 boundary_check。
2. 若 pointer 是 N-D tensor：返回与 pointer shape 相同的 tensor，mask 和 other 会广播。
3. 若 pointer 来自 make_block_ptr：mask 和 other 必须为 None，可使用 boundary_check 和 padding_option。

### 使用示例

```python
@triton.jit
def load_kernel(out_ptr, in_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)
```

---

## tl.store

将数据张量存储到全局内存中。

```python
triton.language.store(
    pointer,
    value,
    mask=None,
    boundary_check=(),
    cache_modifier='',
    eviction_policy='',
    _semantic=None
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pointer` | `triton.PointerType` 或 `tensor<PointerType>` 或 `PointerType<tensor>`（来源于 make_block_ptr） | 指向 GM 上待存储地址的指针 |
| `value` | `tensor` 或 `scalar` | 要存储的值，支持隐式广播和隐式类型转换 |
| `mask` | `int1` 或 `tensor<int1>` | 可选，当 mask[i]==False 时不存储。仅当 pointer 不来源于 make_block_ptr 时可传入 |
| `boundary_check` | `tuple(int)` | 可选，仅当 pointer 来源于 make_block_ptr 时可传入 |
| `cache_modifier` | `""` 或 `"ca"` 或 `"cg"` | 可选，**对 Ascend 硬件无效** |
| `eviction_policy` | `str` | 控制 NVIDIA PTX 的 eviction 策略，**对 Ascend 硬件无效** |

返回值：无返回值。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

---

## tl.make_block_ptr

创建指向 GM 上张量的块指针。

```python
triton.language.make_block_ptr(
    base: triton.PointerType,
    shape: List[tensor],
    strides: tuple(int | constexpr),
    offsets: tuple(int | constexpr),
    block_shape: tuple(int | constexpr),
    order: tuple(constexpr),
    _semantic=None
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `base` | `triton.PointerType` | 张量的基指针 |
| `shape` | `tuple(int \| constexpr)` | 张量在 GM 上的形状 |
| `strides` | `tuple(int \| constexpr)` | 张量各维度的步长列表 |
| `offsets` | `tuple(int \| constexpr)` | 张量各维度的基址偏移量列表 |
| `block_shape` | `tuple(constexpr)` | 单次从全局内存加载/存储的块的形状 |
| `order` | `tuple(constexpr)` | 加载/存储的维度顺序 |

返回值：`pointer_type<blocked<shape, element_type>>`，指向 tensor 的指针。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64, bool。

**重要约束**：
- make_block_ptr 的结果不允许进行算术运算，改变偏移量需重新调用 make_block_ptr 或使用 tl.advance。
- Ascend 只允许通过调整 `order` 参数的顺序来表达转置语义，不能通过调整 `stride` 参数的顺序实现转置。

### 使用示例

```python
block_ptr = tl.make_block_ptr(
    base=x_ptr,
    shape=(XB, YB, ZB),
    strides=(YB * ZB, ZB, 1),
    offsets=(0, 0, 0),
    block_shape=(XB, YB, ZB),
    order=(2, 1, 0),
)
X = tl.load(block_ptr)
tl.store(block_ptr_out, X)
```

---

## tl.advance

将 make_block_ptr 的 offset 增加一个偏移量。

```python
triton.language.advance(
    base: triton.PointerType,
    offsets: tuple(int | constexpr),
    _semantic=None
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `base` | `triton.PointerType` | 需要被更新的指针，make_block_ptr 的结果 |
| `offsets` | `tuple(int \| constexpr)` | 各维度的偏移量列表，长度需与 base.offsets 相等 |

返回值：更新后的指针。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16。

```python
block_ptr = tl.make_block_ptr(base=x_ptr, shape=(XB, YB, ZB), strides=(YB*ZB, ZB, 1),
                               offsets=(3, 1, 2), block_shape=(XB, YB, ZB), order=(2, 1, 0))
new_ptr = tl.advance(block_ptr, (-3, -1, -2))
X = tl.load(new_ptr)
```

---

## tl.make_tensor_descriptor

创建张量描述符对象（Triton 3.4.0+）。

```python
triton.language.make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    _semantic=None
) -> tensor_descriptor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `base` | `tensor` | 张量的基指针 |
| `shape` | `List[tensor]` | 张量的形状 |
| `strides` | `List[tensor]` | 张量各维度的步长列表。前面的维度必须是 16 字节的整数倍，最后一维必须是连续存储的 |
| `block_shape` | `List[constexpr]` | 从全局内存加载/存储的块的形状 |

返回值：`tensor_descriptor` 对象（不可直接进行算术运算，需配合 load/store 使用）。

**DataType 支持 (Ascend)**：uint8, int8, int16, int32, int64, fp16, fp32, bf16。
**Ascend 限制**：不支持 uint16/uint32/uint64。

**重要约束**：
- `make_tensor_descriptor` / `load_tensor_descriptor` / `store_tensor_descriptor` 需配套使用，不能与 `tl.load()` / `tl.store()` 混用。
- 不支持 `padding_option` 入参。

### 使用示例

```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(in_out_ptr, shape=[M, N], strides=[N, 1],
                                      block_shape=[M_BLOCK, N_BLOCK])
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))
```

---

## tl.load_tensor_descriptor / desc.load

从张量描述符加载数据块。

```python
# 面向对象方法调用（推荐）
value = desc.load(offsets)

# 函数式接口调用
value = triton.language.load_tensor_descriptor(desc, offsets)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `desc` | `tensor_descriptor_base` | 张量描述符对象，由 make_tensor_descriptor 创建 |
| `offsets` | `Sequence[constexpr \| tensor]` | 数据加载的起始偏移量序列 |

返回值：`tensor`，从指定偏移量处加载的数据块。

---

## tl.store_tensor_descriptor / desc.store

将数据块存储到张量描述符指定内存位置。

```python
# 面向对象方法调用（推荐）
desc.store(offsets, value)

# 函数式接口调用
triton.language.store_tensor_descriptor(desc, offsets, value)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `desc` | `tensor_descriptor_base` | 张量描述符对象 |
| `offsets` | `Sequence[constexpr \| tensor]` | 数据存储的起始偏移量序列 |
| `value` | `tensor` | 待写入的张量数据块 |

返回值：`tensor`，实际写入的数据块。

---

## Ascend 通用限制总结

- **cache_modifier, eviction_policy, volatile**：对 Ascend 硬件无效，910 代际均不支持。
- **DataType 缺失**：uint8/uint16/uint32/uint64, fp64 不支持（硬件限制）。
- **padding_option**：当前不支持，待开发。
- **泛化性问题**：与分支、循环语句搭配使用时，如果 pointer 和 mask 的计算过程涉及较复杂的循环和分支语句，可能出现编译问题。
- **UB 溢出误诊（重要）**：`hivm.hir.vsel` / `root alloc` 报错常被误判为上述 mask+循环问题，但真因往往是 UB（Unified Buffer, 192KB）溢出--编译器在 UB 不够时降级产生 vsel。报错 `requires X bits while Y bits available` 即直接证据。修复方向是减小 tile 或对 OC 维度 tiling，而非改写 mask。详见 hardware-specs"存储层级与容量限制"和 op-design-guide G12。
- **离散 mask**：store 中离散 mask 的处理是将 store 拆解为 atomic {load, select, store}，在 corner case 中存在一定泛化性问题。
