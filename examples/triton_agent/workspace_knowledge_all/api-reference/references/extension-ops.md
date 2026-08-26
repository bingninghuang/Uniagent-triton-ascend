# 扩展操作 (Extension Operations)

## tl.compile_hint

编译器提示机制，允许用户为张量附加元数据信息，传递到编译器后端用于指导优化。

```python
triton.language.compile_hint(ptr, hint_name, hint_val=None, _builder=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | `tensor` | 需要附加提示的张量对象 |
| `hint_name` | `str` (constexpr) | 提示的名称标识符 |
| `hint_val` | `None`/`bool`/`int`/`constexpr`/`list` | 提示的值，支持多种类型 |

返回值：无返回值（编译器提示，不改变计算语义）。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**限制**：hint_name 必须为字符串；list 参数仅支持整数数组；同一张量可多次标注。

```python
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.compile_hint(tmp0, "hint_a")
        tl.multibuffer(tmp0, 2)
        tmp2 = tmp0
        tl.compile_hint(tmp2, "hint_b", 42)
        tl.compile_hint(tmp2, "hint_c", True)
        tl.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```

---

## tl.extract_slice

从输入张量中按照指定的偏移量、大小和步幅参数提取一个张量切片。

```python
triton.language.extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `ful` | `tensor` | 要提取切片的源张量 |
| `offsets` | `tuple of ints` | 切片在各维度的起始偏移量 |
| `sizes` | `tuple of ints` | 切片在各维度的大小 |
| `strides` | `tuple of ints` | 切片在各维度的步长 |

返回值：`tensor`，提取的切片张量。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, uint16, uint32, uint64, fp16, fp32, bf16。

```python
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    out_sub = tl.extract_slice(output, [block_start], [32], [1])
    tl.store(output_ptr + block_start + tl.arange(0, 32), out_sub)
```

---

## tl.insert_slice

将一个子张量插入到另一个张量的指定位置。

```python
triton.language.insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `ful` | `tensor` | 接收插入的目标张量 |
| `sub` | `tensor` | 要插入的子张量，形状须与 sizes 匹配 |
| `offsets` | `tuple of ints` | 插入的起始偏移量 |
| `sizes` | `tuple of ints` | 插入区域的大小 |
| `strides` | `tuple of ints` | 插入区域的步长 |

返回值：`tensor`，插入子张量后的新张量。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, uint16, uint32, uint64, fp16, fp32, bf16。

```python
@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 提取切片
    x_sub = tl.extract_slice(x, [offset], [size], [1])
    y_sub = tl.extract_slice(y, [offset], [size], [1])
    output_sub = x_sub + y_sub
    # 将计算结果插入回原张量
    output = tl.load(output_ptr + offsets, mask=mask)
    output = tl.insert_slice(output, output_sub, [offset], [size], [1])
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## tl.get_element

根据给定的索引，从输入张量中读取单个元素。

```python
triton.language.get_element(src, indice, _builder=None, _generator=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor` | 要被访问的源张量 |
| `indice` | `tuple of ints` 或 `tuple of tensors` | 用于指定元素位置的索引 |

返回值：`scalar`，与 src 张量元素类型相同的标量值。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, uint16, uint32, uint64, fp16, fp32, bf16。
**约束**：`indice` 的长度必须与 `src` 张量的维度数相同。

```python
@triton.jit
def index_select_kernel(in_ptr, indices_ptr, out_ptr, dim,
                         g_stride: tl.constexpr, indice_length: tl.constexpr,
                         g_block: tl.constexpr, g_block_sub: tl.constexpr,
                         other_block: tl.constexpr):
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)
        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            for i in range(0, g_block_sub):
                gather_offset = tl.get_element(indices, (i,)) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx)
                tmp_buf = tl.insert_slice(tmp_buf, val[None, :],
                                          offsets=(i, 0), sizes=(1, other_block), strides=(1, 1))
            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], tmp_buf)
```

---

## tl.multibuffer

为张量设置多缓冲，允许编译器对同一张量创建多个副本。

```python
triton.language.multibuffer(src, size, _builder=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor` | 需要进行多缓冲设置的源张量 |
| `size` | `int` 或 `constexpr` | 要创建的缓冲区副本数量 |

返回值：无返回值（编译器提示）。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, uint16, uint32, uint64, fp16, fp32, bf16, bool。
**Ascend 限制**：当前实现仅支持 `size` 为 2。

```python
@triton.jit
def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + xindex, xmask)
        tl.multibuffer(tmp0, 2)  # 设置双缓冲
        tl.store(out_ptr0 + xindex, tmp0, xmask)
```

---

## tl.parallel

专门用于多核心并行执行的迭代器，继承自 `range`，提供显式的多核心并行语义。

```python
triton.language.parallel(arg1, arg2=None, step=None, num_stages=None,
                         loop_unroll_factor=None, bind_sub_block: bool = False,
                         _semantic=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `arg1` | `int`/`constexpr` | 必需 | 起始值（单参数时为结束值，从0开始） |
| `arg2` | `int`/`constexpr` | - | 结束值（不包含） |
| `step` | `int`/`constexpr` | `1` | 每次迭代的步长 |
| `num_stages` | `int` | - | 流水线阶段数 |
| `loop_unroll_factor` | `int` | - | 循环展开因子 |
| `bind_sub_block` | `bool` | `False` | 绑定到子块，启用多核心并行执行 |

**DataType 支持 (Ascend)**：int8, int16, int32, int64。
**注意**：`parallel` 相比 `range` 移除了 `disallow_acc_multi_buffer`、`flatten`、`warp_specialize`、`disable_licm` 参数。`bind_sub_block` 功能是否完全实现待验证。

```python
@triton.jit
def parallel_kernel(input_ptr, output_ptr0, output_ptr1, n_elements: tl.constexpr):
    x = tl.arange(0, n_elements)
    x0 = x // 4
    x1 = x % 4
    a_ptr = input_ptr + x0
    b_ptr = input_ptr + x0
    for i in tl.parallel(0, 3, 1, bind_sub_block=False):
        a_ptr += x0
        b_ptr += x0
    a_ptr += x1
    b_ptr += x1
    val = tl.load(a_ptr + 0)
    tl.store(output_ptr0 + x, val)
    val = tl.load(b_ptr)
    tl.store(output_ptr1 + x, val)
```

---

## sync_block 系列操作

显式的核心间同步指令，用于协调 Cube-Vector 架构中不同核心间的执行顺序和数据一致性。

### tl.sync_block_set

生产者核心完成任务后，向消费者发送同步信号。

```python
triton.language.sync_block_set(sender, receiver, event_id, _builder=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `sender` | `str` | 发送方核心类型："cube" 或 "vector" |
| `receiver` | `str` | 接收方核心类型："cube" 或 "vector" |
| `event_id` | `int` | 事件ID，用于区分不同的同步点（0-15） |

**限制**：sender 和 receiver 不能相同；event_id 必须在 0-15 范围内。

### tl.sync_block_wait

消费者核心等待生产者的同步信号。

```python
triton.language.sync_block_wait(sender, receiver, event_id, _builder=None)
```

参数同 `sync_block_set`。event_id 必须与对应 `sync_block_set` 使用的 ID 一致。

### tl.sync_block_all

全局屏障同步，让所有指定类型的核心同步到同一点。

```python
triton.language.sync_block_all(mode, event_id, _builder=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `mode` | `str` | 同步模式："all_cube"、"all_vector" 或 "all" |
| `event_id` | `int` | 全局同步事件ID（0-15） |

### 使用示例

```python
import triton.language as tl
import triton.language.ascend as al

@triton.jit
def sync_example():
    # Cube 核心计算并通知 Vector
    with al.Scope(core_mode="cube"):
        # ... 执行 Cube 计算 ...
        tl.sync_block_set("cube", "vector", 0)

    # Vector 核心等待 Cube 完成
    with al.Scope(core_mode="vector"):
        tl.sync_block_wait("cube", "vector", 0)
        # ... 执行 Vector 计算 ...

@triton.jit
def flash_attention_fwd(q_ptr, k_ptr, v_ptr, o_ptr, ...):
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    with al.Scope(core_mode="cube"):
        for start_n in range(0, N, BLOCK_N):
            qk = tl.dot(q, k)
            tl.sync_block_set("cube", "vector", 0)
            tl.sync_block_wait("vector", "cube", 1)
            pv = tl.dot(p, v)
            tl.sync_block_set("cube", "vector", 2)

    with al.Scope(core_mode="vector"):
        for start_n in range(0, N, BLOCK_N):
            tl.sync_block_wait("cube", "vector", 0)
            m_new, l_new, softmax_out = _softmax(qk, m_prev, l_prev)
            tl.sync_block_set("vector", "cube", 1)
            tl.sync_block_wait("cube", "vector", 2)
            acc = _update_output(pv, softmax_out, acc)

    with al.Scope(core_mode="cube"):
        tl.sync_block_all("all", 0)
    tl.store(o_ptr + offsets, acc)
```

---

## tl.index_select_simd (Ascend 专用)

在非尾轴维度上并行 gather 多个索引，以 tile 为单位将数据零拷贝地从全局内存（GM）直接搬运到统一缓冲区（UB）。等效于 `torch.index_select` 的高性能实现。

```python
triton.language.extra.ascend.libdevice.index_select_simd(
    src, dim, index, src_shape, src_offset, read_shape
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor`/`pointer` | 源张量指针，位于 GM 上 |
| `dim` | `int` | 执行 index_select 的维度，取值 [0, len(src_shape)-2]，不支持尾轴 |
| `index` | `tensor` | 1D 索引数组，位于 UB 上 |
| `src_shape` | `Tuple[int]` | 源张量的完整形状 |
| `src_offset` | `Tuple[int]` | 读取起始位置，dim 维度可设为 -1 |
| `read_shape` | `Tuple[int]` | 读取数据大小，dim 维度必须设为 -1 |

返回值：`tensor`（位于 UB 上），形状与 read_shape 一致。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, uint8, fp16, fp32, bf16, bool。
**限制**：
- dim 不能是尾轴（最后一个维度）。
- index 必须是 1D 张量，数据类型为 int32 或 int64。
- 不检查索引越界，用户需自行保证。
- GPU 平台不支持此操作（Ascend 专用 intrinsic）。

```python
import triton.language.extra.ascend.libdevice as libdevice

@triton.jit
def embedding_kernel(embed_ptr, indices_ptr, output_ptr,
                     vocab_size: tl.constexpr, embed_dim: tl.constexpr):
    pid = tl.program_id(0)
    indices = tl.load(indices_ptr + pid * 16 + tl.arange(0, 16))
    embeddings = libdevice.index_select_simd(
        src=embed_ptr, dim=0, index=indices,
        src_shape=(vocab_size, embed_dim),
        src_offset=(-1, 0), read_shape=(-1, embed_dim)
    )
    offsets = tl.arange(0, 16)[:, None] * embed_dim + tl.arange(0, embed_dim)[None, :]
    tl.store(output_ptr + pid * 16 * embed_dim + offsets, embeddings)
```

---

## Ascend 通用限制总结

- **multibuffer**：仅支持 size=2。
- **parallel**：移除了 flatten/warp_specialize 等参数；bind_sub_block 功能待验证。
- **sync_block**：sender/receiver 不能相同；event_id 范围 0-15；需配合 `al.Scope` 使用。
- **index_select_simd**：Ascend 专用，不支持尾轴，index 必须 1D。
- **extract_slice/insert_slice/get_element**：支持所有基本数据类型（不含 bool）。
