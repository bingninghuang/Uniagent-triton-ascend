# al.* 地址空间与核控制接口

## 1. al.ascend_address_space -- 地址空间枚举

### 背景

为支持 Ascend 级编程的需要，提供用户手动创建指定地址空间上的内存（buffer），对接 `hivm::AddressSpace` 枚举。

### 接口说明

```python
al.ascend_address_space.UB    # Unified Buffer
al.ascend_address_space.L1   # L1 Cache Buffer (cbuf)
al.ascend_address_space.L0A  # L0A (ca)
al.ascend_address_space.L0B  # L0B (cb)
al.ascend_address_space.L0C  # L0C (cc)
```

无返回值，无入参。需要配合 `bl.alloc` 使用。

### 约束

- 需要配合 `bl.alloc` 使用

### 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

@triton.jit
def allocate_local_buffer(XBLOCK: tl.constexpr):
    # 默认地址空间（不携带地址空间信息）
    bl.alloc(tl.float32, [XBLOCK])
    # 指定 UB 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)
    # 指定 L1 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L1)
    # 指定 L0A 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0A)
    # 指定 L0B 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0B)
    # 指定 L0C 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0C)
    # 独占内存
    bl.alloc(
        tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB, is_mem_unique=True
    )
```

---

## 2. al.scope -- Cube/Vector 核分离作用域

### 硬件背景

昇腾处理器包含多种类型的计算单元（例如用于矩阵运算的 Cube Unit 和用于向量/标量运算的 Vector Unit）。`al.scope` 允许内核开发者显式地告诉 Triton 编译器，特定代码区域应该目标哪种硬件单元，从而实现更精细的性能调优和资源利用。

### 接口说明

```python
with al.scope(core_mode: str, disable_auto_sync: bool = False):
    # 此代码块内的 Triton 语句将根据指定的 core_mode 进行编译和执行
    ...
```

`al.scope` 是上下文管理器（Context Manager），专为 Triton 内核中的代码块指定昇腾硬件的执行模式。

### 参数

| 参数名 | 类型 | 必需 | 说明 | 可选值 |
|--------|------|------|------|--------|
| `core_mode` | `str` | 是 | 指定该作用域内代码将要使用的昇腾核心类型 | `"vector"`, `"cube"`, `"SIMT"`, `"SIMD"` |
| `disable_auto_sync` | `bool` | 否 | 禁用自动同步，默认 `False` | `True` / `False` |

### core_mode 常用值

| 值 | 目标核心 | 用途 |
|----|----------|------|
| `"vector"` | Vector Unit（向量核心） | 元素级操作，如加法、乘法、激活函数、数据加载和存储 |
| `"cube"` | Cube Unit（矩阵核心） | 矩阵计算，特别是矩阵乘法（GEMM）和卷积操作，通常与 `tl.dot` 关联 |
| `"SIMT"` | Single instruction multiple thread | - |
| `"SIMD"` | Single instruction multiple data | - |

### 约束说明

- **并行执行**：cube 和 vector 作用域内的操作并行执行
- **单作用域**：每个 kernel 支持一个 cube 作用域和一个 vector 作用域
- **显式同步**：跨作用域的数据依赖需要使用同步操作（如 `sync_block_set` / `sync_block_wait`）

### 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def kernel_scope_vector(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)

@triton.jit
def kernel_scope_cube(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="cube"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)

# 作用域逃逸：scope 内定义的变量在 scope 外仍可使用
@triton.jit
def kernel_scope_escape(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
    # x 在 scope 外仍可使用
    a = x + 1.0
    tl.store(out_ptr + i, a, mask=i < n)

# 嵌套作用域
@triton.jit
def kernel_nested_scope(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        with al.scope(core_mode="vector"):
            with al.scope(core_mode="cube"):
                x = tl.load(x_ptr + i, mask=i < n)
                y = tl.load(y_ptr + i, mask=i < n)
                result = x + y
                tl.store(out_ptr + i, result, mask=i < n)

# 禁用自动同步
@triton.jit
def kernel_scope_disable_auto_sync(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)
```

## 3. al.sub_vec_id -- Vector 子核 ID

### 硬件背景

昇腾硬件 AIC 与 AIV 核数配比不同（1:N），Triton 编程抽象屏蔽了 Cube 核与 Vector 核的硬件细节，由编译器通过 AutoSubTiling Pass 自动实现数据切分。`sub_vec_id` 编程接口返回 N 个 Vector 核的 sub id，允许算子开发者根据 vector 核 sub id 决定每个核处理哪些数据。

### 接口说明

```python
def sub_vec_id() -> i16
```

- **返回值**：返回范围为 `[0, N)` 的 Sub Vector ID
- **入参**：无

### 约束说明

仅在 AIC 和 AIV 核混合使用场景中有效，不可在纯 Cube 类算子或者纯 Vector 类算子中使用，否则会触发编译报错。

### 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def verify_sub_vec_id_kernel(out_ptr, N: tl.constexpr):
    with al.scope(core_mode="vector"):
        sub_id = al.sub_vec_id()
        offs = sub_id * N + tl.arange(0, N)
        out_ptrs = out_ptr + offs
        tl.store(out_ptrs, sub_id.to(tl.int32))
```

编译后会生成 `hivm.hir.get_sub_block_idx` 指令，并且模块属性会包含 `hivm.disable_auto_tile_and_bind_subblock`。

---

## 4. al.subview -- Buffer 子视图

### 硬件背景

昇腾硬件 A5 支持了定义新视图，仅通过偏移、大小和步幅实现，不复制底层数据。

### 接口说明

**接口一（函数调用）：**

```python
def subview(
    src: bl.buffer,
    offsets: List[tl.tensor],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    builder: ir.builder
) -> bl.buffer
```

**接口二（方法调用）：**

```python
def subview(
    self,
    offsets: List[tl.tensor],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _builder=None
) -> bl.buffer
```

返回值：`bl.buffer`

### 入参说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `src` | `bl.buffer` | 源 buffer |
| `offsets` | `List[tl.tensor]` | 偏移（支持 tensor 或 constexpr 传入） |
| `sizes` | `List[tl.constexpr]` | 输出的 size |
| `strides` | `List[tl.constexpr]` | 步长 |

### 约束说明

- size、offset、stride 必须大于 0（offset 可以为 0），不能为负值
- size 的每一个维度的大小不能大于原 buffer 的大小
- 子视图的每一个维度的大小不能超过原 buffer 的大小
- stride 的访问不能超过 src 的大小，stride 所有元素全为 1
- 参数的设置要指明每一个维度的值，参数维度应该和输入 buffer 的维度保持一致
- offset 必须 32 字节对齐
- 子视图中最后一个维度的第二行第一个点的偏移必须是 32 字节对齐
- `sizes`、`strides` 传入类型：`List[tl.constexpr]`（不要误传 tensor，否则会报错类型不匹配）；`offsets` 补充支持了 tensor 传入（也可以传入 constexpr）

### 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

@triton.jit
def test_subview_kernel1(XBLOCK: tl.constexpr):
    # 分配一个本地 buffer
    src_buffer = bl.alloc(tl.float32, [XBLOCK, XBLOCK])
    result_buffer = bl.subview(
        src_buffer,
        offsets=[1, 0],
        sizes=[XBLOCK - 2, XBLOCK],
        strides=[1, 1],
    )

@triton.jit
def test_subview_kernel2(
    XBLOCK: tl.constexpr,
    offset: tl.constexpr,
    size: tl.constexpr,
    stride: tl.constexpr
):
    src_buffer = bl.alloc(tl.float32, [XBLOCK, XBLOCK])
    bl.subview(
        src_buffer,
        offsets=[offset, 0],
        sizes=[size, XBLOCK],
        strides=[stride, 1],
    )
```
