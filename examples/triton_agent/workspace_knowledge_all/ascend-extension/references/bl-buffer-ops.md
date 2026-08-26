# bl.* Buffer 级操作接口

## 1. bl.alloc -- Buffer 分配

### 背景

为支持 Ascend 级编程的需要，提供用户手动创建指定地址空间上的内存（buffer）。本接口是硬件无关的接口，对接 `memref.alloc`。

### 接口说明

```python
def alloc(
    etype: tl.dtype,
    shape: List[tl.constexpr],
    _address_space: address_space = None,
    is_mem_unique: bool = False,
    _builder=None
) -> buffer
```

### 返回值

返回一个 buffer language 下的 buffer 类型，与 triton language 下的 tensor 做语义上的隔离，不支持相互赋值，需要 `to_tensor` 和 `to_buffer` 来显式转换。表示一段分配在指定地址空间的内存，携带数据类型、形状和地址空间三部分信息。

### 入参

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `etype` | `tl.dtype` | 是 | 数据类型 / element type |
| `shape` | `List[tl.constexpr]` | 是 | buffer 的形状 |
| `_address_space` | `bl.address_space` | 否 | buffer 所在的地址空间，默认为空（不携带地址空间信息） |
| `is_mem_unique` | `bool` | 否 | 是否独占内存，生成的 annotation.mark 在 plan memory 时会用到，默认为 `False` |

### 昇腾平台数据类型支持

| 类型 | int8 | int16 | int32 | uint64 | int64 | fp16 | fp32 | bf16 |
|------|------|-------|-------|--------|-------|------|------|------|
| Ascend | 支持 | 支持 | 支持 | 支持 | 支持 | - | 支持 | 支持 |

### 约束说明

- dtype 不支持 `tl.void`
- shape 每个元素必须是正整数
- 需自行保证符合指定的地址空间上的大小限制
- `_address_space` 参数默认为空，表示不携带任何地址空间信息

### 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

@triton.jit
def allocate_local_buffer(XBLOCK: tl.constexpr):
    # 默认地址空间
    bl.alloc(tl.float32, [XBLOCK])
    # UB 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)
    # L1 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L1)
    # L0A 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0A)
    # L0B 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0B)
    # L0C 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0C)
    # 独占内存
    bl.alloc(
        tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB, is_mem_unique=True
    )
```

---

## 2. bl.to_buffer -- Tensor 转 Buffer

### 背景

用于将 `tl.tensor` 张量对象转换为昇腾硬件专用的 `bl.buffer` 缓冲区对象，是张量与硬件内存缓冲区的核心转换接口。

### 接口说明

```python
def to_buffer(
    tensor: tl.tensor,
    space: address_space = None,
    bind_buffer: buffer = None,
    _builder=None
) -> buffer
```

### 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `tensor` | `tl.tensor` | 是 | 需要转换为缓冲区的输入张量 |
| `space` | `bl.address_space` | 否 | 指定目标缓冲区所在的昇腾硬件地址空间 |
| `bind_buffer` | `bl.buffer` | 否 | 将张量直接绑定到指定的目标缓冲区 |
| `_builder` | - | 内部参数 | 编译器自动传参，用户无需使用 |

### 返回值

- 返回与输入张量对应的 `bl.buffer` 对象
- 若传入 `bind_buffer` 参数，直接返回该绑定缓冲区本身

### 约束说明

- 接口约束规则与 `bl.alloc` 保持一致
- 地址空间参数需严格匹配昇腾硬件支持的内存区域（UB/L1/L0A/L0B/L0C）

### 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

@triton.jit
def to_buffer_kernel():
    # 1. 基础转换：无指定地址空间
    a = tl.full((32, 2, 4), 0, dtype=tl.int64)
    a_buf = bl.to_buffer(a)
    # 2. 转换并指定 UB 地址空间
    b = tl.full((32, 2, 4), 0, dtype=tl.int64)
    b_buf = bl.to_buffer(b, al.ascend_address_space.UB)
    # 3. 转换并指定 L1 地址空间
    c = tl.full((32, 2, 4), 0, dtype=tl.int64)
    c_buf = bl.to_buffer(c, al.ascend_address_space.L1)
    # 4. 转换并指定 L0A 地址空间
    d = tl.full((32, 2, 4), 0, dtype=tl.int64)
    d_buf = bl.to_buffer(d, al.ascend_address_space.L0A)
    # 5. 转换并指定 L0B 地址空间
    e = tl.full((32, 2, 4), 0, dtype=tl.int64)
    e_buf = bl.to_buffer(e, al.ascend_address_space.L0B)
    # 6. 转换并指定 L0C 地址空间
    f = tl.full((32, 2, 4), 0, dtype=tl.int64)
    f_buf = bl.to_buffer(f, al.ascend_address_space.L0C)
```

---

## 3. bl.to_tensor -- Buffer 转 Tensor

### 背景

将 Ascend 上分配的 buffer 转成 `tl.tensor` 并返回。

### 接口说明

```python
def to_tensor(memref: bl.buffer, writable: bool = True, _builder=None) -> tl.tensor
```

### 入参说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `memref` | `bl.buffer` | 是 | 输入 bl.buffer 对象 |
| `writable` | `bool` | 否 | 如果设置成 `True`，返回的 tensor 在 bufferization 过程中允许被原地修改，默认为 `True` |
| `_builder` | - | 内部参数 | 编译器自动传参，用户无需使用 |

### 返回值

`tl.tensor`

### 约束说明

接口约束同 `bl.alloc`。

### 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

@triton.jit
def kernel_func(XBLOCK: tl.constexpr):
    buffer1 = bl.alloc(tl.float32, [XBLOCK])
    # 方法调用形式
    buffer1.to_tensor(writable=True)
    # 函数调用形式
    buffer2 = bl.alloc(tl.float32, [XBLOCK])
    bl.to_tensor(buffer2, writable=True)
```

---

## 4. bind_buffer -- Tensor 绑定到 Buffer

### 硬件背景

将 tensor 绑定到 buffer 上。通过 `bl.to_buffer` 的 `bind_buffer` 参数实现。

### 接口说明

```python
def to_buffer(
    tensor: tl.tensor,
    space: address_space = None,
    bind_buffer: buffer = None,
    _builder=None
) -> buffer
```

### 入参

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `tensor` | `tl.tensor` | 是 | 要转换的 tensor |
| `address_space` | `bl.address_space` | 否 | buffer 所在的地址空间 |
| `bind_buffer` | `bl.buffer` | 否 | 需要绑定到的 target buffer |

### 返回值

如果使用 `bind_buffer` 参数，返回 `bind_buffer` 本身。

### 约束说明

- `bind_buffer` 参数必须是 buffer 类型
- tensor 和 bind_buffer 的 shape 和 element type 必须一致
- 不允许将一个 tensor 与多个 buffer 绑定
- 理论上支持运算的类型都支持
- 实际后端实现时，在 OneShotBufferize 之后替换的是 source 和 target 的 alloc，因此二者的 shape 需要一致

### 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

@triton.jit
def bind_buffer():
    alloc = bl.alloc(tl.float32, [32, 32], al.ascend_address_space.UB)
    tensor = tl.full((32, 32), 0, dtype=tl.float32)
    bl.to_buffer(tensor, bind_buffer=alloc)
```

## 5. al.fixpipe -- L0C 到 UB 数据搬运

### 硬件背景

A5 增加了 L0C 到 UB 的数据通路，为实现此通路，临时方案在前端显式调用此通路。

### 接口说明

```python
def fixpipe(
    src: tl.tensor,
    dst: bl.buffer,
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,
    pre_quant_mode: FixpipePreQuantMode = FixpipePreQuantMode.NO_QUANT,
    pre_relu_mode: FixpipePreReluMode = FixpipePreReluMode.NO_RELU,
    _builder=None,
) -> None

class FixpipeDMAMode(enum.Enum):
    NZ2DN = ascend_ir.FixpipeDMAMode.NZ2DN
    NZ2ND = ascend_ir.FixpipeDMAMode.NZ2ND
    NZ2NZ = ascend_ir.FixpipeDMAMode.NZ2NZ

class FixpipeDualDstMode(enum.Enum):
    NO_DUAL = ascend_ir.FixpipeDualDstMode.NO_DUAL
    COLUMN_SPLIT = ascend_ir.FixpipeDualDstMode.COLUMN_SPLIT
    ROW_SPLIT = ascend_ir.FixpipeDualDstMode.ROW_SPLIT

class FixpipePreQuantMode(enum.Enum):
    NO_QUANT = ascend_ir.FixpipePreQuantMode.NO_QUANT
    F322BF16 = ascend_ir.FixpipePreQuantMode.F322BF16
    F322F16 = ascend_ir.FixpipePreQuantMode.F322F16
    S322I8 = ascend_ir.FixpipePreQuantMode.S322I8

class FixpipePreReluMode(enum.Enum):
    LEAKY_RELU = ascend_ir.FixpipePreReluMode.LEAKY_RELU
    NO_RELU = ascend_ir.FixpipePreReluMode.NO_RELU
    NORMAL_RELU = ascend_ir.FixpipePreReluMode.NORMAL_RELU
    P_RELU = ascend_ir.FixpipePreReluMode.P_RELU
```

### 入参

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `src` | `tl.tensor` | 源张量，必须位于 L0C 内存区域 |
| `dst` | `bl.buffer` | 目标缓冲区，必须位于 UB 内存区域 |
| `dma_mode` | `al.FixpipeDMAMode` | HIVM 数据搬运模式，可选值：NZ2DN、NZ2ND、NZ2NZ |
| `dual_dst_mode` | `al.FixpipeDualDstMode` | 双目标模式控制，仅 NZ2ND/普通模式可启用 |
| `pre_quant_mode` | `al.FixpipePreQuantMode` | 量化/类型转换模式 |
| `pre_relu_mode` | `al.FixpipePreReluMode` | 激活函数模式 |

### 返回值

无返回值，直接使用入参 `dst`。

### 约束说明

- fixpipe 仅支持从 L0C 到 UB 的数据搬运
- src 必须是 `tl.dot` 后的结果
- dst 必须是 memscope 为 UB 的 buffer

### 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

@triton.jit
def fixpipe(A_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    row_matmul = tl.program_id(0)
    offs_i = tl.arange(0, tl.constexpr(M))[:, None]
    offs_k = tl.arange(0, K)

    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)  # [M, K]

    ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.fixpipe(a_vals, ub, dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL)
```

---

## 6. triton_launch_kernel -- C 级别 Kernel 发射接口

### 接口概述

`triton_launch_kernel` 是 Ascend 后端 launcher stub 动态库（`.so`）中导出的 C 语言运行时接口，用于在已通过 CANN runtime 注册 kernel function handle 后，直接发射 Triton 算子到 NPU 上执行。

该接口以 `extern "C"` 方式导出，与标准 Python JIT 调用路径（`@triton.jit` -> `kernel[grid](...)`）**并列独立**。普通用户通过 `@triton.jit` 调用时不经过此函数；它面向需要 C 级别 kernel 发射能力的高级场景。

### 函数签名

```c
extern "C" {
void triton_launch_kernel(
    const char* kernelName,
    const void* func,
    rtStream_t stream,
    int gridX,
    int gridY,
    int gridZ,
    const int64_t* shapes_data,
    const int* shape_dims,
    int num_tensors,
    const int* tensor_kinds,
    const void* const* kernel_args,
    const size_t* arg_sizes,
    int num_args
);
}
```

### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `kernelName` | `const char*` | kernel 名称字符串，用于日志和 profiling |
| `func` | `const void*` | CANN runtime 注册后的 kernel function handle |
| `stream` | `rtStream_t` | CANN 运行时 stream 句柄 |
| `gridX` | `int` | 启动网格 X 维度 |
| `gridY` | `int` | 启动网格 Y 维度 |
| `gridZ` | `int` | 启动网格 Z 维度 |
| `shapes_data` | `const int64_t*` | 各 tensor shape 拼接的一维数组，可为 `nullptr` |
| `shape_dims` | `const int*` | 每个 tensor 的维度数，与 shapes_data 配对 |
| `num_tensors` | `int` | tensor 总数 |
| `tensor_kinds` | `const int*` | tensor 类型标记：0=INPUT, 1=OUTPUT, 2=INPUT_OUTPUT |
| `kernel_args` | `const void* const*` | kernel 参数指针数组 |
| `arg_sizes` | `const size_t*` | 各 kernel 参数大小（字节） |
| `num_args` | `int` | kernel 参数个数 |

### launch_args 内存布局

函数内部将所有发射参数组装到连续缓冲区中，按以下顺序布局：

```
[ffts_addr] -> [syncBlockLock_ptr] -> [workspace_addr_ptr] ->
[kernel_arg_0] [kernel_arg_1] ... [kernel_arg_N-1] ->
[gridX] [gridY] [gridZ] ->
[DTData]   // 仅当 TRITON_DEVICE_PRINT="true" 时存在
```

### stream 同步策略

| 模式 | 行为 | 函数返回时机 |
|------|------|-------------|
| TaskQueue 启用（默认） | 将 `rtKernelLaunch` 封装为 `std::function`，通过 `triton_async_launch` 提交到任务队列 | 提交后立即返回 |
| TaskQueue 禁用 | 同步执行 `rtKernelLaunch`，随后调用 `rtStreamSynchronize(stream)` | 等待 kernel 执行完成后返回 |

### 调用路径

**路径 A（标准 Python JIT）：** 不经过 `triton_launch_kernel`

```
用户代码: kernel[grid](args...)
  -> NPULauncher.__call__()
  -> Python C 扩展函数 launch()
  -> _launch() 内部函数
  -> rtKernelLaunch() -> NPU 硬件执行
```

**路径 B（直接调用 triton_launch_kernel）：**

```
第三方 C/C++ 代码
  -> dlopen / dlsym 获取 triton_launch_kernel 符号
  -> triton_launch_kernel()
  -> rtKernelLaunch() -> NPU 硬件执行
```

### 适用场景

| 场景 | 推荐路径 |
|------|----------|
| 日常 Triton 算子开发 | 路径 A（`@triton.jit`） |
| 自定义部署流水线 | 路径 B（直接调用） |
| 推理引擎集成 | 路径 B |
| 同签名 kernel 复用 | 路径 B |

### 最小调用示例（C/C++）

```c
#include <cstring>
#include <dlfcn.h>
#include <vector>
#include "rt.h"

typedef void (*triton_launch_kernel_t)(
    const char*, const void*, rtStream_t,
    int, int, int,
    const int64_t*, const int*, int, const int*,
    const void* const*, const size_t*, int
);

void launch_kernel_via_stub(
    const char* stub_so_path,
    const char* kernel_name,
    const void* func,
    rtStream_t stream,
    int grid_x, int grid_y, int grid_z)
{
    void* handle = dlopen(stub_so_path, RTLD_LAZY);
    auto launch_fn = (triton_launch_kernel_t)dlsym(handle, "triton_launch_kernel");

    float alpha = 1.0f;
    int N = 1024;
    const void* arg_ptrs[] = { &alpha, &N };
    const size_t arg_sizes[] = { sizeof(float), sizeof(int) };

    const int64_t shapes[] = {1, 1024, 1, 1024};
    const int dims[] = {2, 2};
    const int kinds[] = {0, 1};  // INPUT, OUTPUT

    launch_fn(
        kernel_name, func, stream,
        grid_x, grid_y, grid_z,
        shapes, dims, 2, kinds,
        arg_ptrs, arg_sizes, 2
    );

    dlclose(handle);
}
```

### 环境变量影响

| 环境变量 | 默认值 | 影响 |
|----------|--------|------|
| `TRITON_COMPILE_ONLY` | `"false"` | 为 `"true"` 时跳过 kernel 发射，仅编译 |
| `TRITON_DEVICE_PRINT` | `"false"` | 为 `"true"` 时启用 device printf |
| `TRITON_ENABLE_TASKQUEUE` | `"true"` | 为 `"true"` 时启用异步 TaskQueue 模式 |
| `TRITON_GRID_WARN_PRINT` | `"false"` | 为 `"true"` 时 grid 超物理核数则输出性能警告 |

### 限制与注意事项

- 仅支持昇腾 NPU 平台，不支持 GPU
- 函数返回类型为 `void`，不提供稳定错误码契约
- 参数校验失败时静默返回
- 不同编译产物生成的 `triton_launch_kernel` 不可互换使用
- `kernel_args` 深拷贝增加 `sum(arg_sizes[i])` 字节的临时内存开销
- `shapes_data` / `shape_dims` / `tensor_kinds` 主要用于 msprof tensor 信息上报，不传不影响 kernel 正确性
