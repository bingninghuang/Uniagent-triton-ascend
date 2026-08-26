# al.* 数据复制与同步接口

## 1. al.copy -- UB 数据复制

### 背景

功能类似 `copy_from_ub_to_l1`，在 `copy_from_ub_to_l1` 的基础上增加了 UB 到 UB 的复制。原来的 `copy_from_ub_to_l1` 已添加废弃警告，推荐使用 `al.copy`。

### 接口说明

```python
def copy(
    src: tl.tensor | bl.buffer,
    dst: tl.tensor | bl.buffer,
    _builder=None
) -> None
```

### 参数

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `src` | tensor / buffer | 是 | 源数据，位于 UB 上 |
| `dst` | tensor / buffer | 是 | 目标数据，位于 L1 或 UB 上 |

### 返回值

无

### 约束说明

- src 和 dst 必须同时为 tensor 或者 buffer，tensor 暂时不支持
- src 的 address space 必须为 UB，dst 的 address space 必须为 L1 或 UB
- src 和 dst 类型、形状必须相同

### 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

@triton.jit
def copy(A_ptr, A1_ptr, M: tl.constexpr, N: tl.constexpr):
    offs_a = tl.arange(0, M)[:, None]
    offs_b = tl.arange(0, N)[None, :]
    offs_c = (offs_a) * M + (offs_b)
    a_ptr = A_ptr + offs_c
    a_val = tl.load(a_ptr)
    a1_ptr = A1_ptr + offs_c
    a1_val = tl.load(a1_ptr)

    add = tl.add(a_val, a1_val)
    add_ub = bl.to_buffer(add, al.ascend_address_space.UB)

    # UB -> L1 复制
    A_l1 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L1)
    al.copy(add_ub, A_l1)

    # UB -> UB 复制
    A_ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.copy(add_ub, A_ub)
```

---

## 2. al.copy_from_ub_to_l1 -- UB 到 L1 复制（已废弃）

### 硬件背景

昇腾硬件 A5 支持了直接从 UB 复制数据到 L1，避免了先从 UB 到 GM 再从 GM 到 L1 复制两次，可以提高数据复制的效率。

### 接口说明

```python
def copy_from_ub_to_l1(
    src: tl.tensor | bl.buffer,
    dst: tl.tensor | bl.buffer,
    _builder=None
) -> None
```

### 参数

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `src` | tensor / buffer | 是 | 源数据，位于 UB 上 |
| `dst` | tensor / buffer | 是 | 目标数据，位于 L1 上 |

### 约束说明

- src 和 dst 必须同时为 tensor 或者 buffer，tensor 暂时不支持
- src 的 address space 必须为 UB，dst 的 address space 必须为 L1
- src 和 dst 类型、形状必须相同

### 注意

此接口已添加废弃警告，推荐使用 `al.copy` 替代。

---

## 3. al.debug_barrier -- VF 手动同步屏障

### 硬件背景

支持 VF 手动同步。

### 接口说明

```python
class SYNC_IN_VF(enum.Enum):
    VV_ALL = auto()
    VST_VLD = auto()
    VLD_VST = auto()
    VST_VST = auto()
    VS_ALL = auto()
    VST_LD = auto()
    VLD_ST = auto()
    VST_ST = auto()
    SV_ALL = auto()
    ST_VLD = auto()
    LD_VST = auto()
    ST_VST = auto()

@builtin
def debug_barrier(
    sync_mode: SYNC_IN_VF,
    _builder=None,
) -> None:
    return semantic.debug_barrier(sync_mode.name, _builder)
```

### 入参

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `sync_mode` | `al.SYNC_IN_VF` | 指定 barrier 的类型 |

### SYNC_IN_VF 枚举说明

| 类型 | 说明 |
|------|------|
| `VV_ALL` | 阻塞 vector load/store 指令直到所有 vector load/store 指令完成 |
| `VST_VLD` | 阻塞 vector load 指令直到所有 vector store 指令完成 |
| `VLD_VST` | 阻塞 vector store 指令直到所有 vector load 指令完成 |
| `VST_VST` | 阻塞 vector store 指令直到所有 vector store 指令完成 |
| `VS_ALL` | 阻塞 scalar load/store 指令直到所有 vector load/store 指令完成 |
| `VST_LD` | 阻塞 scalar load 指令直到所有 vector store 指令完成 |
| `VLD_ST` | 阻塞 scalar store 指令直到所有 vector load 指令完成 |
| `VST_ST` | 阻塞 scalar store 指令直到所有 vector store 指令完成 |
| `SV_ALL` | 阻塞 vector load/store 指令直到所有 scalar load/store 指令完成 |
| `ST_VLD` | 阻塞 vector load 指令直到所有 scalar store 指令完成 |
| `LD_VST` | 阻塞 vector store 指令直到所有 scalar load 指令完成 |
| `ST_VST` | 阻塞 vector store 指令直到所有 scalar store 指令完成 |

### 约束

- 仅可在 `al.scope` 中使用（目前未拦截）

### 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def triton_sub(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 - tmp1
        tl.debug_barrier()
        tl.store(out_ptr0 + (x0), tmp2, None)
```

## 4. al.sync_block_all -- 全核同步

### 硬件背景

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数插入同步语句来避免数据读写错误。

### 接口说明

```python
def sync_block_all(mode, event_id, _builder=None)
```

### 入参

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `mode` | `str` | 是 | 同步模式，可选值见下表 |
| `event_id` | `int` | 是 | 标记 ID，范围 `[0, 15]` |

### mode 可选值

| mode 值 | 说明 |
|---------|------|
| `"all_cube"` | 同步所有 Cube 核 |
| `"all_vector"` | 同步所有 Vector 核 |
| `"all"` | 同步所有 Cube 核和 Vector 核 |
| `"all_sub_vector"` | Vector 子块间同步 |

### 返回值

无

### 约束

- mode 可选字符串：`all_cube` / `all_vector` / `all` / `all_sub_vector`
- event_id 范围是 `[0, 15]`

### 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def test_sync_block_all():
    al.sync_block_all("all_cube", 8)
    al.sync_block_all("all_vector", 9)
    al.sync_block_all("all", 10)
    al.sync_block_all("all_sub_vector", 11)
```

---

## 5. al.sync_block_set -- 分离模式核间同步（发送端）

### 硬件背景

面向分离模式的核间同步控制接口。与 `sync_block_wait` 配合使用。使用时需传入核间同步的标记 ID（flagId），每个 ID 对应一个初始值为 0 的计数器。执行 `CrossCoreSetFlag` 后 ID 对应的计数器增加 1；执行 `CrossCoreWaitFlag` 时如果对应的计数器数值为 0 则阻塞不执行；如果对应的计数器大于 0，则计数器减一，同时后续指令开始执行。

### 接口说明

```python
def sync_block_set(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _builder=None)

class PIPE(enum.Enum):
    PIPE_S = ascend_ir.PIPE.PIPE_S
    PIPE_V = ascend_ir.PIPE.PIPE_V
    PIPE_M = ascend_ir.PIPE.PIPE_M
    PIPE_MTE1 = ascend_ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = ascend_ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = ascend_ir.PIPE.PIPE_MTE3
    PIPE_ALL = ascend_ir.PIPE.PIPE_ALL
    PIPE_FIX = ascend_ir.PIPE.PIPE_FIX
```

### 返回值

无返回值

### 入参说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `sender` | `str` | 发送端，仅支持 `"cube"` / `"vector"` |
| `receiver` | `str` | 接收端，仅支持 `"cube"` / `"vector"` |
| `event_id` | `int` | 同步标记 ID，取值范围 `[0, 15]` |
| `sender_pipe` | `al.PIPE` | 发送端流水线类型 |
| `receiver_pipe` | `al.PIPE` | 接收端流水线类型 |
| `_builder` | - | JIT 编译器自动传参 |

### PIPE 枚举说明

| 流水类型 | 含义 |
|----------|------|
| `PIPE_S` | 标量流水线，使用 Tensor GetValue 函数时为此流水 |
| `PIPE_V` | 矢量计算流水及 L0C -> UB 数据搬运流水 |
| `PIPE_M` | 矩阵计算流水 |
| `PIPE_MTE1` | L1 -> L0A、L1 -> L0B 数据搬运流水 |
| `PIPE_MTE2` | GM -> L1、GM -> L0A、GM -> L0B、GM -> UB 数据搬运流水 |
| `PIPE_MTE3` | UB -> GM、UB -> L1 数据搬运流水 |
| `PIPE_ALL` | 所有流水 |
| `PIPE_FIX` | L0C -> GM、L0C -> L1 数据搬运流水 |

### 约束

- `sender != receiver`

---

## 6. al.sync_block_wait -- 分离模式核间同步（接收端）

### 硬件背景

与 `sync_block_set` 配合使用。详见 `sync_block_set` 的硬件背景说明。

### 接口说明

```python
def sync_block_wait(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _builder=None)
```

参数与 `sync_block_set` 完全一致。

### 返回值

无返回值

### 约束

- `sender != receiver`

### sync_block_set / sync_block_wait 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

# Cube -> Vector 同步
@triton.jit
def kernel_sync_cube_to_vector():
    with al.scope(core_mode="cube"):
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

# Vector -> Cube 同步
@triton.jit
def kernel_sync_vector_to_cube():
    with al.scope(core_mode="vector"):
        al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    with al.scope(core_mode="cube"):
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

# 多 ID 同步
@triton.jit
def kernel_sync_multi_id():
    with al.scope(core_mode="cube"):
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
        al.sync_block_set("cube", "vector", 1, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
        al.sync_block_wait("cube", "vector", 1, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)
```
