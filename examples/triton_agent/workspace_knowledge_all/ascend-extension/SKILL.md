---
name: ascend-extension
description: >
  Ascend 扩展 API：al.* 地址空间操作（copy、sync、scope、subview）、bl.* buffer 级操作（alloc、bind_buffer、fixpipe、to_tensor）、libdevice 数学函数库。高级算子开发时使用。
argument-hint: >
  默认查看目录和各章节首句；--section "al.*" 读取特定章节；
  --file references/<topic>.md 读取对应参考；--full 读取全部。
---

# Ascend 扩展 API

## 概述

Triton Ascend 提供了三层扩展 API，用于在标准 Triton `tl.*` 之外进行更底层的硬件控制。这些 API 面向**高级算子开发**场景，当标准 Triton API 无法满足性能调优或内存管理需求时使用。

### 何时使用这些扩展

- **需要显式控制 Ascend 硬件内存层级**（UB/L1/L0A/L0B/L0C）时
- **需要 Cube/Vector 核分离编程**时（`al.scope`）
- **需要跨核同步**时（`al.sync_block_*`）
- **需要 L0C -> UB 数据搬运**时（`al.fixpipe`）
- **需要 buffer 级内存管理**时（`bl.alloc`、`bl.to_buffer`、`bl.to_tensor`）
- **需要 SIMT 模式下的逐元素数学函数**时（`libdevice`）
- **需要 C 级别 kernel 发射**时（`triton_launch_kernel`）

### 导入方式

```python
import triton.language.extra.cann.extension as al      # al.* 接口
import triton.extension.buffer.language as bl            # bl.* 接口
import triton.language.extra.cann.libdevice as libdevice # libdevice 数学函数
```

## API 分类索引

Triton Ascend 提供三层扩展 API：al.*（地址空间声明与显式拷贝同步）、bl.*（buffer 级内存管理）、libdevice（SIMT 数学函数），用于标准 tl.* 之外的底层硬件控制。

### 1. al.* -- 地址空间与同步操作

al.* 含 references/al-address-space.md（地址空间枚举、scope 核分离、sub_vec_id、subview）和 references/al-copy-and-sync.md（al.copy、debug_barrier、sync_block_all/set/wait）两类 API，用于地址空间声明与显式拷贝同步。

| 接口 | 功能 | 参考文件 |
|------|------|----------|
| `al.ascend_address_space` | 地址空间枚举（UB/L1/L0A/L0B/L0C） | `references/al-address-space.md` |
| `al.scope` | Cube/Vector 核分离作用域 | `references/al-address-space.md` |
| `al.sub_vec_id` | 获取 Vector 子核 ID | `references/al-address-space.md` |
| `al.subview` | buffer 子视图（零拷贝切片） | `references/al-address-space.md` |
| `al.copy` | UB -> L1/UB 数据复制 | `references/al-copy-and-sync.md` |
| `al.copy_from_ub_to_l1` | UB -> L1 数据复制（已废弃，用 al.copy 替代） | `references/al-copy-and-sync.md` |
| `al.debug_barrier` | VF 手动同步屏障 | `references/al-copy-and-sync.md` |
| `al.sync_block_all` | 全核同步 | `references/al-copy-and-sync.md` |
| `al.sync_block_set` | 分离模式核间同步（发送端） | `references/al-copy-and-sync.md` |
| `al.sync_block_wait` | 分离模式核间同步（接收端） | `references/al-copy-and-sync.md` |

### 2. bl.* -- Buffer 级操作

bl.* 含 bl.alloc（分配 buffer）、bl.to_buffer（tensor→buffer）、bl.to_tensor（buffer→tensor）、bind_buffer（绑定已有 buffer）、al.fixpipe（L0C→UB 搬运）、triton_launch_kernel（C 级 kernel 发射），详见 references/bl-buffer-ops.md，用于 buffer 级内存管理。

| 接口 | 功能 | 参考文件 |
|------|------|----------|
| `bl.alloc` | 分配指定地址空间的 buffer | `references/bl-buffer-ops.md` |
| `bl.to_buffer` | tensor -> buffer 转换 | `references/bl-buffer-ops.md` |
| `bl.to_tensor` | buffer -> tensor 转换 | `references/bl-buffer-ops.md` |
| `bind_buffer`（通过 `bl.to_buffer` 的 `bind_buffer` 参数） | 将 tensor 绑定到已有 buffer | `references/bl-buffer-ops.md` |
| `al.fixpipe` | L0C -> UB 数据搬运（含量化/激活选项） | `references/bl-buffer-ops.md` |
| `triton_launch_kernel` | C 级别 kernel 发射接口 | `references/bl-buffer-ops.md` |

### 3. libdevice -- SIMT 数学函数库

提供 167 个逐元素数学函数，仅在 SIMT 编译模式下使用。涵盖三角函数、指数对数、舍入运算、类型转换、位操作、贝塞尔函数等。

| 函数类别 | 代表函数 | 参考文件 |
|----------|----------|----------|
| 三角函数 | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | `references/libdevice.md` |
| 双曲函数 | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | `references/libdevice.md` |
| 指数对数 | `exp`, `log`, `exp2`, `log2`, `exp10`, `log10` | `references/libdevice.md` |
| 舍入运算 | `add_rd/rn/ru/rz`, `mul_rd/rn/ru/rz`, `div_rd/rn/ru/rz` | `references/libdevice.md` |
| 类型转换 | `float2int_rd/rn/ru/rz`, `int2float_rd/rn/ru/rz` | `references/libdevice.md` |
| 位操作 | `brev`, `clz`, `popc`, `ffs`, `byte_perm` | `references/libdevice.md` |
| 特殊函数 | `erf`, `erfc`, `lgamma`, `tgamma`, `gamma` | `references/libdevice.md` |
| 快速近似 | `fast_sinf`, `fast_cosf`, `fast_expf`, `fast_logf` | `references/libdevice.md` |
| 融合乘加 | `fma`, `fma_rd/rn/ru/rz` | `references/libdevice.md` |
| 判断函数 | `isinf`, `isnan`, `isfinited`, `signbit` | `references/libdevice.md` |

## 参考文件索引

本节列出 4 个参考文件：al-address-space.md（地址空间与 scope）、al-copy-and-sync.md（拷贝与同步）、bl-buffer-ops.md（buffer 操作）、libdevice.md（167 个数学函数）。

| 文件 | 内容 |
|------|------|
| `references/al-address-space.md` | 地址空间枚举、scope 核分离、sub_vec_id、subview |
| `references/al-copy-and-sync.md` | al.copy、copy_from_ub_to_l1、debug_barrier、sync_block_all/set/wait |
| `references/bl-buffer-ops.md` | alloc、to_buffer、to_tensor、bind_buffer、fixpipe、triton_launch_kernel |
| `references/libdevice.md` | libdevice 167 个数学函数完整参考 |

## 使用注意事项

使用 al.*/bl.* 需注意：仅限 @triton.jit 内核中使用、tensor 与 buffer 不可直接互赋值、libdevice 仅支持 SIMT 模式、fixpipe 仅支持 L0C→UB、scope 中跨作用域需显式同步。

1. **al.* 和 bl.* 接口仅可在 `@triton.jit` 修饰的内核函数中使用**
2. **tensor 和 buffer 不可直接互相赋值**，必须通过 `bl.to_tensor` / `bl.to_buffer` 显式转换
3. **`al.scope` 中 cube 和 vector 作用域并行执行**，跨作用域数据依赖需要显式同步
4. **libdevice 函数默认仅支持 SIMT 编译模式**，需通过 `force_simt_only=True` 启用
5. **`al.fixpipe` 仅支持 L0C -> UB 方向**，且 src 必须是 `tl.dot` 后的结果
6. **`triton_launch_kernel` 是 C 级别接口**，普通用户通过 `@triton.jit` 调用时不会经过此函数
