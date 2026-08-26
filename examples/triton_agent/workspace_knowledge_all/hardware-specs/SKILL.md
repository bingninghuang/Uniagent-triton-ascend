---
name: hardware-specs
description: >
  Ascend NPU 硬件规格索引：各型号 AI Core 数量、VEC/CUBE 核心数、内存容量、带宽等关键参数。按 arch 读取。
argument-hint: >
  默认查看目录和各章节首句；--section "关键硬件约束" 读取特定 ## 或 ### 章节；
  --file references/hw-<arch>.md 读取对应硬件规格；--full 读取全部。
---

# Ascend NPU 硬件规格

## 硬件规格索引（按 arch 读取）

本节列出 11 个 arch（ascend910b1/b2/b2c/b3/b4、910-9362/9372/9381/9382/9391/9392）的规格文件索引，含 AI Core 数、VEC/CUBE 核心数、内存容量、带宽等关键参数。

| arch | 文档 | AI Core | VEC核心数 | CUBE核心数 | GM容量 | L2 Cache | L1 Buffer | UB容量 | 对齐 | VEC吞吐/cycle | 架构特点 |
|------|------|---------|----------|-----------|--------|----------|-----------|--------|------|--------------|---------|
| ascend910b1 | `references/hw-ascend910b1.md` | 24 | 48 | 24 | 64GB | 192MB | 1MB | 192KB | 256B | 2×256B | 传统架构，含 SU |
| ascend910b2 | `references/hw-ascend910b2.md` | 24 | 48 | 24 | 64GB | 192MB | 1MB | 192KB | 256B | 2×256B | 传统架构，含 SU |
| ascend910b2c | `references/hw-ascend910b2c.md` | 24 | 48 | 24 | 64GB | 192MB | 1MB | 192KB | 256B | 2×256B | 传统架构，含 SU |
| ascend910b3 | `references/hw-ascend910b3.md` | 20 | 40 | 20 | 64GB | 192MB | 1MB | 192KB | 256B | 2×256B | 传统架构，含 SU |
| ascend910b4 | `references/hw-ascend910b4.md` | 20 | 40 | 20 | 32GB | 96MB | 1MB | 192KB | 256B | 2×256B | 传统架构，含 SU |
| ascend910-9362 | `references/hw-ascend910-9362.md` | 20 | 40 | 20 | 32GB | 168MB | 512KB | 192KB | 32B | 128B | Cube/Vector 分离，1500MHz |
| ascend910-9372 | `references/hw-ascend910-9372.md` | 20 | 40 | 20 | 64GB | 192MB | 512KB | 192KB | 32B | 128B | Cube/Vector 分离，1800MHz，L2 prefetch |
| ascend910-9381 | `references/hw-ascend910-9381.md` | 24 | 48 | 24 | 64GB | 192MB | 512KB | 192KB | 32B | 128B | Cube/Vector 分离，1800MHz，L2 prefetch |
| ascend910-9382 | `references/hw-ascend910-9382.md` | 24 | 48 | 24 | 64GB | 192MB | 512KB | 192KB | 32B | 128B | Cube/Vector 分离，1800MHz，L2 prefetch |
| ascend910-9391 | `references/hw-ascend910-9391.md` | 24 | 48 | 24 | 64GB | 192MB | 512KB | 192KB | 32B | 128B | Cube/Vector 分离，1850MHz，L2 prefetch |
| ascend910-9392 | `references/hw-ascend910-9392.md` | 24 | 48 | 24 | 64GB | 192MB | 512KB | 192KB | 32B | 128B | Cube/Vector 分离，1850MHz，L2 prefetch |

## 何时读取哪个硬件规格

本节给出按 arch 名称查找对应规格文档以及区分两大架构系列（910B / 93xx）的指引。

### 按 arch 名称查找

根据已知的 arch 名称，按三步定位并读取对应的硬件规格文档。

1. **确定目标 arch**：从编译参数或运行环境中获取 arch 名称（如 `ascend910b4`）。
2. **读取对应文档**：执行 `read_skill --file references/hw-<arch>.md` 读取该型号的完整硬件规格。
3. **关注差异**：不同 arch 在 AI Core 数量、GM 容量、L2 Cache 大小、对齐要求、VEC 吞吐等方面存在差异，直接影响算子的分块策略和性能调优。

### 架构系列说明

910B 系列与 93xx 系列在核心结构、存储容量、对齐和吞吐上有显著差异，需分别理解。

- **910B 系列（b1/b2/b2c/b3/b4）**：传统架构，每个 AI Core 包含 2 VEC + 1 CUBE + 1 SU。L1 Buffer 为 1MB，对齐要求 256B，VEC 每拍处理 2×256 Bytes。
- **910-93xx 系列（9362/9372/9381/9382/9391/9392）**：Cube/Vector 分离架构，无 SU。L1 Buffer 缩减为 512KB，对齐要求降为 32B，VEC 每拍处理 128 Bytes。支持 L2 prefetch、稀疏计算、FixPipe 随路变换（NZ->ND 等）。

## 关键硬件约束（影响算子编写）

本节给出 6 类硬件约束--核心类型与并行、存储层级、对齐要求、CUBE 矩阵乘、数据通路、优化策略，直接影响算子编写。

### 1. 核心类型与并行

- 每个 AI Core 包含 2 个 VEC（向量计算单元）和 1 个 CUBE（矩阵计算单元）。
- VEC 负责逐元素运算、归约等向量操作；CUBE 负责矩阵乘。
- CUBE 与 VEC 可同时执行不同任务，实现流水并行。
- 多个 AI Core 可并行分配任务，需根据 AI Core 数量合理切分 workload。

### 2. 存储层级与容量限制

算子编写需严格遵循各级存储的容量约束：

| 存储层级 | 容量范围 | 用途 | 关键约束 |
|---------|---------|------|---------|
| GM | 32-64GB | 设备主存储 | 全设备共享，HBM 带宽有限 |
| L2 Cache | 96-192MB | 自动缓存 | 多 AI Core 共享，93xx 系列支持 prefetch |
| L1 Buffer | 512KB-1MB | Cube 通用缓存 | 单 AI Core 独享，910B 系列为 1MB，93xx 系列为 512KB |
| L0A | 64KB | 左矩阵 A | `m0 × k0 × sizeof(A.dtype) ≤ 64KB` |
| L0B | 64KB | 右矩阵 B | `k0 × n0 × sizeof(B.dtype) ≤ 64KB` |
| L0C | 128KB | 结果矩阵 C | `m0 × n0 × sizeof(C.dtype) ≤ 128KB`，支持累加 |
| UB | 192KB | 向量运算缓存 | 单 VEC 独享，数据需先搬入 UB 才能 VEC 计算 |

**UB 溢出的编译期症状**：当 kernel 单次预载的 UB 用量超过 192KB（fp32 约 48K 元素，安全值约 32768）上限时，编译器会降级产生 `hivm.hir.vsel` / `hivm.hir.load` `Unsupported op for finding the root alloc` 报错。**这不是 mask/pointer 写法问题**，修复方向是减小 tile 尺寸或对 OC（输出通道）维度做 tiling，而非改写 mask 逻辑。报错信息中的 `requires X bits while Y bits available` 即 UB 溢出的直接证据。

### 3. 对齐要求

对齐要求是算子编写中最关键的约束之一，直接影响数据搬运的合法性：

- **910B 系列（b1/b2/b2c/b3/b4）**：所有 L0/L1/UB 数据传输需 **256 Bytes 对齐**。
- **910-93xx 系列**：所有数据传输需 **32 Bytes 对齐**（UB block size），内部按 128 Bytes 分块处理。

### 4. CUBE 矩阵乘约束

- CUBE 每拍完成 16×16×16 FP16 矩阵乘（8192 FLOPs/cycle）。
- 自动按 16×16 分块，尾块自动补 0 计算。
- 93xx 系列支持更多精度：FP16×FP16->FP16/FP32, FP32×FP32->FP32, HF32×HF32->FP32, INT8×INT8->INT32, INT4/INT2 量化。
- 93xx 系列支持稀疏计算（sparsity=1）。

### 5. 数据通路

数据在各级存储间的搬运通过 MTE（Memory Transfer Engine）完成：

- **MTE1**：L1 -> L0A/L0B（矩阵数据加载到 CUBE 输入）
- **MTE2**：GM -> UB/L1/L0A/L0B（从主存加载数据）
- **MTE3**：UB -> GM, L1 -> L2 Cache（数据写回）
- **FixP**：L0C -> L1/GM（CUBE 结果输出，93xx 系列支持随路量化/反量化/类型转换/ReLU/NZ->ND 变换）

MTE 后台搬运可与 CUBE/VEC 计算重叠，是流水优化的关键。

### 6. 优化策略

- **内存对齐**：所有数据传输必须满足对齐要求（256B 或 32B，取决于 arch）。
- **双缓冲**：L1/UB 双缓冲技术，一块计算一块加载，隐藏访存延迟。
- **数据复用**：调整搬运顺序，让频繁访问数据缓存在 L2；利用 L2 prefetch（93xx 系列）提前加载数据。
- **流水并行**：MTE 搬运与 CUBE/VEC 计算重叠；CUBE 与 VEC 同时执行不同任务。
- **分块策略**：根据 L0A/L0B/L0C 容量和数据类型计算最优 m0/k0/n0 分块大小。

