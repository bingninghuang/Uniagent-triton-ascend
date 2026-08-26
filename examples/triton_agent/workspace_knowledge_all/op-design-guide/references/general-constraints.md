---
name: general-constraints
description: Triton Ascend 算子通用设计约束速查（G1-G12），跨所有算子类别适用
metadata:
  type: reference
---

# Triton Ascend 算子通用设计约束

本文档汇总跨所有算子类别的通用设计约束。生成任何算子前必须遵守以下全部条目。各约束编号 G1-G8 来自 tensor-transform.md 的跨类别提取，G9-G12 从索引计算类迁移经验中提取。

---

## G1 动态读取 Vector Core 数量，禁止硬编码

- **必须**动态读取实际 Vector Core 数量，禁止硬编码 `num_cores=8` 或 `num_cores=48`。
- **正确做法**：
  ```python
  VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)
  ```
  或通过 `triton.runtime.driver.active.utils.get_device_properties(device)` 读取 `num_vectorcore` / `num_aicore`。
- **Why:** 硬编码仅利用部分 Vector Core，导致加速比显著下降；不同 NPU 型号核数不同。

## G2 BLOCK 向上取 2 的幂（NPU 向量化友好）

- **必须**将 BLOCK / MAX_IN_W 等向量化长度声明为 `triton.next_power_of_2(N)`，且作为 `tl.constexpr` 传入 kernel。
- **Why:** Ascend 上 fixed-shape vector load 必须是编译期常量长度；若 `tl.arange(0, N)` 中 N 非 constexpr 会触发 dynamic-shape load，退化为标量循环。

## G3 禁止单 kernel 统所有路径，必须多策略分派

- **必须**按 `(维度, 模式, 采样比例)` 等特征在 host 侧分派到特化 kernel。
- **禁止**用单一通用 kernel 统一处理所有路径（坐标解码 overhead 极大，且丢失特化机会）。
- **通用 kernel 仅作兜底**：仅当不存在特化匹配时使用。

## G4 Grid 总数不超过核数

- **必须**`grid = (min(total_blocks, num_cores),)`，禁止直接 `grid = (total_blocks,)`。
- **Why:** 输出规模大时 total_blocks 可能远超核数，超 grid 上限；且多余 program 会空跑。
- **2D grid 例外**：当 `total_blocks <= num_cores` 时可用 2D grid 充分利用多核。

## G5 int32 索引，避免 int64 降级

- **必须**在 kernel 内将 `tl.program_id` 和 `tl.arange` 结果 `.to(tl.int32)`。
- **Why:** int64 标量会触发地址计算降级；NPU 上 int32 索引更高效。
  ```python
  pid = tl.program_id(0).to(tl.int32)
  offs = (block_start + tl.arange(0, BLOCK)).to(tl.int32)
  ```

## G6 多核负载均衡分配公式

- **必须**按输出元素总数（非输入元素）分配核数，每个 program 处理一段连续输出。
- **负载均衡公式**（确保每个 core 处理的 block 数差不超过 1）：
  ```python
  blocks_per_core = total_blocks // num_cores
  remainder = total_blocks - blocks_per_core * num_cores
  if pid < remainder:
      my_blocks = blocks_per_core + 1
      start_block = pid * (blocks_per_core + 1)
  else:
      my_blocks = blocks_per_core
      start_block = remainder * (blocks_per_core + 1) + (pid - remainder) * blocks_per_core
  ```

## G7 输入必须 contiguous

- **必须** Host 侧进入 kernel 前调用 `x = x.contiguous()`。
- **Why:** 避免非连续张量导致 kernel 内 stride 计算复杂化。

## G8 坐标比较转 float32

- **禁止**直接对整数坐标使用 `tl.where(coord < 0, ...)`。
- **必须**先 `.to(tl.float32)` 再比较。
- **Why:** Triton Ascend 整数比较可能降级；同时 `tl.cast` 对负数是向零截断而非 floor，坐标计算需用 float32 比较修正。

## G9 索引数据类型约束（禁用 fp64/uint）

- **必须**将索引张量进入 Vector 路径前统一转为 `tl.int32`。
- **必须**在计算全局扁平偏移时，对可能溢出的偏移使用 `tl.int64`。
- **禁止**使用 `fp64`、`uint` 类型。
- **Why:** NPU 上 int32 索引更高效；fp64 和 uint 在 Ascend 上不支持或性能极差。

## G10 fp16/bf16 累加必须使用 fp32 workspace

- **必须**在 fp16/bf16 scatter/accumulate 场景，分配 `torch.float32` 中间 buffer，最后 cast 回原始 dtype。
- **禁止**直接用 `tl.atomic_add` 累加 fp16/bf16 输出。
- **Why:** fp16/bf16 累加精度不足，会导致 verify 失败。

## G11 向量 tl.atomic_add 优先于标量循环

- **必须**在连续输出段使用向量 `tl.atomic_add(ptr + tl.arange(0, BLOCK_E), vals, mask=...)`。
- **禁止**用 `for e in range(BLOCK_E): tl.atomic_add(...)` 标量循环。
- **标量 atomic 仅作为 fallback**：仅在不连续/冲突密集场景使用。
- **Why:** 向量 atomic 一次覆盖多个连续元素，原子数下降 1~2 个数量级；标量拆分会破坏 SIMD。

## G12 UB 容量校验与 tiling

- **必须**保证 UB preload 尺寸小于 UB 容量上限（fp32 约 32768 元素，即 192KB / 4B = 48K 元素；实际安全值约 32768）。
- **必须**在 `dim_size > UB_LIMIT` 时分块为 `BLOCK_*` + `BLOCK_*_SUB`。
- **Why:** 一次性预载整行但 `dim_size > UB_LIMIT` 会导致 spilling，反而更慢。
- **编译期症状**：UB 溢出会触发编译器降级，报错 `hivm.hir.vsel` / `hivm.hir.load` `Unsupported op for finding the root alloc`，常被误判为 mask/循环写法问题。**此报错不是 mask 问题**，修复方向是减小 tile 或对 OC 维度 tiling，不要误改为改写 mask 逻辑。

---

## 分区策略说明

关于交错分区（interleave）与连续分区（contiguous partitioning）：

- **交错分区**（`for block_idx in range(pid, num_blocks, num_cores)`）：在 `total_blocks <= num_cores` 时可接受，天然负载均衡。
- **连续分区**（每个 core 处理连续的 block 范围）：对于内存密集型算子优先使用，以获得更好的缓存局部性。
- 两种策略均可接受，根据算子特征选择：计算密集型可用交错，内存密集型优先连续。
