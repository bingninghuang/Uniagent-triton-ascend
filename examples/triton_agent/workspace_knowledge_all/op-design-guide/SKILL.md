---
name: op-design-guide
description: >
  Triton Ascend 算子类别设计指南索引：按算子类别提供 L1/L2/L3 级别的设计约束、算法骨架和关键 kernel 实现。
  覆盖 7 大类别：tensor-transform / index-computation / math-compute / normalization / object-detection / quantization / transformer-inference。
argument-hint: >
  默认查看类别索引和通用约束；--category "索引计算" 查看特定类别各章节首句；
  --file references/<category>.md 读取对应类别完整设计指南；--file references/general-constraints.md 读取通用约束速查。
---

# Triton Ascend 算子类别设计指南

## 算子类别识别表

根据算子语义选择对应的设计指南文件。生成代码前**必须**先读取对应类别的完整文档。

| 类别 | 参考文件 | 典型算子 | 核心优化哲学 |
|------|---------|----------|-------------|
| tensor-transform | `references/tensor-transform.md` | Interpolate / Pad / Repeat | 多策略分派 + 精度匹配 + 连续访存 |
| index-computation | `references/index-computation.md` | Index / Gather / Scatter / Sort | 维度特化分派 + 向量化读写/原子 |
| math-compute | `references/math-compute.md` | Sum / FFT / Sort | 归约维选择 + 累加精度 + 分段策略 |
| normalization | `references/normalization.md` | GroupNorm / AdaIN | 跨 tile 两阶段归约 + fp32 累加 |
| object-detection | `references/object-detection.md` | NMS / IOU | fused kernel + 2D Tiling + broadcast |
| quantization | `references/quantization.md` | DynamicQuant / SwigluQuant | static/dynamic 分派 + 精度对齐 |
| transformer-inference | `references/transformer-inference.md` | RoPE / MoE / Attention | per-position 向量化 + 寄存器 top-k |

> 通用约束速查：`references/general-constraints.md`（G1-G12，跨所有类别适用）

---

## 通用设计约束

以下 8 条跨类别约束在生成任何算子前必须遵守。完整说明见 `references/general-constraints.md`。

| 编号 | 约束 | 要点 |
|------|------|------|
| G1 | 动态读取 Vector Core 数量 | `torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)`，禁止硬编码 |
| G2 | BLOCK 取 2 的幂 | `triton.next_power_of_2(N)` 作为 `tl.constexpr` 传入 |
| G3 | 多策略分派 | 按 (维度, 模式, 比例) 在 host 侧分派特化 kernel，禁止单 kernel 统所有路径 |
| G4 | Grid 不超核数 | `grid = (min(total_blocks, num_cores),)`，1D grid |
| G5 | int32 索引 | `tl.program_id` 和 `tl.arange` 结果 `.to(tl.int32)`，避免 int64 降级 |
| G6 | 负载均衡公式 | 按**输出**元素数分配核数，block 数差不超过 1 |
| G7 | 输入必须 contiguous | Host 侧 `x = x.contiguous()` |
| G8 | 坐标比较转 float32 | 禁止直接对整数坐标 `tl.where`，先 `.to(tl.float32)` |

> **分区策略**：交错分区（interleave）在 `total_blocks <= num_cores` 时可接受；对于内存密集型算子，优先使用连续分区（contiguous partitioning）以获得更好的缓存局部性。

---

## 使用说明

本节给出算子生成前的必读步骤、各类别文件读取路径和关键注意事项，确保在编写代码前完成正确的准备工作。

### 生成算子前的必读步骤

依次完成五步：识别算子类别 -> 读取通用约束 -> 读取类别设计指南 -> 遵守 L1 硬性约束 -> 参考 L2 算法骨架，最后学习 L3 优化技巧。

1. **识别算子类别**：根据算子语义对照上方识别表，确定所属类别。
2. **读取通用约束**：执行 `read_skill --skill op-design-guide --file references/general-constraints.md` 阅读 G1-G12 完整说明。
3. **读取类别设计指南**：执行 `read_skill --skill op-design-guide --file references/<category>.md` 阅读该类别的 L1 硬性约束、L2 算法骨架和 L3 关键 kernel 实现。
4. **遵守 L1 约束**：类别文档中的 Layer 1 约束是硬性的，首次生成必须全部满足。
5. **参考 L2 骨架**：Layer 2 的 host 侧分派决策树和 kernel 骨架必须一次写对。
6. **学习 L3 技巧**：Layer 3 的关键 kernel 实现贴出优化重点和易错代码，可参考但实现方式可不同。

### 各类别文件路径

下表列出 general-constraints 及 7 个算子类别（tensor-transform / index-computation / math-compute / normalization / object-detection / quantization / transformer-inference）共 8 个文件的读取命令，每个文件对应一种算子语义。

| 类别 | 读取命令 |
|------|---------|
| tensor-transform | `read_skill --skill op-design-guide --file references/tensor-transform.md` |
| index-computation | `read_skill --skill op-design-guide --file references/index-computation.md` |
| math-compute | `read_skill --skill op-design-guide --file references/math-compute.md` |
| normalization | `read_skill --skill op-design-guide --file references/normalization.md` |
| object-detection | `read_skill --skill op-design-guide --file references/object-detection.md` |
| quantization | `read_skill --skill op-design-guide --file references/quantization.md` |
| transformer-inference | `read_skill --skill op-design-guide --file references/transformer-inference.md` |
| general-constraints | `read_skill --skill op-design-guide --file references/general-constraints.md` |

### 关键注意事项

注意区分三个层次：L1 硬性约束必须全部满足，L2 算法骨架须一次写对，L3 优化技巧可灵活参考；同时禁止混用不同类别的经验。

- 各类别文档按 Layer 1（硬性约束）-> Layer 2（算法骨架）-> Layer 3（关键技巧）组织。
- **禁止混用**不同类别的经验：每个类别的优化哲学不同，交叉引用通用约束即可。
- 类别文档中的性能基准（加速比区间）仅供参考，实际性能取决于具体 shape 和 dtype。
- `template.md` 未包含在此 skill 中（仅为骨架模板，无技术内容）。
