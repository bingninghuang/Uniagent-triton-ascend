---
name: examples
description: >
  Triton Ascend 算子开发参考：基础知识（API/Grid/内存/性能/调试）、示例概览、按算子类型的优化策略指南（Element-wise、MatMul、Reduce、Attention、Sort/Select、Interpolate、Layout-transform）。每个参考文件包含代码片段、优化策略和 Ascend NPU 特定约束。
argument-hint: >
  默认查看目录和各章节首句；--section "参考文件索引" 读取特定章节；
  --file references/<topic>.md 读取对应参考文件；--full 读取全部。
---

# Triton Ascend 算子开发参考

本 Skill 包含 Triton Ascend 算子开发的完整参考材料，覆盖基础知识、示例概览和按算子类型的优化策略指南。所有内容面向昇腾 NPU 平台。

## 参考文件索引

本节列出 9 个参考文件，分为基础知识与 API、按算子类型的优化策略指南两组。

### 基础知识与 API

本子节含 triton-ascend-fundamentals.md（API 参考手册、编程基础、Grid 配置、内存与性能优化、调试清单）和 triton-ascend-examples.md（4 个核心示例概览）两个文件。

| 参考文件 | 内容 | 用途 |
|---------|------|------|
| `references/triton-ascend-fundamentals.md` | API 参考手册 + 编程基础（标准五步模式、内核启动、Mask）+ Grid 配置策略（大 shape 处理、动态核心数）+ 内存访问优化 + 性能优化 + 调试清单 | 编写任何算子前必读；遇到 API 不确定时查阅 |
| `references/triton-ascend-examples.md` | 4 个核心示例（Vector Add / MatMul / Layer Norm / Double Kernel）的概览和关键代码结构 | 快速了解 Triton Ascend 的常见代码骨架 |

### 按算子类型的优化策略指南

本子节含 7 个文件：triton-ascend-elementwise.md（加法/激活）、triton-ascend-matmul.md（矩阵乘）、triton-ascend-reduce.md（归约/归一化）、triton-ascend-attention.md（注意力）、triton-ascend-sort-select.md（NMS/TopK）、triton-ascend-interpolate.md（上/下采样）、triton-ascend-layout-transform.md（permute/transpose）。

| 算子类型 | 参考文件 | 适用算子 | 关键策略 |
|---------|---------|---------|---------|
| Element-wise | `references/triton-ascend-elementwise.md` | add/mul/relu/sigmoid/gelu/exp 等 | 连续内存访问、核内循环优化、VEC 核心数、MishBackward/GELU 数学近似 |
| MatMul | `references/triton-ascend-matmul.md` | matmul/bmm/linear/gemm | 512B 行宽对齐、固定 CUBE 核心数启动、核间循环处理多块 |
| Reduce | `references/triton-ascend-reduce.md` | sum/mean/max/min/softmax/layernorm | 块内归约 + 原子操作、FP32 累加防精度损失、exp 前减最大值 |
| Attention | `references/triton-ascend-attention.md` | self-attention/flash-attention/scaled-dot-product | 分块计算 + 在线 Softmax，避免存储完整注意力矩阵 |
| Sort/Select | `references/triton-ascend-sort-select.md` | NMS/TopK/ArgSort | 禁止 break/continue，用 tl.where + mask；tile-wise partial sort + merge |
| Interpolate | `references/triton-ascend-interpolate.md` | nearest/bilinear/bicubic/area | 坐标映射精度、边界 clamp、Keys' bicubic 权重、标量循环策略 |
| Layout-transform | `references/triton-ascend-layout-transform.md` | permute/transpose/reshape-as-copy | 模式分发 + 专用 tile-based kernel，禁止单一 generic gather |

## 按算子类型选择参考

本节给出按算子语义选择参考文件的决策表，列出常见算子与对应参考文件的映射关系。

- **Element-wise 算子**（加法、乘法、激活函数等）：`references/triton-ascend-elementwise.md`
- **Reduce 算子**（sum、max、softmax 等）：`references/triton-ascend-reduce.md`
- **Normalize 算子**（layer_norm、rms_norm 等）：`references/triton-ascend-reduce.md`（归一化属于 reduce 类）
- **Cube 算子**（matmul、attention 等）：`references/triton-ascend-matmul.md`（基础）和 `references/triton-ascend-attention.md`（进阶）
- **Sort/Select 算子**（NMS、TopK）：`references/triton-ascend-sort-select.md`
- **Interpolate 算子**（上/下采样）：`references/triton-ascend-interpolate.md`
- **Layout-transform 算子**（permute、transpose）：`references/triton-ascend-layout-transform.md`
- **基础知识与 API 查询**：`references/triton-ascend-fundamentals.md`
- **代码骨架参考**：`references/triton-ascend-examples.md`

## 使用建议

本节概括使用要点：先读基础知识再按算子类型加载对应指南，遇到性能或调试问题查阅 fundamentals 对应章节，大 shape 参考 Grid 配置策略，Sort/Select 类算子务必先阅读其约束。

1. 阅读 `references/triton-ascend-fundamentals.md` 了解 API、编程模式（标准五步模式）、Grid 配置、内存与性能优化、调试清单
2. 不熟悉 Triton Ascend 代码骨架时，参考 `references/triton-ascend-examples.md` 中的 4 个核心示例概览
3. 根据算子类型加载对应的优化策略指南；融合算子可同时加载多个参考文件
4. 遇到性能问题时，参考 `triton-ascend-fundamentals.md` 中的「性能优化」和「内存访问优化」章节，以及各算子类型指南中的优化策略
5. 遇到调试问题时，参考 `triton-ascend-fundamentals.md` 中的「调试清单」章节，以及各算子类型指南中的「常见错误」/「调试 Checklist」
6. 处理大 shape 时，参考 `triton-ascend-fundamentals.md` 中「Grid 配置策略」的交错循环和连续分块法
7. 编写 Sort/Select 类算子前阅读其约束章节——Triton Ascend 不支持 `break`/`continue`/`return`，必须用 `tl.where` + mask 表达条件逻辑
