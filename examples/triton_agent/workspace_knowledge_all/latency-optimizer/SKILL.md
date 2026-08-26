---
name: latency-optimizer
description: >
  擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
  按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点，
  确保优化前后功能一致、精度一致。
  ⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。
  ⚠️ 仅在 run_verify 报告 verified_success: true 后使用。
argument-hint: >
  默认查看目录和各章节首句；--section "优化点索引" 查看完整索引表；
  --section "主流程" 查看执行流程；
  --file references/optimization-points/xx.md 读取具体优化文档；
  --file references/checklist.md 读取代码规范。
---

# Latency Optimizer Skill

<role>
你是一个擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
你的任务是按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点。
**必须确保优化前后的功能一致性和精度一致性。**
**⚠️ 仅在 run_verify 报告 verified_success: true 后使用本 skill。**
**⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。**
</role>

## 优化点索引

以下仅列出优化点索引，包含适用条件、命中条件及参考文档路径。**每个优化点的详细说明（典型代码特征、判断逻辑、优化方法、代码示例）请见对应参考文档。**

| 序号 | 优化点 | 适用条件 | 命中条件 | 参考文档 |
|------|--------|----------|----------|----------|
| 1 | 入参静态化优化 | 存在可声明为 `tl.constexpr` 的固定参数 | 单次 kernel 启动后不变的参数未声明 `tl.constexpr` | `references/optimization-points/01-constexpr-parameters.md` |
| 2 | Tiling 优化 | 多维张量规约/归一化算子，规约轴非最连续轴 | 分块策略导致跨步访存 | `references/optimization-points/02-tiling-optimization.md` |
| 3 | 分核优化 | Grid 设置不合理或未充分利用 NPU 资源 | Grid 与物理核数严重偏离，或每个 program 处理数据量过小 | `references/optimization-points/03-vector-core-partition.md` |
| 4 | 离散访存优化 | 通过随机/不可预测索引访问全局内存 | 索引来源于 `tl.load` 加载值或 kernel 入参 | `references/optimization-points/04-discrete-memory-access.md` |
| 5 | Scalar 转 Vector 优化 | 存在可转换为向量操作的标量操作 | 存在标量广播、标量规约、标量控制流、`int` 比较/除法/取余、`atomic_*` 标量操作 | `references/optimization-points/05-scalar-to-vector.md` |
| 6 | 避免向量 API 标量降级 | 向量操作可能被编译器降级为标量循环 | 算术/比较/扩展乘法/cumsum/cumprod/reduce 满足降级条件 | `references/optimization-points/06-avoid-scalar-lowering.md` |
| 7 | Pass 消除合并优化 | 多次遍历相同数据计算不同统计量 | 可通过自适应 `BLOCK_SIZE` 消除循环，或可合并多次遍历 | `references/optimization-points/07-pass-merge.md` |
| 8 | 维度合并优化 | 多层嵌套循环处理连续维度且维度间无依赖 | 存在 3 层及以上连续维度嵌套循环可合并 | `references/optimization-points/08-dimension-merge.md` |
| 9 | Libdevice 函数使用 | 手动实现数学函数而 libdevice 已有优化版本 | 存在手动实现的 math 函数且 libdevice 有对应版本 | `references/optimization-points/09-libdevice-usage.md` |
| 10 | 循环不变量外提 | 嵌套循环内层有只依赖外层变量的 `tl.load` | 内层循环重复加载相同值 | `references/optimization-points/10-loop-invariant-hoisting.md` |
| 11 | Load 指令重排序 | 循环内多个 `tl.load`/`tl.store` 因数据依赖阻塞 | 存在可提前发射的 load 指令 | `references/optimization-points/11-load-order.md` |
| 12 | Grid 形状与多路径特化 | 单一 kernel 无法同时覆盖大小 grid 场景 | Host 侧可在运行时根据 workload 选择不同 kernel 路径 | `references/optimization-points/12-grid-dispatch-specialization.md` |
| 13 | Autotune 自动调优 | 存在一个或多个未充分调优的可调参数 | 存在可调 `tl.constexpr` 参数且未使用 `@triton.autotune` | `references/optimization-points/13-autotune.md` |
| 14 | 混合策略自动选择 | 不同 shape 或数据类型需要不同优化策略 | 存在 shape/dtype 相关的条件分支选择不同 kernel/策略 | `references/optimization-points/14-mixed-strategy.md` |
| 15 | 维度合并与大 BLOCK 累加 | 归一化算子中存在嵌套循环/低 mask 覆盖率/标量累加过多 | stats kernel 中连续维度处理低效 | `references/optimization-points/15-dimension-merge-large-block.md` |
| 16 | 连续拷贝聚合优化 | 纯内存拷贝型算子，多个输出块在输入侧连续 | 满足连续性且当前按 chunk 细粒度分核 | `references/optimization-points/16-continuous-copy-aggregation.md` |
| 17 | 消除冗余的边界运算 | `tl.load(..., mask=m, other=d)` 后运算链出现冗余边界保护 | KVR 分析可证存在冗余 `tl.where`、`* mask`、`+ 0` 等 | `references/optimization-points/17-redundant-boundary-operation.md` |
| 18 | Kernel 分裂优化 | 多 Case 场景下泛用 Kernel 性能未达标 | `total_cases > 1` 且 `speedup_vs_torch < 0.8`，存在可特化分组 | `references/optimization-points/18-kernel-splitting.md` |
| 19 | Cube/MTE3 分阶段批量解耦优化 | 多输出 kernel 中 Cube 累加输出与 atomic scatter 输出在同一循环体交替（MTE3 阻塞 Cube），且某归约维靠多 program atomic 竞争归约 | Cube/MTE3 交替阻塞 + 归约维 atomic 爆炸，该维可单 program UB 累加，重算成本可接受 | `references/optimization-points/19-cube-mte3-decoupling.md` |
| 20 | Host 侧张量维度拼接优化 | 算子内存在复合点积 `a·c + b·d`（多次 `tl.dot` + 中间累加），各分段为同一对象连续维度 | 各分段独立存储、内存连续可 `concat`，且拼接后不溢出 UB | `references/optimization-points/20-host-tensor-concat.md` |
| 21 | Workspace 物化解耦优化 | 多输出 kernel 输出间循环遍历顺序冲突（UB 放不下常驻累加器且 atomic 太贵），存在可物化复用的共享中间量 | 多 pass 重复 gather + 重算共享中间量，且 pass 间循环顺序 genuine 冲突无法合并 | `references/optimization-points/21-workspace-decoupling.md` |
| 22 | Latency-Bound 循环维度 Tile 合并 | kernel 处于 latency-bound（算力利用率极低，dot 固定 issue/同步开销主导），存在外层循环每迭代发起一组 dot，且 dot 某维（常 M）小于 cube 微块可放大 | profiling 算力利用率 <5% 且带宽未饱和但 dot 调用频繁，外层循环放大 dot 维度可减迭代数，放大后连续单 tile 在 UB/CC 内 | `references/optimization-points/22-latency-bound-tile-merge.md` |
| 23 | Ascend Interpolate 专用优化 | 算子类型为 interpolate/upsample_* | 代码为 Interpolate 类算子，存在坐标/权重运行时计算或离散访存 | `references/optimization-points/23-ascend-interpolate-optimization.md` |
| 24 | Ascend Pooling 专用优化 | 算子类型为 MaxPool/AvgPool | 代码为 Pooling 类算子，存在 1D 扁平索引或布局/边界优化空间 | `references/optimization-points/24-ascend-pooling-optimization.md` |

**检查规则**：Agent 必须严格按照上述顺序逐一检查优化点，**每次只能尝试一个优化点，命中后才能加载对应参考文档；未命中则跳过，禁止加载参考文档。**

## 算子类别与高频优化点

不同类别算子的性能瓶颈分布不同，以下列出常见类别及其**必须检查**的优化点。
当算子属于对应类别且性能不达标时，这些点不得被跳过。

| 算子类别 | 识别特征 | 高频命中点 | 说明 |
|---------|---------|-----------|------|
| **Tiled Reduction** | 存在 `for t in range(0, N, BLOCK)` 内对 `tl.load` 结果做 `tl.sum` 归约 | 5, 7, 8, 15 | 标量累加器、嵌套循环、mask 覆盖率是核心瓶颈 |
| **Multi-kernel** | stats + apply 双 kernel（BatchNorm/LayerNorm/GroupNorm/InstanceNorm/RMSNorm 等归一化算子） | 5, 7, 8, 15, 18 | 继承 Tiled Reduction 全部瓶颈 + kernel 分裂 |
| **Broadcast EW** | `add/sub/mul/div` 逐元素操作，存在 shape 不等需广播 | 1, 2, 8, 12 | 入参静态化、tiling、多路径调度是关键 |
| **Scatter/Gather** | 通过随机/不可预测索引访问全局内存 | 4, 5 | 离散访存和 scatter-add 并行轴选择 |
| **MatMul** | 矩阵乘法 | 2, 13 | tiling 和 autotune 自动调优 |
| **Memory-bound Copy** | Split/Concat/Pad/Chunk 等纯访存算子 | 16 | 连续拷贝聚合 |
| **Pooling** | MaxPool/AvgPool | 24 | 1D 扁平索引或布局/边界优化 |
| **Interpolate** | interpolate/upsample | 23 | 坐标/权重运行时计算或离散访存 |
| **Permute/Layout-transform** | permute/transpose/reshape-as-copy | 1, 2, 8, 12, 13, 14, 16 | 模式特化、连续维度合并、view 短路；专用 kernel 内部必须是 tile-based 连续访存，禁止 element-wise gather 冒充特化 |

> **Permute/Layout-transform 补充**：若常见模式专用 kernel 内部仍使用逐元素 `div`/`mod` 或 `tl.where` 链进行 gather/scatter，或未通过 `view` 合并连续维度，则优化点 2（Tiling）和 8（维度合并）**必须检查**，不得跳过。

> **通用规则**：多 case（`total_cases > 1`）且 `speedup_vs_torch < 0.8` 时，
> 无论属于哪个类别，优化点 18（Kernel 分裂）**必须检查**，不得跳过。

## 主流程（必须严格执行）

Agent 必须始终处于以下主流程中。进入任一子流程（参考文档、checklist 修复、验证）后，**完成后必须回到主流程继续执行**，禁止在中途跳出或跳过主流程步骤。

### 主流程步骤

1. **按顺序检查优化点**：`1 -> 2 -> 3 -> ... -> 24`。
2. **对当前优化点判断是否命中**：
   - **未命中**（代码特征不满足 或 适用条件不成立）-> 跳过，检查下一优化点。
   - **命中**（代码特征满足 且 适用条件成立）-> 加载对应参考文档，进入子流程应用优化策略；**应用完成后必须回到本步骤继续后续流程**。
3. **代码规范检查**：应用优化后，**必须加载 `references/checklist.md`**，逐项检查代码规范。
4. **规范修复循环**：
   - 如果代码规范不满足 -> 修改代码直到满足所有规范要求，然后重新执行步骤 3。
   - 如果代码规范满足 -> 继续步骤 5。
5. **返回主循环**：以当前优化后的代码为新的检查对象，**回到步骤 1**，继续按顺序检查下一个优化点。

### 主流程示意图

以下流程图直观展示了主流程各步骤之间的执行顺序与回流路径。

```
┌─────────────────────────────────────┐
│ 1. 按顺序检查优化点 1->2->...->24       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. 当前优化点是否命中？              │
│    未命中 ──-> 检查下一优化点 ──-> 回到 1 │
│    命中   ──-> 加载参考文档并应用优化   │
│               （子流程，完成后必须回来） │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. 加载 checklist.md 检查代码规范    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. 规范是否满足？                    │
│    不满足 ──-> 修改代码 ──-> 回到 3   │
│    满足   ──-> 回到 1 继续检查下一个点 │
└─────────────────────────────────────┘
```

### 关键约束

- ⚠️ **只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式**。
- ⚠️ **必须先命中优化点的「命中条件」，才能加载参考文档；未命中则跳过**。
- ⚠️ **一次优化迭代只能使用一个优化点**。可以有多个迭代轮次：
  - 第一轮：检查 `1->2->3->...`，命中优化点 X，应用并回到 1 继续
  - 第二轮：检查 `1->2->...`，命中优化点 Y，应用并回到 1 继续
  - 第三轮：检查 `1->2->...`，命中优化点 Z，应用并回到 1 继续
  - ...
  - 直到所有优化点都不命中，本轮主流程结束。
- ⚠️ **一次只能参考一个文档**；参考文档仅用于当前命中优化点的子流程，完成后立即返回主流程。

## 优化验证规则

**⚠️ 强制要求：在进行任何精度验证或性能验证之前，必须先执行 checklist 检查，确保所有代码规范都已满足。验证流程如下：**

1. **Checklist 检查**：加载 `references/checklist.md`，逐项检查代码是否满足所有规范要求
2. **不满足规范** -> 修改代码直到满足所有规范要求，然后重新执行 checklist 检查确认
3. **满足规范后** -> 调用 `run_verify` 执行精度验证和性能验证

- **成功**：优化后的性能不劣化（speedup ≥ 1.0），该优化结果作为下一次优化迭代的基线
- **失败**：优化后的性能劣化（speedup < 1.0），放弃本次优化结果，以优化前的代码作为下一次优化迭代的基线

## 最终步骤

在所有优化点检查完毕后，依次执行 Block Size Scaling 和 Kernel 分裂作为最终收尾优化。

### Block Size Scaling

在所有指令级优化策略应用完毕后，**必须加载** `references/block_size_scaling.md`，执行 Block Size Scaling 作为最终优化步骤。

触发条件：
1. Kernel 中存在单维度 `BLOCK_SIZE` 参数
2. 该参数可静态解析其数值
3. Kernel 不含多维分块参数
4. BLOCK_SIZE 未由 `triton.autotune` 的 configs 管理

从当前值开始每次 x2，上限 65536（Triton Ascend 单块元素数硬限制），verify 失败即到达硬件上限停止搜索。

### Kernel 分裂优化

完成 Block Size Scaling 后，若任务满足多 Case 且泛用 Kernel 性能仍未达标（`total_cases > 1` 且 `speedup_vs_torch < 0.8`），**必须加载** `references/optimization-points/18-kernel-splitting.md`，执行 Kernel 分裂优化。

## 关键约束

- **必须产出代码文件**：即使无优化点，也要写出输出文件
- **功能一致性**：优化前后的计算结果必须一致
- **精度一致性**：数值精度不能降低
- **可执行性**：输出代码必须能被 run_verify 直接验证和基准测试
- **一次一个优化点**：每次只应用一个优化点，验证通过后再进入下一轮
