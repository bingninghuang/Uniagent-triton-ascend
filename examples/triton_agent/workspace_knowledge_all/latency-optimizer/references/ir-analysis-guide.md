# IR/msprof 性能分析指南

> Ascend910B3 专用。本文件整合 IR 分析与 msprof profiling 两条诊断路径，帮助快速定位性能瓶颈并映射到优化点。

## 硬件规格 (Ascend910B3)

| 参数 | 值 |
|------|-----|
| AI Core | 20 个 |
| Vector Core | 40 个 |
| Cube Core | 20 个 |
| UB 容量 | 192 KB |
| 对齐要求 | 256B |

## 一、collect_ir 工具使用

### Tier 1: 快速诊断（读取摘要）

调用 `collect_ir` 工具收集算子 IR。工具会自动提取并生成 IR 摘要，包含瓶颈标识：

1. 工具提取编译器最终阶段 IR（`last_pass.mlir`）
2. 返回摘要包含：标量操作密度、访存模式、流水结构、同步密度等关键指标
3. 根据摘要中的瓶颈标识定位优化方向

### Tier 2: 深度分析（读取 IR 文件）

当摘要不足以定位时，读取 collect_ir 输出的 IR 文件进行深度分析：

| IR 文件 | 描述 | 用途 |
|---------|------|------|
| `<kernel>_ttir.mlir` | Triton 前端 IR | 查看原始 Triton 操作映射 |
| `<kernel>_ttadapter.mlir` | 适配器 IR | 查看设备适配后的操作 |
| `<kernel>_last_pass.mlir` | BishengIR 最终阶段 IR | **核心分析文件**，查看最终指令级结构 |

### IR 分析要点

检查 `last_pass.mlir` 中的以下模式：

1. **标量密集模式**：过多 `arith` 操作（int64 比较、类型转换），且未被流水重叠掩盖
2. **访存模式**：非连续 load/store、冗余拷贝、不对齐 tiling
3. **流水结构**：Vector 和 Cube 操作是否正确重叠；标量操作是否在关键路径上
4. **同步/屏障密度**：独立操作之间是否有不必要的 barrier

> **重要**：优先进行 IR 分析，msprof 指标作为补充验证。高 scalar_ratio 不一定意味着标量瓶颈——如果标量操作被流水重叠掩盖，则不是瓶颈。只有当 IR 确认标量操作在关键路径上时才需要处理。

## 二、collect_profiling 工具使用

### msprof 基本命令

```bash
# 采集指定 kernel 的性能数据
msprof op --kernel-name=target_kernel_name --output=$HOME/projects/output python3 test_op.py

# 采集所有算子的性能数据
msprof op --output=$HOME/projects/output python3 test_op.py
```

### op_summary 关键指标

| 指标 | 含义 | 理想值 | 瓶颈信号 |
|------|------|--------|----------|
| aiv_vec_time(us) | Vector 流水执行时间 | - | - |
| aiv_vec_ratio | Vector 流水利用率 | > 80% | < 30% 说明 Vector 断流 |
| aiv_mte2_time(us) | MTE2 搬入时间 | - | - |
| aiv_mte2_ratio | MTE2 搬入占比 | < 50% | > 50% 说明搬运瓶颈 |
| aiv_mte3_time(us) | MTE3 搬出时间 | - | - |
| aiv_scalar_time(us) | Scalar 执行时间 | - | - |
| aiv_scalar_ratio | Scalar 流水利用率 | < 20% | > 30% 说明标量退化（需 IR 确认） |
| aic_cube_time(us) | Cube 流水执行时间 | - | - |
| aic_cube_ratio | Cube 流水利用率 | > 80% | < 30% 说明 Cube 断流 |

### 诊断规则

```
1. aiv_vec_ratio < 10%  -> Vector 未充分发挥算力
2. aiv_scalar_ratio > 30% -> 存在标量退化（需 IR 确认是否在关键路径上）
3. aiv_mte2_ratio > 50%  -> 搬运瓶颈
4. aic_cube_ratio < 30%  -> Cube 断流
5. Block Dim > 物理核数(40 VEC / 20 CUBE) -> Host 调度开销过大
6. Block Dim 远小于物理核数 -> 核心利用率不足
```

### 理论性能计算

```python
# 搬运理论耗时 = 搬运数据量(Byte) / 理论带宽
data_size = 4 * 4096 * 4096  # sizeof(float) * 4096 * 4096
bandwidth = 1.8e12  # 1.8 TB/s (GM 峰值带宽)
latency = data_size / bandwidth

# 计算理论耗时 = 计算数据量(Element) / 理论算力
compute_size = 32 * 1024  # 32K elements
peak_flops = 11.06e12  # 11.06 TOPS (Vector 理论峰值)
latency = compute_size / peak_flops

# 对比：实际性能达理论 70% 以上通常认为接近最优
# 理论性能 = max(搬运理论耗时, 计算理论耗时)
```

## 三、瓶颈到优化点映射

| 瓶颈类型 | msprof 特征 | IR 特征 | 对应优化点 |
|----------|-------------|---------|-----------|
| 计算密集 | aiv_vec_ratio 或 aic_cube_ratio 高 | 计算指令密集 | 2, 7, 8, 13, 15 |
| 访存密集 | aiv_mte2_ratio > 50% | 非连续 load/store | 2, 4, 10, 11, 16, 17 |
| 标量退化 | aiv_scalar_ratio > 30%（IR 确认关键路径） | arith 操作密集 | 1, 5, 6, 9 |
| 流水隐藏标量 | aiv_scalar_ratio 高但 IR 显示重叠 | 标量在 MTE2 等待间隙执行 | 无需操作 |
| 同步开销 | 流水图中有大量等待 | barrier 密集 | 19, 21 |
| 并行度不足 | Block Dim 与核数偏离 | grid 设置不合理 | 3, 12, 14, 18 |
| 流水冲突 | MTE3 阻塞 Cube | Cube/MTE3 交替 | 19, 20, 22 |

## 四、常见性能反模式

### 反模式1：int64/i32 的 Compare 退化为标量

**问题**：i64/i32 的比较在 NPU 上无法启用 Vector，退化为标量计算。

**诊断**：aiv_scalar_ratio 异常高，IR 中 SCALAR 指令饱和。

**解决**：将比较操作数转换为 fp32。对应优化点 5/6。

```python
# 优化前
cols = tl.arange(0, BLOCK_N)  # int64
xbar = tl.where(cols < N, x - mean, 0.0)  # 退化为标量

# 优化后
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)  # 向量化
```

### 反模式2：Grid 分核数过多

**问题**：Grid 远超物理核数（40 VEC / 20 CUBE），Host 调度开销大。

**诊断**：op_summary 中 Block Dim 远大于物理核数。

**解决**：固定核数为物理核数，核内循环处理。对应优化点 3。

### 反模式3：Tiling 过小导致搬运冗余

**问题**：BLOCK_SIZE 过小，大量冗余搬运指令，MTE2 利用率低。

**诊断**：aiv_mte2_time 远大于理论搬运时间，MTE2 流水断流。

**解决**：增大 BLOCK_SIZE 或使用 autotune。对应优化点 2/13。

### 反模式4：无 for 循环导致无法存算并行

**问题**：算子无 Tiling 切分，单次执行完成，multiBuffer 无法使能。

**诊断**：IR 中 MTE2 和 Vector 完全串行，无重叠。

**解决**：添加 for 循环实现 Tiling。对应优化点 2。

### 反模式5：尾轴不对齐导致自动补齐

**问题**：Tensor 尾轴大小不满足 256B 对齐要求，硬件自动补齐浪费空间和带宽。

**诊断**：UB 使用量异常大于预期，性能随 shape 变化波动大。

**解决**：使用借轴转置或 1D load 规避自动补齐。对应优化点 2/8。

### 反模式6：care_padding 导致 MTE2-Vector 同步

**问题**：care_padding=True（默认）时，MTE2 必须等待 Vector 初始化完成，降低并行度。

**诊断**：IR 中 MTE2 和 Vector 存在串行依赖。

**解决**：确认 padding 区域不影响结果后，设置 care_padding=False。

## 五、IR 分析决策流程

1. 调用 `collect_ir` 收集 IR
2. 读取 `last_pass.mlir`，按上述 IR 分析要点检查
3. 若需验证，调用 `collect_profiling` 收集 msprof 数据
4. 按瓶颈到优化点映射表选择优化方向
5. 读取对应优化点参考文档
6. 应用优化 -> 代码规范检查 -> run_verify 确认
7. 重新提取 IR 确认优化效果，检查是否有新瓶颈

> **IR 多轮迭代**：优化点 25（IR 分析）支持多轮重复命中。每轮重新提取 IR，聚焦：(a) 上一轮优化是否带来预期效果；(b) 是否还有新的优化建议。当 IR 分析无新建议时退出迭代。
