---
name: api-reference
description: Triton Ascend API 参考手册，提供全部 tl.* 操作的函数签名、参数说明、数据类型支持表、Ascend 限制和使用示例。在编写 Triton-Ascend 算子时用于查阅 API 细节。
argument-hint: >
  默认查看目录和各章节首句；--section "数学运算" 读取特定 API 类别；
  --file references/<category>.md 读取对应 API 参考；--full 读取全部。
---

# Triton Ascend API 参考手册

本技能提供 Triton Language (`tl.*`) 在 Ascend 平台上的完整 API 参考。所有 API 按功能分类，每个类别对应一个参考文件。

## 使用方法

使用 `read_skill --file references/<category>.md` 读取特定类别的 API 参考。例如：

- `read_skill --file references/math-ops.md` - 查阅数学运算 API
- `read_skill --file references/memory-ops.md` - 查阅内存/指针操作 API
- `read_skill --file references/reduction-ops.md` - 查阅归约操作 API

## API 索引

总览：17 类 tl.* 操作，每类对应 references/ 下一个参考文件，下表首行列出该类包含的接口名及 Ascend 关键限制。

### 数学运算 (Math Operations)
tl.abs / tl.add / tl.sub / tl.mul / tl.div / tl.floordiv / tl.mod / tl.neg / tl.ceil / tl.floor / tl.sqrt / tl.sqrt_rn 等 30 个逐元素数学运算；不支持 fp64/uint16-64，softmax 不支持 int，fma acc 不支持 fp16。文件: references/math-ops.md

| API | 说明 | 运算符 |
|-----|------|--------|
| `tl.abs` | 绝对值 | |
| `tl.add` | 加法 | `+` |
| `tl.sub` | 减法 | `-` |
| `tl.mul` | 乘法 | `*` |
| `tl.div` | 除法 | `/` |
| `tl.floordiv` | 整除 | `//` |
| `tl.mod` | 取模 | `%` |
| `tl.neg` | 取负 | `-x` |
| `tl.ceil` | 向上取整 | |
| `tl.floor` | 向下取整 | |
| `tl.sqrt` | 平方根 | |
| `tl.sqrt_rn` | 平方根（舍入到最近偶数） | |
| `tl.rsqrt` | 平方根倒数 | |
| `tl.exp` | 自然指数 | |
| `tl.exp2` | 2的指数 | |
| `tl.log` | 自然对数 | |
| `tl.log2` | 以2为底的对数 | |
| `tl.sin` | 正弦 | |
| `tl.cos` | 余弦 | |
| `tl.sigmoid` | Sigmoid 函数 | |
| `tl.erf` | 误差函数 | |
| `tl.softmax` | Softmax | |
| `tl.fma` | 融合乘加 (a*b+c) | |
| `tl.fdiv` | 快速除法 | |
| `tl.div_rn` | 舍入到最近偶数的除法 | |
| `tl.clamp` | 数值截断 | |
| `tl.maximum` | 元素级最大值 | |
| `tl.minimum` | 元素级最小值 | |
| `tl.cdiv` | 向上整除 | |
| `tl.umulhi` | 无符号高半乘法 | |

**Ascend 关键限制**: 不支持 fp64; 不支持 uint16/uint32/uint64; softmax 不支持 int 类型; fma 的 acc 不支持 fp16。

---

### 内存/指针操作 (Memory & Pointer Operations)
tl.load / tl.store / tl.make_block_ptr / tl.advance / tl.make_tensor_descriptor / desc.load / desc.store 等 pointer 与 block pointer 操作；cache_modifier/padding_option 无效，tensor_descriptor 不能与 load/store 混用。文件: references/memory-ops.md

| API | 说明 |
|-----|------|
| `tl.load` | 从全局内存加载张量数据 |
| `tl.store` | 将数据存储到全局内存 |
| `tl.make_block_ptr` | 创建指向 GM 上张量的块指针 |
| `tl.advance` | 更新 make_block_ptr 的 offset |
| `tl.make_tensor_descriptor` | 创建张量描述符 (Triton 3.4.0+) |
| `desc.load` / `tl.load_tensor_descriptor` | 从张量描述符加载数据块 |
| `desc.store` / `tl.store_tensor_descriptor` | 将数据块存储到张量描述符 |

**Ascend 关键限制**: cache_modifier/eviction_policy/volatile 对 Ascend 无效; padding_option 不支持; tensor_descriptor 操作不能与 tl.load/store 混用。

---

### 归约操作 (Reduction Operations)
tl.sum / tl.max / tl.min / tl.argmax / tl.argmin / tl.reduce / tl.xor_sum 等 7 个轴归约操作；不支持 fp64/uint16-64，axis=None+return_indices 不支持，sum dtype 参数暂不支持。文件: references/reduction-ops.md

| API | 说明 |
|-----|------|
| `tl.sum` | 沿指定轴求和 |
| `tl.max` | 沿指定轴取最大值，支持返回索引 |
| `tl.min` | 沿指定轴取最小值，支持返回索引 |
| `tl.argmax` | 返回最大值所在下标 |
| `tl.argmin` | 返回最小值所在下标 |
| `tl.reduce` | 使用自定义 combine_fn 进行规约 |
| `tl.xor_sum` | 沿指定轴计算异或和 |

**Ascend 关键限制**: 不支持 fp64, uint16/uint32/uint64; axis=None 且 return_indices=True 时不支持; sum 的 dtype 参数暂不支持。

---

### 扫描/排序操作 (Scan & Sort Operations)
tl.associative_scan / tl.cumsum / tl.cumprod / tl.histogram / tl.sort / tl.topk 等 6 个扫描与排序操作；sort/topk 不支持 int32/uint8/int64/fp64/bool，topk 要求 k 为 2 的幂。文件: references/scan-sort-ops.md

| API | 说明 |
|-----|------|
| `tl.associative_scan` | 使用自定义 combine_fn 进行关联扫描 |
| `tl.cumsum` | 累积和（前缀和） |
| `tl.cumprod` | 累积乘积（前缀乘积） |
| `tl.histogram` | 计算直方图 |
| `tl.sort` | 沿维度排序 |
| `tl.topk` | 返回前 k 个最大元素 |

**Ascend 关键限制**: sort/topk 不支持 int32/uint8/int64/fp64/bool（编译器限制）; topk 要求 k 为2的幂，仅支持最后一个维度; histogram 仅支持一维。

---

### 原子操作 (Atomic Operations)
tl.atomic_add / tl.atomic_and / tl.atomic_or / tl.atomic_xor / tl.atomic_xchg / tl.atomic_max / tl.atomic_min / tl.atomic_cas 等 8 个原子操作；sem 仅支持 acq_rel，scope 仅支持 gpu，不支持 fp64。文件: references/atomic-ops.md

| API | 说明 |
|-----|------|
| `tl.atomic_add` | 原子加法 |
| `tl.atomic_and` | 原子逻辑与 |
| `tl.atomic_or` | 原子逻辑或 |
| `tl.atomic_xor` | 原子逻辑异或 |
| `tl.atomic_xchg` | 原子交换 |
| `tl.atomic_max` | 原子取最大值 |
| `tl.atomic_min` | 原子取最小值 |
| `tl.atomic_cas` | 原子比较和交换 |

**Ascend 关键限制**: sem 仅支持 "acq_rel"; scope 仅支持 "gpu"; 所有原子操作不支持 fp64。

---

### 创建操作 (Creation Operations)
tl.arange / tl.zeros / tl.zeros_like / tl.full / tl.cast / tl.cat 等 6 个张量创建操作；arange 上限 1048576，cat 仅 1D+can_reorder，cast 扩展 overflow_mode=saturate。文件: references/creation-ops.md

| API | 说明 |
|-----|------|
| `tl.arange` | 生成连续整数序列 |
| `tl.zeros` | 创建零张量 |
| `tl.zeros_like` | 创建与输入相同形状的零张量 |
| `tl.full` | 创建填充指定值的张量 |
| `tl.cast` | 类型转换（支持 bitcast 和 overflow_mode） |
| `tl.cat` | 拼接两个张量 |

**Ascend 关键限制**: arange 最大 1048576 元素; cat 仅支持 1D 和 can_reorder=True; cast 有 Ascend 扩展 overflow_mode="saturate"。

---

### 形状操作 (Shape Manipulation Operations)
tl.broadcast / tl.broadcast_to / tl.expand_dims / tl.interleave / tl.join / tl.permute / tl.ravel / tl.reshape / tl.split / tl.trans / tl.view 等 11 个形状操作；reshape can_reorder 仅 False，permute/trans 不支持维度高于8。文件: references/shape-manipulation-ops.md

| API | 说明 |
|-----|------|
| `tl.broadcast` | 广播两个张量到共同形状 |
| `tl.broadcast_to` | 广播张量到目标形状 |
| `tl.expand_dims` | 在指定轴插入大小为1的维度 |
| `tl.interleave` | 在最后一个维度上交错排列 |
| `tl.join` | 沿新维度连接两个张量 |
| `tl.permute` | 重新排列维度顺序 |
| `tl.ravel` | 展平为一维张量 |
| `tl.reshape` | 改变张量形状 |
| `tl.split` | 沿最后一个维度分割 |
| `tl.trans` | 转置维度（优化版本） |
| `tl.view` | 创建张量视图 |

**Ascend 关键限制**: reshape 的 can_reorder 仅支持 False; permute/trans 不支持维度高于8的转置。

### 索引操作 (Indexing Operations)
tl.flip / tl.gather / tl.swizzle2d / tl.where 等 4 个索引操作；gather 仅 fp16/fp32/bf16，swizzle2d 仅 int32/int64+2D，支持 1~5 维。文件: references/indexing-ops.md

| API | 说明 |
|-----|------|
| `tl.flip` | 沿指定维度翻转 |
| `tl.gather` | 沿指定维度按索引收集 |
| `tl.swizzle2d` | 行优先索引转列优先索引 |
| `tl.where` | 条件选择 |

**Ascend 关键限制**: gather 仅支持 fp16/fp32/bf16; swizzle2d 仅支持 int32/int64 和 2D; 所有操作支持 1~5 维。

---

### 比较操作 (Comparing Operations)
tl.eq / tl.ne / tl.gt / tl.ge / tl.lt / tl.le 等 6 个比较操作；不支持 fp64/uint16-64，eq/ne/ge/lt/le 不支持 uint8。文件: references/comparing-ops.md

| API | 说明 | 运算符 |
|-----|------|--------|
| `tl.eq` | 相等比较 | `==` |
| `tl.ne` | 不等比较 | `!=` |
| `tl.gt` | 大于 | `>` |
| `tl.ge` | 大于等于 | `>=` |
| `tl.lt` | 小于 | `<` |
| `tl.le` | 小于等于 | `<=` |

**Ascend 关键限制**: 不支持 fp64; eq/ne/ge/lt/le 不支持 uint8; 不支持 uint16/uint32/uint64。

---

### 逻辑操作 (Logical Operations)
tl.and / tl.or / tl.xor / tl.not / tl.invert / tl.logical_and / tl.logical_or / tl.lshift / tl.rshift / tl.neg 等 10 个逻辑位操作；logical_and/or 支持整型+浮点型，lshift/rshift 右操作数仅标量。文件: references/logical-ops.md

| API | 说明 | 运算符 |
|-----|------|--------|
| `tl.and` | 按位与 | `&` |
| `tl.or` | 按位或 | `\|` |
| `tl.xor` | 按位异或 | `^` |
| `tl.not` | 逻辑非 | `not` |
| `tl.invert` | 按位取反 | `~` |
| `tl.logical_and` | 逻辑与 | |
| `tl.logical_or` | 逻辑或 | |
| `tl.lshift` | 左移位 | `<<` |
| `tl.rshift` | 右移位 | `>>` |
| `tl.neg` | 取负 | `-` |

**Ascend 关键限制**: logical_and/logical_or 在 Ascend 上支持所有整型和浮点型（GPU 仅支持 bool）; lshift/rshift 右操作数仅支持标量。

---

### 线性代数操作 (Linear Algebra Operations)
tl.dot / tl.dot_scaled 等 2 个矩阵乘操作；dot acc 不支持 fp16，out_dtype 不支持 int8/fp16，dot_scaled 不支持 fp4/fp8。文件: references/linear-algebra-ops.md

| API | 说明 |
|-----|------|
| `tl.dot` | 矩阵乘，支持2D/3D |
| `tl.dot_scaled` | 缩放矩阵乘 |

**Ascend 关键限制**: dot 的 acc 不能为 fp16（默认 fp32）; out_dtype 不支持 int8/fp16; dot_scaled 不支持 fp4/fp8 格式; 输入推荐范围 [-5, 5]; scale broadcast 倍数至少 16。

---

### 编译器提示操作 (Compiler Hint Operations)
tl.assume / tl.debug_barrier / tl.max_constancy / tl.max_contiguous / tl.multiple_of 等 5 个编译器提示操作；不支持 uint/fp64，values 维度须与 input 相同。文件: references/compiler-hint-ops.md

| API | 说明 |
|-----|------|
| `tl.assume` | 向编译器提供条件假设 |
| `tl.debug_barrier` | 插入调试屏障 |
| `tl.max_constancy` | 声明值的常量性模式 |
| `tl.max_contiguous` | 声明连续性模式 |
| `tl.multiple_of` | 声明值是某数的倍数 |

**Ascend 关键限制**: max_constancy/max_contiguous/multiple_of 不支持 uint/fp64; values 维度必须与 input 维度相同。

---

### 调试操作 (Debug Operations)
tl.device_assert / tl.device_print / tl.static_assert / tl.static_print 等 4 个调试操作；不支持 uint/fp64，static_assert 须 constexpr，device_print prefix 必填。文件: references/debug-ops.md

| API | 说明 | 环境变量 | 执行时机 |
|-----|------|---------|---------|
| `tl.device_assert` | 运行时断言 | `TRITON_DEBUG` 非0 | 运行时 |
| `tl.device_print` | 运行时打印 | `TRITON_DEVICE_PRINT=True` | 运行时 |
| `tl.static_assert` | 编译时断言 | 无 | 编译时 |
| `tl.static_print` | 编译时打印 | 无 | 编译时 |

**Ascend 关键限制**: device_print/static_print 不支持 uint/fp64; static_assert 条件必须为 constexpr; device_print 的 prefix 必填。

---

### 扩展操作 (Extension Operations)
tl.compile_hint / tl.extract_slice / tl.insert_slice / tl.get_element / tl.multibuffer / tl.parallel / tl.sync_block_* / tl.index_select_simd 等 10 个扩展操作；multibuffer 仅 size=2，sync_block event_id 0-15，index_select_simd 不支持尾轴。文件: references/extension-ops.md

| API | 说明 |
|-----|------|
| `tl.compile_hint` | 为张量附加元数据提示 |
| `tl.extract_slice` | 提取张量切片 |
| `tl.insert_slice` | 插入子张量到目标张量 |
| `tl.get_element` | 读取单个元素 |
| `tl.multibuffer` | 设置多缓冲 |
| `tl.parallel` | 多核心并行迭代器 |
| `tl.sync_block_set` | Cube-Vector 核心间同步信号发送 |
| `tl.sync_block_wait` | Cube-Vector 核心间同步信号等待 |
| `tl.sync_block_all` | Cube-Vector 全局屏障同步 |
| `tl.index_select_simd` | Ascend 专用高效索引选择 |

**Ascend 关键限制**: multibuffer 仅支持 size=2; sync_block event_id 范围 0-15; index_select_simd 不支持尾轴。

---

### 随机数生成 (Random Number Generation)
tl.rand / tl.randint / tl.randn / tl.randint4x 等 4 个随机数生成操作；seed 仅支持 int/bool 不支持浮点，randint4x 最高效。文件: references/random-ops.md

| API | 说明 | 返回类型 | 分布 |
|-----|------|---------|------|
| `tl.rand` | 均匀分布随机数 | float32 | U(0, 1) |
| `tl.randint` | 整数随机数 | int32 | 均匀分布 |
| `tl.randn` | 正态分布随机数 | float32 | N(0, 1) |
| `tl.randint4x` | 4个整数随机块 | 4x int32 | 均匀分布 |

**Ascend 关键限制**: seed 支持 int 类型和 bool，不支持浮点类型; randint4x 是最高效入口点。

---

### 迭代器 (Iterators)
tl.range / tl.static_range / tl.parallel 等 3 个循环迭代器；range flatten/warp_specialize/disable_licm 不完全，仅支持整型索引。文件: references/iterators.md

| API | 说明 | 特点 |
|-----|------|------|
| `tl.range` | 通用循环迭代器 | 支持 num_stages, loop_unroll_factor |
| `tl.static_range` | 编译时循环展开 | 所有参数必须为 constexpr |
| `tl.parallel` | 多核心并行迭代器 | 支持 bind_sub_block |

**Ascend 关键限制**: range 的 flatten/warp_specialize/disable_licm 功能不完全; 仅支持整型索引。

---

### 内联汇编 (Inline Assembly)
tl.inline_asm_elementwise 1 个内联汇编操作；寄存器仅 int64/float32，约束仅 l，仅支持 1D 张量输入。文件: references/inline-assembly.md

| API | 说明 |
|-----|------|
| `tl.inline_asm_elementwise` | 执行内联汇编代码，逐元素操作 |

**Ascend 关键限制**: 寄存器仅支持 int64(s64) 和 float32(f32); 约束仅支持 `l`; 仅支持1D张量输入。

---

## Ascend 通用 DataType 限制速查

以下类型在大部分 Ascend 操作中**不支持**（具体支持情况请查阅对应参考文件）：

| 类型 | 限制说明 |
|------|---------|
| fp64 | 所有操作均不支持（硬件限制） |
| uint16/uint32/uint64 | 大部分操作不支持（硬件限制） |
| uint8 | 部分操作支持，多数不支持 |
| fp8e4/fp8e5 | 所有操作不支持（硬件限制） |

## 常用 Ascend 特有扩展

Ascend 平台特有扩展接口：tl.cast overflow_mode="saturate" / tl.index_select_simd / tl.sync_block_* / tl.parallel bind_sub_block 等，详见下表。

| 扩展 | 说明 |
|------|------|
| `tl.cast` 的 `overflow_mode="saturate"` | 整数转换溢出处理，饱和模式 |
| `tl.index_select_simd` | Ascend 专用高效索引选择 |
| `tl.sync_block_*` | Cube-Vector 架构核心间同步 |
| `tl.parallel` 的 `bind_sub_block` | 多核心并行执行 |
