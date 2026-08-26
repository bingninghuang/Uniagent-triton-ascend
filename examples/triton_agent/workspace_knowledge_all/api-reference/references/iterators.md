# 迭代器 (Iterators)

## tl.range

向上计数的迭代器，类似于 Python 的 `range()` 函数，但允许传入更多编译优化参数。

```python
triton.language.range(arg1, arg2=None, step=None, num_stages=None,
                      loop_unroll_factor=None, disallow_acc_multi_buffer=False,
                      flatten=False, warp_specialize=False, disable_licm=False,
                      _semantic=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `arg1` | `int`/`constexpr` | 必需 | 起始值（单参数时为结束值，从0开始） |
| `arg2` | `int`/`constexpr` | - | 结束值（不包含） |
| `step` | `int`/`constexpr` | `1` | 每次迭代的步长 |
| `num_stages` | `int` | - | 流水线阶段数（同时执行的迭代数量） |
| `loop_unroll_factor` | `int` | - | 循环展开因子（<2表示不展开） |
| `disallow_acc_multi_buffer` | `bool` | `False` | 禁止 dot 操作累加器的多缓冲优化 |
| `flatten` | `bool` | `False` | 自动展平嵌套循环为单层循环 |
| `warp_specialize` | `bool` | `False` | 启用 warp 专业化（仅 Blackwell GPU） |
| `disable_licm` | `bool` | `False` | 禁用循环不变代码外提优化 |

**DataType 支持 (Ascend)**：int8, int16, int32, int64。
**Ascend 限制**：
- 不支持 uint8/uint16/uint32/uint64, fp64。
- `disallow_acc_multi_buffer`, `flatten`, `warp_specialize`, `disable_licm` 相关功能在 Ascend 上还不完全支持。

```python
@triton.jit
def basic_examples():
    # 单参数：0到9
    for i in tl.range(10):
        pass
    # 双参数：2到9
    for i in tl.range(2, 10):
        pass
    # 三参数：0到10，步长为2
    for i in tl.range(0, 10, 2):
        pass

@triton.jit
def advanced_examples():
    # 使用循环优化参数
    for i in tl.range(0, 100, num_stages=3, loop_unroll_factor=4):
        pass
    # 嵌套循环展平
    for i in tl.range(0, 10, flatten=True):
        for j in tl.range(0, 20, flatten=True):
            pass
```

---

## tl.static_range

静态范围的迭代器，与 `range` 类似但会在编译时进行积极的循环展开优化。适用于已知且较小的循环次数场景。

```python
triton.language.static_range(arg1, arg2=None, step=None, _semantic=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `arg1` | `constexpr` | 必需 | 起始值（单参数时为结束值，从0开始） |
| `arg2` | `constexpr` | - | 结束值（不包含） |
| `step` | `constexpr` | `1` | 每次迭代的步长 |

**DataType 支持 (Ascend)**：int8, int16, int32, int64。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

**与 `range` 的区别**：`static_range` 通过牺牲代码大小来换取运行时性能，整个循环在编译时展开，无循环控制开销。所有参数必须为 `constexpr`。

```python
@triton.jit
def optimized_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # 使用 static_range 进行小规模循环展开，消除循环开销
    for i in tl.static_range(BLOCK_SIZE):
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)

    # 对比：使用 range 会有循环控制开销
    for i in tl.range(BLOCK_SIZE):
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)
```

---

## Ascend 通用限制总结

| 迭代器 | 用途 | 参数限制 | Ascend 功能完整性 |
|--------|------|---------|------------------|
| range | 通用循环迭代 | 支持 num_stages, loop_unroll_factor | flatten/warp_specialize/disable_licm 功能不完全 |
| static_range | 编译时循环展开 | 所有参数必须为 constexpr | 完全支持 |
| parallel | 多核心并行 | 支持 bind_sub_block | bind_sub_block 功能待验证 |

- **range vs static_range**：`range` 适用于通用场景，支持流水线和展开优化；`static_range` 适用于小规模已知循环，编译时完全展开。
- **DataType**：仅支持整型（int8, int16, int32, int64），不支持 uint 和浮点类型。
- **parallel**：继承自 `range`，移除了部分参数，增加了 `bind_sub_block` 参数用于多核心并行。
