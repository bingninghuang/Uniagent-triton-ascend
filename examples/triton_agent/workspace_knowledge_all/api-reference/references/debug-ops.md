# 调试操作 (Debug Operations)

## tl.device_assert

在设备运行时进行断言检查，如果条件不满足则输出错误信息。

**使用前必须设置环境变量 `TRITON_DEBUG` 为非 0 值才能生效。**

```python
triton.language.device_assert(cond, msg='', _semantic=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cond` | `bool` | 必需 | 运行时需要断言的条件表达式 |
| `msg` | `str` | `''` | 断言失败时显示的错误消息 |

返回值：无返回值。

**DataType 支持**：仅 bool。

```python
@triton.jit
def basic_device_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tl.device_assert(pid >= 0, "Program ID must be non-negative")
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.device_assert(tl.min(x) >= 0, "All values must be non-negative")
```

---

## tl.device_print

在 NPU 运行时从设备端打印信息。与 `static_print` 不同，这是在内核执行时实时输出信息。

**使用前必须设置环境变量 `TRITON_DEVICE_PRINT` 为 `True`。**

```python
triton.language.device_print(prefix, *args, hex=False, _semantic=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prefix` | `str` | 必需 | 打印值之前的前缀字符串（必须提供） |
| `args` | `tensor`/`scalar` | 必需 | 要打印的值 |
| `hex` | `bool` | `False` | 是否以十六进制格式打印所有值 |

返回值：无返回值。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。
**Shape 限制**：仅支持 1~5 维 tensor。

**注意**：`prefix` 字符串前缀在使用 `device_print` 时是必须加上的，否则会导致编译错误。

```python
@triton.jit
def kernel(x_ptr):
    idx = tl.arange(0, 3)
    idy = tl.arange(0, 4)
    offset = idx[:, None] * 4 + idy[None, :]
    val = tl.load(x_ptr + offset)
    tl.device_print("val:", val)
```

---

## tl.static_assert

在编译时断言条件是否成立，如果条件不满足则编译失败。不需要设置调试环境变量。

```python
triton.language.static_assert(cond, msg='', _semantic=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cond` | `bool` | 必需 | 编译时需要断言的条件表达式 |
| `msg` | `str` | `''` | 断言失败时显示的错误消息 |

返回值：无返回值。

**注意**：`cond` 语句中值的类型必须为 `constexpr`。在 `static_assert` 的条件中出现非常量会编译错误。

```python
@triton.jit
def basic_static_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # 检查 BLOCK_SIZE 是否为2的幂次
    tl.static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)
    # 带自定义错误消息
    tl.static_assert(BLOCK_SIZE >= 64, "BLOCK_SIZE must be at least 64 for performance")
```

---

## tl.static_print

在编译时打印信息，类似于 Python 的 `print()` 函数，但在内核编译期间执行。

```python
triton.language.static_print(*values, sep: str = ' ', end: str = '\n',
                             file=None, flush=False, _semantic=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `values` | `tensor`/`scalar` | 必需 | 要打印的值，支持多个参数 |
| `sep` | `str` | `' '` | 值之间的分隔符 |
| `end` | `str` | `'\n'` | 打印结束时的后缀 |

返回值：无返回值。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp16, fp32, bf16, bool。
**Ascend 限制**：不支持 uint8/uint16/uint32/uint64, fp64。

**注意**：如果打印非常量结果，会打印 `数据类型[数据shape(标量为空)]` 的值。非常量不支持 fstring 打印方式。

```python
@triton.jit
def basic_static_print_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # 在编译时打印常量的值
    tl.static_print("BLOCK_SIZE =", BLOCK_SIZE)
    tl.static_print(BLOCK_SIZE)
    # 支持 fstring 打印方式（仅常量）
    tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")

    # 打印非常量
    idx = tl.arange(0, 4)
    val = tl.load(x_ptr + idx)
    tl.static_print("val:", val)  # 输出类似 val:int32[constexpr[4]]
```

---

## Ascend 通用限制总结

| 操作 | 环境变量要求 | 执行时机 | DataType 限制 |
|------|------------|---------|-------------|
| device_assert | `TRITON_DEBUG` 非0 | 运行时 | 仅 bool |
| device_print | `TRITON_DEVICE_PRINT=True` | 运行时 | 不支持 uint/fp64 |
| static_assert | 无 | 编译时 | cond 必须为 constexpr |
| static_print | 无 | 编译时 | 不支持 uint/fp64 |

- **device_assert**：需要 `TRITON_DEBUG` 环境变量。
- **device_print**：需要 `TRITON_DEVICE_PRINT=True` 环境变量，prefix 必填。
- **static_assert**：条件必须为编译时常量。
- **static_print**：非常量不支持 fstring 打印。
