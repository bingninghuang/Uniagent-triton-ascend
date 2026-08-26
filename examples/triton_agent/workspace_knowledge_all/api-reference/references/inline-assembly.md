# 内联汇编 (Inline Assembly)

## tl.inline_asm_elementwise

在 Triton 内核中执行内联汇编代码，实现对张量的逐元素操作。

```python
triton.language.inline_asm_elementwise(asm, constraints, args, dtype,
                                       is_pure, pack, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `asm` | `str` | 要执行的汇编代码，必须匹配目标平台的汇编格式 |
| `constraints` | `str` | LLVM 格式的汇编约束条件 |
| `args` | `tensor` | 输入张量，其值会传递给汇编块 |
| `dtype` | `dtype` 或 `Sequence[dtype]` | 返回张量的元素类型（可以是单个类型或类型元组） |
| `is_pure` | `bool` | 如果为 True，编译器假设汇编块没有副作用 |
| `pack` | `int` | 每次内联汇编调用处理的元素数量 |

返回值：`tensor`，汇编操作后的结果张量。

**DataType 支持 (Ascend)**：int8, int16, int32, int64, fp32。
**Ascend 限制**：
- 不支持 uint8/uint16/uint32/uint64, fp16, fp64, bf16, bool。
- 内联汇编的寄存器仅支持 `int64(s64)` 和 `float32(f32)`。
- 约束限制仅支持 `l`。
- 目前仅支持输入一维张量，计算高维张量需展开。

```python
import triton.language as tl

@triton.jit
def triton_asm_add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="""
        ADD.s64 $0, $1, $2
        """,
        constraints=(
            "=l,l,l"
        ),
        args=[x, y],
        dtype=tl.int64,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## Ascend 限制总结

| 限制项 | Ascend 支持 | 说明 |
|--------|-----------|------|
| 寄存器类型 | int64(s64), float32(f32) | 仅支持这两种寄存器类型 |
| 约束 | `l` | 仅支持 LLVM 约束 `l` |
| 张量维度 | 1D | 仅支持一维张量输入 |
| 数据类型 | int8, int16, int32, int64, fp32 | 不支持 uint/fp16/bf16/fp64/bool |

**与 GPU 的差异**：
- GPU 支持所有数据类型和更多寄存器类型。
- Ascend 仅支持 int64 和 float32 寄存器，约束仅支持 `l`。
- Ascend 不支持高维张量直接输入，需要先展平为一维。
