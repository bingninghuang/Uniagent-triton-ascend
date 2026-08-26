# 线性代数操作 (Linear Algebra Operations)

## tl.dot

对两个 tensor 进行矩阵乘操作。tensor 需要是二维或三维并且维度需一致。对于三维块，tl.dot 执行批量矩阵乘法，其中每个块的第一维代表批量维度。

```python
triton.language.dot(input, other, acc=None, input_precision=None,
                    allow_tf32=None, max_num_imprecise_acc=None,
                    out_dtype=triton.language.float32, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 第一个输入，2D 或 3D 张量，取值范围建议 [-5, 5] 以避免溢出 |
| `other` | `tensor` | 第二个输入，2D 或 3D 张量，取值范围建议 [-5, 5] |
| `acc` | `tensor` | 累加结果张量。如果不为 None，结果会加到该张量上。支持 float16/float32/int32 |
| `input_precision` | - | NVIDIA 精度模式选项，用于决定是否启用 Tensor Cores 加速 |
| `max_num_imprecise_acc` | `int` | 低精度累加次数（Ascend 不支持低精度累加） |
| `out_dtype` | `tl.dtype` | 输出结果类型，支持 fp32, int32 |

返回值：`tensor`，矩阵乘结果。

**DataType 支持 (Ascend)**：int8, int16, int32, fp16, fp32, bf16, bool。
**Ascend 限制**：
- 不支持 uint8/uint16/uint32/uint64, int64, fp64。
- acc 不能为 fp16，硬件默认使用 fp32 以保证精度。
- `max_num_imprecise_acc` 暂不支持。
- `out_dtype` 不支持 int8 和 fp16 类型。
- `input_precision` 和 `allow_tf32` 对 Ascend 硬件无效。

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  acc_dtype: tl.constexpr,
                  stride_am: tl.constexpr, stride_ak: tl.constexpr,
                  stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                  stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)
```

---

## tl.dot_scaled

计算以缩放格式表示的两个矩阵块的矩阵乘积。

```python
triton.language.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
                            acc=None, lhs_k_pack=True, rhs_k_pack=True,
                            out_dtype=triton.language.float32, _semantic=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `lhs` | `tensor` | 左矩阵张量（支持 bf16、fp16 格式） |
| `lhs_scale` | `tensor` | 左矩阵缩放张量（支持 int8 格式） |
| `lhs_format` | `str` | 左矩阵张量的存放格式（支持 "bf16" 和 "fp16"） |
| `rhs` | `tensor` | 右矩阵张量（支持 bf16、fp16 格式） |
| `rhs_scale` | `tensor` | 右矩阵缩放张量（支持 int8 格式） |
| `rhs_format` | `str` | 右矩阵张量的存放格式（支持 "bf16" 和 "fp16"） |
| `acc` | `tensor` | 累积张量 |
| `lhs_k_pack` | `bool` | true 沿 K 维度打包，false 沿 M 维度打包 |
| `rhs_k_pack` | `bool` | true 沿 K 维度打包，false 沿 N 维度打包 |

返回值：`tensor`，计算缩放矩阵乘后输出的值。

**DataType 支持 (Ascend)**：bf16, fp16。
**Ascend 限制**：
- 不支持 fp4、fp8 格式（硬件限制）。
- 缩放张量的值为 int8（GPU 上为 uint8）。
- 输入矩阵 lhs、rhs 推荐输入范围为 [-5, 5]，超过可能出现极值 inf。
- 由于硬件对齐要求，scale 矩阵做 broadcast 的倍数至少应为 16。
- Shape 支持 2~3 维 tensor，但 scale 矩阵只支持 2 维。

```python
@triton.jit
def dot_scale_kernel(a_base, stride_a0: tl.constexpr, stride_a1: tl.constexpr,
                     a_scale, b_base, stride_b0: tl.constexpr, stride_b1: tl.constexpr,
                     b_scale, out,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     type_a: tl.constexpr, type_b: tl.constexpr):
    a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0, stride_a1)[None, :] * stride_a1
    b_ptr = b_base + tl.arange(0, BLOCK_K)[:, None] * stride_b0 + tl.arange(0, BLOCK_N)[None, :] * stride_b1
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if a_scale is not None:
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(scale_a_ptr)
    if b_scale is not None:
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        b_scale = tl.load(scale_b_ptr)
    accumulator = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b,
                                 acc=accumulator, out_dtype=tl.float32)
    out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr, accumulator.to(a.dtype))
```

---

## Ascend 通用限制总结

- **dot**：acc 不能为 fp16（默认 fp32）；不支持 max_num_imprecise_acc；out_dtype 不支持 int8/fp16。
- **dot_scaled**：不支持 fp4/fp8 格式；缩放张量为 int8（GPU 为 uint8）；推荐输入范围 [-5, 5]；scale broadcast 倍数至少 16。
- **Shape**：均支持 2~3 维 tensor（dot_scaled 的 scale 矩阵仅支持 2D）。
