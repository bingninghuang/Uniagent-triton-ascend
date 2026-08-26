# Libdevice 数学函数库

## 概述

Libdevice 是 Triton Ascend 提供的 SIMT 模式逐元素数学函数库，通过 `triton.language.extra.cann.libdevice` 导入。共包含 167 个函数，涵盖三角函数、指数对数、舍入运算、类型转换、位操作、特殊函数等。

### 导入方式

```python
import triton.language.extra.cann.libdevice as libdevice
```

### SIMT 编译模式

libdevice 函数默认仅在 SIMT 编译模式下使用，需通过 `force_simt_only=True` 启用：

```python
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import torch

@triton.jit
def triton_kernel(input, output, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        x = tl.load(input + (x0), None)
        y = libdevice.abs(x)
        tl.store(output + (x0), y, None)

# 启用 SIMT 编译
input = torch.randn((128, 4096), dtype=torch.int32).npu()
output = torch.zeros_like(input)
triton_kernel[512, 1, 1](input, output, 1024, 1024, force_simt_only=True)
```

---

## 函数分类索引

### 一、基本数学运算

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `abs` | `libdevice.abs(x)` | int32, float32 | int32, float32 | 绝对值 |
| `cbrt` | `libdevice.cbrt(x)` | float32 | float32 | 立方根 |
| `ceil` | `libdevice.ceil(x)` | float32 | float32 | 向上取整 |
| `exp` | `libdevice.exp(x)` | float32 | float32 | e^x |
| `exp2` | `libdevice.exp2(x)` | float32 | float32 | 2^x |
| `exp10` | `libdevice.exp10(x)` | float32 | float32 | 10^x |
| `expm1` | `libdevice.expm1(x)` | float32 | float32 | e^x - 1 |
| `floor` | `libdevice.floor(x)` | float32 | float32 | 向下取整 |
| `fmod` | `libdevice.fmod(x, y)` | float32, float32 | float32 | 浮点取模，结果与 x 同号 |
| `hypot` | `libdevice.hypot(x, y)` | float32, float32 | float32 | 欧几里得距离 sqrt(x^2+y^2) |
| `log` | `libdevice.log(x)` | float32 | float32 | 自然对数 |
| `log2` | `libdevice.log2(x)` | float32 | float32 | 以 2 为底对数 |
| `log10` | `libdevice.log10(x)` | float32 | float32 | 以 10 为底对数 |
| `log1p` | `libdevice.log1p(x)` | float32 | float32 | log(1+x) |
| `logb` | `libdevice.logb(x)` | float32 | float32 | 提取浮点数指数值 |
| `ilogb` | `libdevice.ilogb(x)` | float32 | float32 | 提取无偏指数 |
| `nearbyint` | `libdevice.nearbyint(x)` | float32 | float32 | 最近邻整数 |
| `pow` | `libdevice.pow(x, y)` | float32, float32 | float32 | x^y |
| `rcbrt` | `libdevice.rcbrt(x)` | float32 | float32 | 立方根倒数 |
| `reciprocal` | `libdevice.reciprocal(x)` | float32 | float32 | 1/x |
| `remainder` | `libdevice.remainder(x, y)` | float32, float32 | float32 | x - n*y (n 为 x/y 最近邻整数) |
| `rint` | `libdevice.rint(x)` | float32 | float32 | 最近偶数舍入到最近邻整数 |
| `round` | `libdevice.round(x)` | float32 | float32 | 最近偶数舍入到最近邻整数 |
| `rsqrt` | `libdevice.rsqrt(x)` | float32 | float32 | 平方根倒数 |
| `sqrt` | `libdevice.sqrt(x)` | float32 | float32 | 平方根 |
| `trunc` | `libdevice.trunc(x)` | float32 | float32 | 截断取整（向零舍入） |
| `cyl_bessel_i0` | `libdevice.cyl_bessel_i0(x)` | float32 | float32 | 修正零阶贝塞尔函数 |
| `cyl_bessel_i1` | `libdevice.cyl_bessel_i1(x)` | float32 | float32 | 修正一阶贝塞尔函数 |
| `fdim` | `libdevice.fdim(x, y)` | float32, float32 | float32 | 正差，x>y 返回 x-y，否则返回 0 |
| `gamma` | `libdevice.gamma(x)` | float32 | float32 | 伽马函数 |
| `lgamma` | `libdevice.lgamma(x)` | float32 | float32 | log|gamma(x)| |
| `tgamma` | `libdevice.tgamma(x)` | float32 | float32 | 伽马函数 |
| `ldexp` | `libdevice.ldexp(x, exp)` | float32, int32 | float32 | x * 2^exp |
| `scalbn` | `libdevice.scalbn(x, n)` | float32, int32 | float32 | x * 2^n |
| `nextafter` | `libdevice.nextafter(x, y)` | float32, float32 | float32 | 从 x 朝 y 的下一个可表示浮点数 |
| `saturatef` | `libdevice.saturatef(x)` | float32 | float32 | 限制在 [0.0, 1.0] |

### 二、三角函数

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `sin` | `libdevice.sin(x)` | float32 | float32 | 正弦（弧度） |
| `cos` | `libdevice.cos(x)` | float32 | float32 | 余弦（弧度） |
| `tan` | `libdevice.tan(x)` | float32 | float32 | 正切（弧度） |
| `asin` | `libdevice.asin(x)` | float32 | float32 | 反正弦，范围 [-pi/2, pi/2] |
| `acos` | `libdevice.acos(x)` | float32 | float32 | 反余弦，范围 [0, pi] |
| `atan` | `libdevice.atan(x)` | float32 | float32 | 反正切，范围 [-pi/2, pi/2] |
| `atan2` | `libdevice.atan2(x, y)` | float32, float32 | float32 | x/y 的反正切，范围 [-pi, pi] |
| `sinpi` | `libdevice.sinpi(x)` | float32 | float32 | sin(pi * x) |
| `cospi` | `libdevice.cospi(x)` | float32 | float32 | cos(pi * x) |

### 三、双曲函数

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `sinh` | `libdevice.sinh(x)` | float32 | float32 | 双曲正弦 |
| `cosh` | `libdevice.cosh(x)` | float32 | float32 | 双曲余弦 |
| `tanh` | `libdevice.tanh(x)` | float32 | float32 | 双曲正切 |
| `asinh` | `libdevice.asinh(x)` | float32 | float32 | 反双曲正弦 |
| `acosh` | `libdevice.acosh(x)` | float32 | float32 | 反双曲余弦，范围 [0, +inf] |
| `atanh` | `libdevice.atanh(x)` | float32 | float32 | 反双曲正切，范围 [-1, 1] |

### 四、舍入控制运算

舍入模式后缀：`_rd`（向下）、`_rn`（最近偶数）、`_ru`（向上）、`_rz`（向零）。

#### 加法

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `add_rd` | `libdevice.add_rd(x, y)` | float32, float32 | float32 |
| `add_rn` | `libdevice.add_rn(x, y)` | float32, float32 | float32 |
| `add_ru` | `libdevice.add_ru(x, y)` | float32, float32 | float32 |
| `add_rz` | `libdevice.add_rz(x, y)` | float32, float32 | float32 |

#### 减法

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `sub_rd` | `libdevice.sub_rd(x, y)` | float32, float32 | float32 |
| `sub_rn` | `libdevice.sub_rn(x, y)` | float32, float32 | float32 |
| `sub_ru` | `libdevice.sub_ru(x, y)` | float32, float32 | float32 |
| `sub_rz` | `libdevice.sub_rz(x, y)` | float32, float32 | float32 |

#### 乘法

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `mul_rd` | `libdevice.mul_rd(x, y)` | float32, float32 | float32 |
| `mul_rn` | `libdevice.mul_rn(x, y)` | float32, float32 | float32 |
| `mul_ru` | `libdevice.mul_ru(x, y)` | float32, float32 | float32 |
| `mul_rz` | `libdevice.mul_rz(x, y)` | float32, float32 | float32 |

#### 除法

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `fdiv` | `libdevice.fdiv(x, y)` | float32, float32 | float32 |
| `div_rd` | `libdevice.div_rd(x, y)` | float32, float32 | float32 |
| `div_rn` | `libdevice.div_rn(x, y)` | float32, float32 | float32 |
| `div_ru` | `libdevice.div_ru(x, y)` | float32, float32 | float32 |
| `div_rz` | `libdevice.div_rz(x, y)` | float32, float32 | float32 |

#### 倒数

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `rcp_rd` | `libdevice.rcp_rd(x)` | float32 | float32 |
| `rcp_rn` | `libdevice.rcp_rn(x)` | float32 | float32 |
| `rcp_ru` | `libdevice.rcp_ru(x)` | float32 | float32 |
| `rcp_rz` | `libdevice.rcp_rz(x)` | float32 | float32 |

#### 平方根（舍入控制）

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `sqrt_rd` | `libdevice.sqrt_rd(x)` | float32 | float32 |
| `sqrt_rn` | `libdevice.sqrt_rn(x)` | float32 | float32 |
| `sqrt_ru` | `libdevice.sqrt_ru(x)` | float32 | float32 |
| `sqrt_rz` | `libdevice.sqrt_rz(x)` | float32 | float32 |
| `rsqrt_rn` | `libdevice.rsqrt_rn(x)` | float32 | float32 |

#### 融合乘加（FMA）

计算 `x * y + z`，支持舍入模式控制：

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `fma` | `libdevice.fma(x, y, z)` | float32, float32, float32 | float32 |
| `fma_rd` | `libdevice.fma_rd(x, y, z)` | float32, float32, float32 | float32 |
| `fma_rn` | `libdevice.fma_rn(x, y, z)` | float32, float32, float32 | float32 |
| `fma_ru` | `libdevice.fma_ru(x, y, z)` | float32, float32, float32 | float32 |
| `fma_rz` | `libdevice.fma_rz(x, y, z)` | float32, float32, float32 | float32 |

### 五、类型转换

#### 浮点数转整数（32 位）

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `float2int_rd` | `libdevice.float2int_rd(x)` | float32 | int32 |
| `float2int_rn` | `libdevice.float2int_rn(x)` | float32 | int32 |
| `float2int_ru` | `libdevice.float2int_ru(x)` | float32 | int32 |
| `float2int_rz` | `libdevice.float2int_rz(x)` | float32 | int32 |

#### 浮点数转无符号整数（32 位）

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `float2uint_rd` | `libdevice.float2uint_rd(x)` | float32 | uint32 |
| `float2uint_rn` | `libdevice.float2uint_rn(x)` | float32 | uint32 |
| `float2uint_ru` | `libdevice.float2uint_ru(x)` | float32 | uint32 |
| `float2uint_rz` | `libdevice.float2uint_rz(x)` | float32 | uint32 |

#### 浮点数转整数（64 位）

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `float2ll_rd` | `libdevice.float2ll_rd(x)` | float32 | int64 |
| `float2ll_rn` | `libdevice.float2ll_rn(x)` | float32 | int64 |
| `float2ll_ru` | `libdevice.float2ll_ru(x)` | float32 | int64 |
| `float2ll_rz` | `libdevice.float2ll_rz(x)` | float32 | int64 |

#### 浮点数转无符号整数（64 位）

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `float2ull_rd` | `libdevice.float2ull_rd(x)` | float32 | uint64 |
| `float2ull_rn` | `libdevice.float2ull_rn(x)` | float32 | uint64 |
| `float2ull_ru` | `libdevice.float2ull_ru(x)` | float32 | uint64 |
| `float2ull_rz` | `libdevice.float2ull_rz(x)` | float32 | uint64 |

#### 整数转浮点数

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `int2float_rd` | `libdevice.int2float_rd(x)` | int32 | float32 |
| `int2float_rn` | `libdevice.int2float_rn(x)` | int32 | float32 |
| `int2float_ru` | `libdevice.int2float_ru(x)` | int32 | float32 |
| `int2float_rz` | `libdevice.int2float_rz(x)` | int32 | float32 |
| `uint2float_rd` | `libdevice.uint2float_rd(x)` | uint32 | float32 |
| `uint2float_rn` | `libdevice.uint2float_rn(x)` | uint32 | float32 |
| `uint2float_ru` | `libdevice.uint2float_ru(x)` | uint32 | float32 |
| `uint2float_rz` | `libdevice.uint2float_rz(x)` | uint32 | float32 |
| `ll2float_rd` | `libdevice.ll2float_rd(x)` | int64 | float32 |
| `ll2float_rn` | `libdevice.ll2float_rn(x)` | int64 | float32 |
| `ll2float_ru` | `libdevice.ll2float_ru(x)` | int64 | float32 |
| `ll2float_rz` | `libdevice.ll2float_rz(x)` | int64 | float32 |
| `ull2float_rd` | `libdevice.ull2float_rd(x)` | uint64 | float32 |
| `ull2float_rn` | `libdevice.ull2float_rn(x)` | uint64 | float32 |
| `ull2float_ru` | `libdevice.ull2float_ru(x)` | uint64 | float32 |
| `ull2float_rz` | `libdevice.ull2float_rz(x)` | uint64 | float32 |

#### 浮点数舍入为整数

| 函数 | 原型 | 输入类型 | 返回类型 |
|------|------|----------|----------|
| `llrint` | `libdevice.llrint(x)` | float32 | int64 |
| `llround` | `libdevice.llround(x)` | float32 | int64 |

#### 位重解释（不进行数值转换）

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `float_as_int` | `libdevice.float_as_int(x)` | float32 | int32 | float 的比特位重解释为 int32 |
| `float_as_uint` | `libdevice.float_as_uint(x)` | float32 | uint32 | float 的比特位重解释为 uint32 |
| `int_as_float` | `libdevice.int_as_float(x)` | int32 | float32 | int32 的比特位重解释为 float |
| `uint_as_float` | `libdevice.uint_as_float(x)` | uint32 | float32 | uint32 的比特位重解释为 float |

### 六、位操作

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `brev` | `libdevice.brev(x)` | int32 | int32 | 32 位整数位反转 |
| `clz` | `libdevice.clz(x)` | int32 | int32 | 前导零数量，范围 [0, 32] |
| `ffs` | `libdevice.ffs(x)` | int32 | int32 | 最低置 1 位的索引，范围 [0, 32] |
| `popc` | `libdevice.popc(x)` | int32 | int32 | 置 1 位的数量，范围 [0, 32] |
| `byte_perm` | `libdevice.byte_perm(x, y, s)` | int32, int32, int32 | int32 | 从两个 32 位整数中选择字节组成新整数 |
| `mul24` | `libdevice.mul24(x, y)` | int32, int32 | int32 | 低 24 位乘法 |
| `mulhi` | `libdevice.mulhi(x, y)` | int32, int32 | int32 | 有符号乘法高 32 位 |
| `umulhi` | `libdevice.umulhi(x, y)` | int32, int32 | int32 | 无符号乘法高 32 位 |
| `hadd` | `libdevice.hadd(x, y)` | int32, int32 | int32 | (x+y) 的一半的取整结果 |
| `rhadd` | `libdevice.rhadd(x, y)` | int32, int32 | int32 | (x+y) 平均值的取整结果 |
| `sad` | `libdevice.sad(x, y, z)` | int32, int32, int32 | int32 | \|x-y\|+z |

### 七、判断与符号函数

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `isinf` | `libdevice.isinf(x)` | float32 | bool | 是否为无穷大 |
| `isnan` | `libdevice.isnan(x)` | float32 | bool | 是否为 NaN |
| `isfinited` | `libdevice.isfinited(x)` | float32 | bool | 是否为有限值 |
| `finitef` | `libdevice.finitef(x)` | float32 | bool | 是否为有限浮点数 |
| `signbit` | `libdevice.signbit(x)` | float32 | int32 | 获取符号位 |
| `copysign` | `libdevice.copysign(x, y)` | float32, float32 | float32 | \|x\| 的符号设为 y 的符号 |

### 八、特殊函数

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `erf` | `libdevice.erf(x)` | float32 | float32 | 误差函数 |
| `erfc` | `libdevice.erfc(x)` | float32 | float32 | 互补误差函数 1-erf(x) |
| `erfinv` | `libdevice.erfinv(x)` | float32 | float32 | 逆误差函数 |
| `erfcinv` | `libdevice.erfcinv(x)` | float32 | float32 | 逆互补误差函数 |
| `erfcx` | `libdevice.erfcx(x)` | float32 | float32 | 缩放互补误差函数 exp(x^2)*erfc(x) |
| `normcdf` | `libdevice.normcdf(x)` | float32 | float32 | 标准正态分布累积分布函数 |
| `normcdfinv` | `libdevice.normcdfinv(x)` | float32 | float32 | 标准正态分布累积分布逆函数 |
| `relu` | `libdevice.relu(x)` | float32 | float32 | 修正线性单元 max(x, 0) |

### 九、贝塞尔函数

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `j0` | `libdevice.j0(x)` | float32 | float32 | 零阶第一类贝塞尔函数 |
| `j1` | `libdevice.j1(x)` | float32 | float32 | 一阶第一类贝塞尔函数 |
| `jn` | `libdevice.jn(n, x)` | int32, float32 | float32 | n 阶第一类贝塞尔函数 |
| `y0` | `libdevice.y0(x)` | float32 | float32 | 零阶第二类贝塞尔函数 |
| `y1` | `libdevice.y1(x)` | float32 | float32 | 一阶第二类贝塞尔函数 |
| `yn` | `libdevice.yn(n, x)` | int32, float32 | float32 | n 阶第二类贝塞尔函数 |

### 十、快速近似函数

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `fast_sinf` | `libdevice.fast_sinf(x)` | float32 | float32 | 快速近似正弦 |
| `fast_cosf` | `libdevice.fast_cosf(x)` | float32 | float32 | 快速近似余弦 |
| `fast_tanf` | `libdevice.fast_tanf(x)` | float32 | float32 | 快速近似正切 |
| `fast_expf` | `libdevice.fast_expf(x)` | float32 | float32 | 快速近似指数 |
| `fast_exp10f` | `libdevice.fast_exp10f(x)` | float32 | float32 | 快速近似 10^x |
| `fast_logf` | `libdevice.fast_logf(x)` | float32 | float32 | 快速近似自然对数 |
| `fast_log2f` | `libdevice.fast_log2f(x)` | float32 | float32 | 快速近似 log2 |
| `fast_log10f` | `libdevice.fast_log10f(x)` | float32 | float32 | 快速近似 log10 |
| `fast_dividef` | `libdevice.fast_dividef(x, y)` | float32, float32 | float32 | 快速近似除法 |
| `fast_powf` | `libdevice.fast_powf(x, y)` | float32, float32 | float32 | 快速近似幂函数 |

### 十一、向量范数

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `norm3d` | `libdevice.norm3d(x, y, z)` | float32, float32, float32 | float32 | 三维欧几里得范数 |
| `norm4d` | `libdevice.norm4d(x, y, z, w)` | float32, float32, float32, float32 | float32 | 四维欧几里得范数 |
| `rnorm3d` | `libdevice.rnorm3d(x, y, z)` | float32, float32, float32 | float32 | 三维欧几里得范数倒数 |
| `rnorm4d` | `libdevice.rnorm4d(x, y, z, w)` | float32, float32, float32, float32 | float32 | 四维欧几里得范数倒数 |
| `rhypot` | `libdevice.rhypot(x, y)` | float32, float32 | float32 | 欧几里得距离倒数 |

### 十二、其他

| 函数 | 原型 | 输入类型 | 返回类型 | 说明 |
|------|------|----------|----------|------|
| `flip` | `libdevice.flip(ptr, dim)` | tensor, int32 | tensor | 沿指定维度反转张量元素顺序 |

---

## 编译模式支持汇总

以下函数同时支持 SIMT 和 SIMD 编译模式（其余仅支持 SIMT）：

- `acos` -- SIMT, SIMD
- `atan` -- SIMT, SIMD
- `atan2` -- SIMT, SIMD
- `div_rz` -- SIMT, SIMD
- `fast_dividef` -- SIMT, SIMD
- `fast_expf` -- SIMT, SIMD
- `float_as_int` -- SIMT, SIMD
- `fmod` -- SIMT, SIMD
- `ilogb` -- SIMT, SIMD
- `isinf` -- SIMT, SIMD
- `isnan` -- SIMT, SIMD
- `ldexp` -- SIMT, SIMD
- `log1p` -- SIMT, SIMD
- `pow` -- SIMT, SIMD
- `reciprocal` -- SIMT, SIMD
- `relu` -- SIMT, SIMD
- `round` -- SIMT, SIMD
- `tan` -- SIMT, SIMD
- `tanh` -- SIMT, SIMD
- `trunc` -- SIMT, SIMD
