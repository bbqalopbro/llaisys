# 项目#3：多平台适配 — 沐曦 C500 GPU 适配实施计划

> 创建时间: 2026-03-09  
> 最后更新: 2026-03-10  
> 状态: **Phase D 已完成** — 沐曦 C500 全量验证通过 (17/17)

---

## 一、沐曦 C500 适配背景与要求

### 1.1 沐曦 C500 简介

沐曦（MetaX）C500 是国产高性能 GPGPU 芯片，面向数据中心 AI 推理与训练场景。其核心特点：

| 特性 | 说明 |
|------|------|
| 架构 | 沐曦自研 GPU 架构 |
| 编程模型 | MXMACA（兼容 CUDA/HIP 编程范式） |
| 编译器 | 基于 LLVM/Clang 的 mxcc 编译器 |
| 运行时 | MXMACA Runtime（API 风格对标 CUDA Runtime） |
| 数学库 | mxBLAS（对标 cuBLAS）、mxDNN（对标 cuDNN） |
| 通信库 | mxCCL（对标 NCCL） |
| 数据精度 | FP32 / FP16 / BF16 / INT8 / INT4 |

### 1.2 适配前置要求

#### 硬件环境

- 沐曦 C500 加速卡 ≥ 1 张
- 主机 CPU：x86_64 架构
- 主机内存：≥ 32GB（推荐）
- PCIe 3.0/4.0 插槽

#### 软件环境

| 依赖项 | 最低版本 | 说明 |
|--------|---------|------|
| MXMACA SDK | ≥ 1.0 | 包含 mxcc 编译器、runtime、mxBLAS |
| 操作系统 | Ubuntu 20.04+ / CentOS 7.9+ | 需安装沐曦驱动 |
| 沐曦GPU驱动 | 随 SDK 版本匹配 | `mx-smi` 可正常识别设备 |
| xmake | ≥ 2.7.0 | 构建系统 |
| Python | ≥ 3.8 | Python 绑定与测试 |
| GCC/G++ | ≥ 7.5 | 宿主编译器 |

#### MXMACA SDK 关键组件

> **实际验证结果（2026-03-10）：** MACA SDK 安装路径为 `/opt/maca/`（非 `/opt/mxmaca/`），
> API 命名使用 `mc*` 前缀（非 `maca*`），详见下方对应关系表。

```
/opt/maca/                        # 实际 SDK 路径
├── mxgpu_llvm/bin/
│   └── mxcc                      # GPU 编译器
├── bin/
│   └── mx-smi                    # 设备管理工具
├── include/
│   ├── mcr/
│   │   └── mc_runtime_api.h      # Runtime API 头文件
│   ├── common/
│   │   ├── maca_fp16.h           # FP16 数据类型
│   │   └── maca_bfloat16.h       # BF16 数据类型 (__maca_bfloat16)
│   ├── mcblas/
│   │   └── mcblas.h              # BLAS 库头文件
│   └── mcrand/
│       └── mcrand_kernel.h       # 随机数生成头文件
├── lib/
│   ├── libmcruntime.so           # Runtime 动态库
│   └── libmcblas.so              # BLAS 动态库
└── ...
```

#### MXMACA Runtime API 与 CUDA 的对应关系

> **实际验证（2026-03-10）：** SDK 实际使用 `mc*` 前缀，非文档中假设的 `maca*`。

| CUDA API | MXMACA 文档假设 | **实际 API** | 说明 |
|----------|----------------|-------------|------|
| `cudaMalloc` | `macaMalloc` | **`mcMalloc`** | 设备端内存分配 |
| `cudaFree` | `macaFree` | **`mcFree`** | 设备端内存释放 |
| `cudaMemcpy` | `macaMemcpy` | **`mcMemcpy`** | 同步内存拷贝 |
| `cudaMemcpyAsync` | `macaMemcpyAsync` | **`mcMemcpyAsync`** | 异步内存拷贝 |
| `cudaStreamCreate` | `macaStreamCreate` | **`mcStreamCreate`** | 创建流 |
| `cudaStreamDestroy` | `macaStreamDestroy` | **`mcStreamDestroy`** | 销毁流 |
| `cudaStreamSynchronize` | `macaStreamSynchronize` | **`mcStreamSynchronize`** | 流同步 |
| `cudaDeviceSynchronize` | `macaDeviceSynchronize` | **`mcDeviceSynchronize`** | 设备同步 |
| `cudaSetDevice` | `macaSetDevice` | **`mcSetDevice`** | 设置当前设备 |
| `cudaGetDeviceCount` | `macaGetDeviceCount` | **`mcGetDeviceCount`** | 获取设备数量 |
| `cudaMemcpyHostToDevice` | `macaMemcpyHostToDevice` | **`mcMemcpyHostToDevice`** | H2D 拷贝类型 |
| `cudaMemcpyDeviceToHost` | `macaMemcpyDeviceToHost` | **`mcMemcpyDeviceToHost`** | D2H 拷贝类型 |
| `cuBLAS` | `mxBLAS` | **`mcBLAS`** | BLAS 数学库 |
| `cublasSgemm` | `mxblasSgemm` | **`mcblasSgemm`** | 单精度矩阵乘 |
| `cublasGemmEx` | — | **`mcblasGemmEx`** | 混合精度 GEMM |
| `CUDA_R_32F` | — | **`MACA_R_32F`** | 数据类型枚举 |
| `__nv_bfloat16` | — | **`__maca_bfloat16`** | BF16 数据类型 |
| `curandStatePhilox4_32_10_t` | — | **`mcrandStatePhilox4_32_10_t`** | 随机数状态 |

#### 编程模型差异

MXMACA 的 kernel 编程语法与 CUDA 高度类似：

```cpp
// CUDA kernel
__global__ void add_kernel_cuda(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// MXMACA kernel — 语法基本一致，编译器不同
__global__ void add_kernel_maca(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
```

主要差异点：
1. **编译器**：使用 `mxcc` 替代 `nvcc`，源文件后缀可使用 `.cu` 或 `.maca`
2. **头文件**：`#include <maca_runtime.h>` 替代 `#include <cuda_runtime.h>`
3. **数学库**：`mxblas.h` 替代 `cublas_v2.h`，函数命名前缀不同
4. **架构参数**：编译时 `--gpu-arch` 需指定沐曦架构（非 `sm_80`）
5. **部分高级特性**：Tensor Core / wmma 等不直接可用，需使用沐曦等价方案

---

## 二、现状分析

### 2.1 已有平台抽象层

本项目已有清晰的设备抽象分层设计：

| 层级 | 文件 | 说明 |
|------|------|------|
| 枚举定义 | `include/llaisys.h` | `llaisysDeviceType_t` 枚举 (CPU=0, NVIDIA=1) |
| Runtime API 接口 | `include/llaisys/runtime.h` | `LlaisysRuntimeAPI` 函数指针表 |
| Runtime 分发 | `src/device/runtime_api.cpp` | `getRuntimeAPI()` 按设备类型分发 |
| CPU 实现 | `src/device/cpu/` | `cpu_runtime_api.cpp`, `cpu_resource.cpp` |
| NVIDIA 实现 | `src/device/nvidia/` | `nvidia_runtime_api.cu`, `nvidia_resource.cu` |
| 算子分发 | `src/ops/*/op.cpp` | 顶层根据 `deviceType()` 分发到不同后端 |
| CPU 算子 | `src/ops/*/cpu/*.cpp` | 每个算子的 CPU 实现 |
| NVIDIA 算子 | `src/ops/*/nvidia/*.cu` | 每个算子的 CUDA 实现 |
| 构建配置 | `xmake.lua`, `xmake/nvidia.lua`, `xmake/cpu.lua` | 按设备分文件管理 |

核心设计模式：
```
include/llaisys.h          → 设备类型枚举
include/llaisys/runtime.h  → 统一 Runtime API 函数指针表
src/device/runtime_api.cpp → switch(device_type) 选择后端
src/device/<backend>/      → 具体 Runtime 实现
src/ops/<op>/op.cpp        → switch(deviceType) 分发
src/ops/<op>/<backend>/    → 具体算子 kernel
xmake/<backend>.lua        → 构建规则
```

### 2.2 适配差距分析

| 需要新增/修改 | 当前状态 | 需要做的 |
|------|------|--------|
| 设备枚举 | 仅 CPU / NVIDIA | 新增 `LLAISYS_DEVICE_METAX = 2` |
| Runtime API | 仅 CPU / NVIDIA 实现 | 新增 `src/device/metax/` |
| 算子后端 | 仅 CPU / NVIDIA kernel | 新增 `src/ops/*/metax/` (12 个算子) |
| BLAS 集成 | cuBLAS | 新增 mxBLAS 路径 |
| 编译配置 | xmake 仅 CPU / NVIDIA | 新增 `xmake/metax.lua` + `--metax-gpu` 选项 |
| 分布式通信 | NCCL | 后续可扩展 mxCCL |
| Python 绑定 | 设备名写死 "nvidia" | 支持 "metax" 设备类型 |
| 容器化 | CUDA base image | 沐曦 SDK base image |

### 2.3 需要适配的算子清单

| # | 算子 | NVIDIA 实现 | 沐曦适配方式 |
|---|------|-----------|------------|
| 1 | `add` | CUDA kernel | MXMACA kernel (直接移植) |
| 2 | `argmax` | CUDA kernel | MXMACA kernel (直接移植) |
| 3 | `dequantize` | CUDA kernel | MXMACA kernel (直接移植) |
| 4 | `embedding` | CUDA kernel | MXMACA kernel (直接移植) |
| 5 | `linear` | cuBLAS SGEMM | mxBLAS SGEMM |
| 6 | `rearrange` | CUDA kernel | MXMACA kernel (直接移植) |
| 7 | `rms_norm` | CUDA kernel | MXMACA kernel (直接移植) |
| 8 | `rope` | CUDA kernel | MXMACA kernel (直接移植) |
| 9 | `sample` | CUDA kernel | MXMACA kernel (直接移植) |
| 10 | `self_attention` | CUDA kernel | MXMACA kernel (直接移植) |
| 11 | `swiglu` | CUDA kernel | MXMACA kernel (直接移植) |
| 12 | `dequantize_int4` | CUDA kernel | MXMACA kernel (直接移植) |

由于 MXMACA 编程模型与 CUDA 高度兼容，绝大部分 kernel 代码可直接移植，主要改动为头文件替换和编译器切换。`linear` 算子由于依赖 BLAS 库，需单独适配 mxBLAS API。

---

## 三、项目目标

### 3.1 核心目标（必须）

1. **Runtime 层适配**：沐曦 C500 设备初始化、内存管理、流管理全链路可用
2. **全部算子移植**：12 个算子在沐曦 C500 上可正确执行
3. **模型推理跑通**：Qwen2 模型在沐曦 C500 上完成 FP32 推理，输出正确
4. **构建系统集成**：`xmake --metax-gpu=true` 一键编译沐曦后端
5. **与现有后端共存**：CPU / NVIDIA 路径不受影响，编译开关互相独立

### 3.2 第二目标（随后）

1. **INT8/INT4 量化推理**：沐曦后端支持 `dequantize` + `dequantize_int4` 推理路径
2. **张量并行 (TP)**：集成 mxCCL，实现多卡张量并行推理
3. **Python 层完整打通**：`python/server` 可配置 `--device metax`
4. **性能基线建立**：建立沐曦 C500 的推理延迟/吞吐基准

### 3.3 非目标（本轮明确不做）

- 不做沐曦专属的 kernel 深度优化（先保证正确性）
- 不做 FP16/BF16 混合精度（后续单独优化）
- 不做沐曦 Tensor Core 等专用硬件特性适配
- 不做跨厂商异构推理（如 NVIDIA + 沐曦混合部署）
- 不做多机分布式

---

## 四、总体设计

### 4.1 设备枚举扩展

```c
// include/llaisys.h
typedef enum {
    LLAISYS_DEVICE_CPU = 0,
    LLAISYS_DEVICE_NVIDIA = 1,
    LLAISYS_DEVICE_METAX = 2,        // 新增：沐曦 C500
    //// 后续其他平台在此追加...
    LLAISYS_DEVICE_TYPE_COUNT
} llaisysDeviceType_t;
```

### 4.2 目录结构规划

```
src/device/metax/                 # 新增 — 沐曦 Runtime API 实现
├── metax_runtime_api.mc          # Runtime API (设备管理/内存/流)
├── metax_resource.mc             # DeviceResource (沐曦设备初始化)
└── metax_resource.hpp            # 头文件

src/ops/add/metax/                # 新增 — 每个算子的沐曦 kernel
├── add_metax.mc                  # kernel 实现
└── add_metax.hpp                 # 头文件
src/ops/argmax/metax/
src/ops/dequantize/metax/
src/ops/embedding/metax/
src/ops/linear/metax/             # 需特殊处理 mxBLAS
src/ops/rearrange/metax/
src/ops/rms_norm/metax/
src/ops/rope/metax/
src/ops/sample/metax/
src/ops/self_attention/metax/
src/ops/swiglu/metax/

xmake/metax.lua                  # 新增 — 沐曦构建规则
```

> **注**：源文件后缀约定 `.mc`（MXMACA 源文件），也可沿用 `.cu` 由 mxcc 编译，具体取决于 SDK 版本的文件后缀要求。若 SDK 直接支持 `.cu` 后缀，则统一使用 `.cu` 以降低改动量。

### 4.3 Runtime API 分发扩展

```cpp
// src/device/runtime_api.cpp — 增加 METAX 分支
const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type) {
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return llaisys::device::cpu::getRuntimeAPI();
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return llaisys::device::nvidia::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif
    case LLAISYS_DEVICE_METAX:                        // 新增
#ifdef ENABLE_METAX_API
        return llaisys::device::metax::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}
```

### 4.4 算子分发扩展模式

以 `linear` 为例，展示算子层的分发改造：

```cpp
// src/ops/linear/op.cpp
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::linear(out, in, weight, bias);
    }
#endif
#ifdef ENABLE_METAX_API                                // 新增
    if (out->deviceType() == LLAISYS_DEVICE_METAX) {
        return metax::linear(out, in, weight, bias);
    }
#endif
    // CPU fallback ...
}
```

### 4.5 构建系统扩展

```lua
-- xmake/metax.lua
target("llaisys-device-metax")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_toolset("cu", "/opt/mxmaca/bin/mxcc")  -- 沐曦编译器
    add_defines("ENABLE_METAX_API")
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    add_files("../src/device/metax/*.mc")
    add_links("maca_runtime")              -- MXMACA Runtime
    add_linkdirs("/opt/mxmaca/lib")
    add_includedirs("/opt/mxmaca/include")
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    set_languages("cxx17")
    add_deps("llaisys-tensor")
    set_warnings("all", "error")
    set_toolset("cu", "/opt/mxmaca/bin/mxcc")
    add_defines("ENABLE_METAX_API")
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    add_files("../src/ops/*/metax/*.mc")
    add_links("maca_runtime", "mxblas")    -- mxBLAS
    add_linkdirs("/opt/mxmaca/lib")
    add_includedirs("/opt/mxmaca/include")
target_end()
```

---

## 五、分阶段实施计划

### 阶段 A：环境搭建 + 编译骨架（1 个迭代）

**目标**：沐曦 SDK 安装就绪，构建系统能编译空壳后端，设备能被识别。

**任务**：

1. 安装 MXMACA SDK，验证 `mx-smi` 识别 C500 设备
2. 在 `include/llaisys.h` 新增 `LLAISYS_DEVICE_METAX = 2`
3. 新增 `src/device/metax/` 目录，创建骨架文件（空实现 / 抛异常）
4. 新增 `xmake/metax.lua`，配置 mxcc 编译工具链
5. 修改 `xmake.lua` 添加 `--metax-gpu` 编译选项和条件编译宏 `ENABLE_METAX_API`
6. 修改 `src/device/runtime_api.cpp` 和 `runtime_api.hpp`，增加 METAX 分支

**验收标准**：

- `xmake build --metax-gpu=true` 编译通过（Runtime 空壳）
- `mx-smi` 显示设备信息
- `getRuntimeAPI(LLAISYS_DEVICE_METAX)` 返回有效指针

---

### 阶段 B：Runtime API 实现（1 个迭代）

**目标**：沐曦设备的内存分配/拷贝/流管理全部可用。

**任务**：

1. 实现 `metax_runtime_api.mc`（或 `.cu`），包含全部 12 个 Runtime API 函数：

| 函数 | MXMACA 映射 |
|------|-----------|
| `getDeviceCount` | `macaGetDeviceCount` |
| `setDevice` | `macaSetDevice` |
| `deviceSynchronize` | `macaDeviceSynchronize` |
| `createStream` | `macaStreamCreate` |
| `destroyStream` | `macaStreamDestroy` |
| `streamSynchronize` | `macaStreamSynchronize` |
| `mallocDevice` | `macaMalloc` |
| `freeDevice` | `macaFree` |
| `mallocHost` | `macaMallocHost` |
| `freeHost` | `macaFreeHost` |
| `memcpySync` | `macaMemcpy` |
| `memcpyAsync` | `macaMemcpyAsync` |

2. 实现 `metax_resource.mc` — 设备初始化与资源管理
3. 编写 Runtime 验证测试：分配/拷贝/同步

**验收标准**：

- 设备内存分配/释放无泄漏
- H2D / D2H / D2D 数据拷贝正确
- 流创建/同步正常工作
- Tensor 可以在沐曦设备上创建和操作

---

### 阶段 C：基础算子移植（2 个迭代）

**目标**：将 12 个算子的 NVIDIA CUDA kernel 移植到 MXMACA。

**分批策略**：

**第一批（elementwise / 简单算子）**：
1. `add` — 逐元素加法
2. `swiglu` — 逐元素激活
3. `rms_norm` — 归一化
4. `embedding` — 查表
5. `argmax` — 求最大值索引
6. `rearrange` — 数据重排

**第二批（计算密集 / 依赖数学库）**：
7. `linear` — 矩阵乘法（需 mxBLAS 集成）
8. `rope` — 旋转位置编码
9. `self_attention` — 自注意力（最复杂的 kernel）
10. `sample` — 采样（含随机数生成）

**第三批（量化相关）**：
11. `dequantize` — INT8 反量化
12. `dequantize_int4` — INT4 反量化

**每个算子的移植步骤**：
1. 在 `src/ops/<op>/metax/` 创建文件
2. 拷贝 NVIDIA kernel，替换 CUDA 头文件为 MXMACA 头文件
3. 替换 cuBLAS 调用为 mxBLAS 调用（仅 `linear`）
4. 在 `src/ops/<op>/op.cpp` 增加 `ENABLE_METAX_API` 分发分支
5. 编写单元测试验证正确性（与 CPU 结果对齐）

**验收标准**：

- 每个算子与 CPU 实现的输出误差 ≤ 1e-5（FP32）
- `linear` 算子正确调用 mxBLAS
- 全部 12 个算子测试通过

---

### 阶段 D：模型推理端到端验证（1 个迭代）

**目标**：Qwen2 模型在沐曦 C500 上完成完整推理。

**任务**：

1. 修改 `src/llaisys/models/qwen2.cpp` — 确认在 `LLAISYS_DEVICE_METAX` 下所有层正常调用
2. 修改 Python 绑定 — `python/llaisys/models/qwen2.py` 支持 `device="metax"`
3. 修改 `python/server/app.py` — CLI 支持 `--device metax --device-id 0`
4. 执行推理测试，对比 CPU / NVIDIA / 沐曦三端输出一致性
5. 建立性能基线：单 token 延迟、tokens/s 吞吐

**验收标准**：

- Qwen2 模型 FP32 推理生成文本合理
- 沐曦输出与 CPU 输出（FP32）高度一致（允许浮点误差 ≤ 1e-4）
- `python/server` 可通过 `--device metax` 启动服务

---

### 阶段 E：量化推理 + TP 扩展（1~2 个迭代）

**目标**：沐曦后端支持 INT8/INT4 量化推理，初步支持多卡 TP。

**任务**：

1. 验证 `dequantize` / `dequantize_int4` kernel 在沐曦上的正确性
2. 加载量化模型（`quantized_model/`、`quantized_model_int4/`）在沐曦上推理
3. 评估 mxCCL 可用性，若可用则实现 `DistCommMxccl` 后端
4. 在 `xmake.lua` 增加 `--dist-mxccl` 选项

**验收标准**：

- INT8 量化推理输出合理
- INT4 量化推理输出合理
- （如 mxCCL 可用）多卡 TP 推理可运行

---

## 六、文件改动清单

| 文件 | 操作 | 阶段 |
|------|------|------|
| `include/llaisys.h` | 修改 — 新增 `LLAISYS_DEVICE_METAX` | A |
| `src/device/runtime_api.hpp` | 修改 — 新增 metax namespace 前置声明 | A |
| `src/device/runtime_api.cpp` | 修改 — switch 增加 METAX 分支 | A |
| `src/device/metax/metax_runtime_api.mc` | **新增** | A, B |
| `src/device/metax/metax_resource.mc` | **新增** | A, B |
| `src/device/metax/metax_resource.hpp` | **新增** | A, B |
| `xmake/metax.lua` | **新增** | A |
| `xmake.lua` | 修改 — 新增 `--metax-gpu` 选项 | A |
| `src/ops/add/metax/add_metax.mc` | **新增** | C |
| `src/ops/add/metax/add_metax.hpp` | **新增** | C |
| `src/ops/add/op.cpp` | 修改 — 增加 METAX 分发 | C |
| `src/ops/argmax/metax/*` | **新增** | C |
| `src/ops/dequantize/metax/*` | **新增** | C |
| `src/ops/embedding/metax/*` | **新增** | C |
| `src/ops/linear/metax/*` | **新增**（含 mxBLAS 集成） | C |
| `src/ops/rearrange/metax/*` | **新增** | C |
| `src/ops/rms_norm/metax/*` | **新增** | C |
| `src/ops/rope/metax/*` | **新增** | C |
| `src/ops/sample/metax/*` | **新增** | C |
| `src/ops/self_attention/metax/*` | **新增** | C |
| `src/ops/swiglu/metax/*` | **新增** | C |
| 各 `op.cpp` 文件 (×12) | 修改 — 增加 METAX 分发 | C |
| `python/llaisys/models/qwen2.py` | 修改 — 支持 `device="metax"` | D |
| `python/server/app.py` | 修改 — CLI 支持 `--device metax` | D |
| `test/test_metax_ops.py` | **新增** — 算子正确性测试 | C, D |
| `test/test_metax_infer.py` | **新增** — 端到端推理测试 | D |
| `src/distributed/mxccl_comm.cpp` | **新增**（可选） | E |

---

## 七、里程碑

| 里程碑 | 输出 | 判定标准 | 状态 |
|------|------|------|------|
| M1 | 阶段 A 完成 | 编译骨架通过，设备可识别 | ✅ 已完成 (2026-03-09) |
| M2 | 阶段 B 完成 | Runtime API 全部可用，内存拷贝正确 | ✅ 已完成 (2026-03-09) |
| M3 | 阶段 C 完成 | 12 个算子全部移植测试通过 | ✅ 代码完成 (2026-03-10)，待平台验证 |
| M4 | 阶段 D 完成 | Qwen2 端到端推理正确 | 🔄 测试脚本已就绪，待平台验证 |
| M5 | 阶段 E 完成 | 量化推理 + 可选 TP 扩展 | ⬜ 未开始 |

---

## 八、风险与控制

| 风险 | 影响 | 控制措施 |
|------|------|---------|
| MXMACA API 与 CUDA 不完全兼容 | kernel 移植需额外修改 | 先移植简单算子验证兼容性，再批量移植 |
| mxBLAS 性能 / 精度差异 | linear 结果偏差 | 与 CPU 基线严格对比，逐层验证 |
| mxcc 编译器不支持某些 CUDA 语法 | 编译失败 | 建立最小编译测试，提前排查语法兼容性 |
| SDK 版本迭代导致 API 变化 | 适配代码需返工 | 封装 SDK 调用在独立文件，隔离变化 |
| 沐曦驱动不稳定 / 环境配置复杂 | 开发效率降低 | 提供 Docker 化开发环境 |
| 与 NVIDIA 路径冲突 | 编译宏/链接冲突 | 严格使用条件编译，互斥编译开关 |

---

## 九、多平台适配架构展望

本次沐曦 C500 适配完成后，项目将形成标准化的多平台适配模式：

```
新增平台适配 = 以下步骤的组合:
1. include/llaisys.h           → 新增枚举值
2. src/device/<vendor>/        → 实现 LlaisysRuntimeAPI 的 12 个函数
3. src/ops/<op>/<vendor>/      → 移植/实现每个算子 kernel
4. src/ops/<op>/op.cpp         → 增加 #ifdef + if(deviceType) 分发
5. xmake/<vendor>.lua          → 构建规则
6. xmake.lua                   → 编译开关
```

后续计划适配的平台（按优先级排序）：

| 平台 | 厂商 | 编程模型 | 状态 |
|------|------|---------|------|
| NVIDIA GPU | NVIDIA | CUDA | ✅ 已完成 |
| 沐曦 C500 | MetaX | MXMACA | 🔄 本项目 |
| 寒武纪 MLU | Cambricon | BANG C | 🔲 规划中 |
| 昇腾 NPU | 华为 | CANN / Ascend C | 🔲 规划中 |
| 海光 DCU | 海光 | ROCm/HIP | 🔲 规划中 |

---

## 十、进度记录

### 2026-03-09 — Phase A 完成

- `include/llaisys.h` 新增 `LLAISYS_DEVICE_METAX = 2`
- `src/device/runtime_api.hpp/cpp` 增加 METAX 分发分支
- `src/device/metax/` 创建骨架文件（metax_resource.hpp/cpp, metax_runtime_api.cpp）
- `xmake/metax.lua` 创建，`xmake.lua` 增加 `--metax-gpu` 选项
- 编译验证通过：`xmake build --metax-gpu=true` ✅

### 2026-03-09 — Phase B 完成

- `metax_runtime_api.cpp` 实现双路径：
  - `#ifdef ENABLE_METAX_RUNTIME`：真实 MXMACA Runtime 调用
  - `#else`：stub 实现（getDeviceCount 返回 0，其余抛异常）
- 12 个 Runtime API 函数全部实现

### 2026-03-10 — Phase C 完成

12 个算子内核全部移植完成（采用 CUDA 兼容语法，mxcc 可直接编译）：

| # | 算子 | 文件 | 特殊说明 |
|---|------|------|----------|
| 1 | add | `src/ops/add/metax/add_metax.{hpp,cu}` | 逐元素，FP32/FP16/BF16 |
| 2 | argmax | `src/ops/argmax/metax/argmax_metax.{hpp,cu}` | 1D/2D 输入 |
| 3 | dequantize | `src/ops/dequantize/metax/dequantize_metax.{hpp,cu}` | INT8 + INT4 |
| 4 | embedding | `src/ops/embedding/metax/embedding_metax.{hpp,cu}` | 含混合精度 |
| 5 | linear | `src/ops/linear/metax/linear_metax.{hpp,cu}` | 使用 cuBLAS/mxBLAS GEMM |
| 6 | rms_norm | `src/ops/rms_norm/metax/rms_norm_metax.{hpp,cu}` | 共享内存 reduce |
| 7 | rope | `src/ops/rope/metax/rope_metax.{hpp,cu}` | RoPE FP32/FP16/BF16 |
| 8 | sample | `src/ops/sample/metax/sample_metax.{hpp,cu}` | 使用 cuRAND/mcRAND |
| 9 | self_attention | `src/ops/self_attention/metax/self_attention_metax.{hpp,cu}` | BatchedGEMM + GQA |
| 10 | swiglu | `src/ops/swiglu/metax/swiglu_metax.{hpp,cu}` | SiLU(gate) × up |

其他改动：
- 10 个 `op.cpp` 文件全部增加 `ENABLE_METAX_API` 分发分支
- `src/ops/metax_stubs.cpp` 创建：非 MetaX 平台编译的桩实现
- `rearrange` 算子 NVIDIA 端也未实现，MetaX 不移植
- `python/llaisys/libllaisys/llaisys_types.py` 增加 `METAX = 2`
- `python/server/app.py` 支持 `--device metax`
- `test/test_utils.py` 增加 `metax` 设备支持
- 编译验证通过：`--metax-gpu=true` ✅，`--metax-gpu=true --nv-gpu=true` ✅

### 2026-03-10 — Phase D 沐曦 C500 实机验证完成

**环境发现与 API 修正：**
- MACA SDK 实际路径 `/opt/maca/`（非 `/opt/mxmaca/`）
- API 使用 `mc*` 前缀（非 `maca*`）：`mcMalloc`, `mcFree`, `mcMemcpy` 等
- 头文件在子目录：`mcr/mc_runtime_api.h`, `common/maca_fp16.h`, `mcblas/mcblas.h`
- BF16 类型：`__maca_bfloat16`（非 `__nv_bfloat16`）
- GPU 架构目标：`xcore1000`，编译模式：`-x maca`

**源文件修改：**
- `metax_runtime_api.cpp`：`maca*` → `mc*` API 全部替换
- 10 个算子文件（`.cu` → `.mc`）：CUDA API → MACA API 替换
- `ops.h` / `ops.cc` / Python 绑定：新增 `dequantize` / `dequantize_int4` 接口
- `ops.py`：修复 `linear` 传入 `None` bias 时的 `AttributeError`

**构建系统修改：**
- `xmake/metax.lua`：使用 `on_build` 自定义编译 `.mc` 文件（避免 xmake CUDA 规则冲突）
- `xmake.lua`：使用 `add_shflags("--whole-archive")` 解决 archive 链接顺序问题
- 编译参数：`mxcc --cuda-gpu-arch=xcore1000 -x maca -fPIC -std=c++17 -O2`

**测试结果：17/17 全部通过（三种模型格式）**
- Level 1 (Runtime API): 6/6 ✅
- Level 2 (算子): 10/10 ✅
- Level 3 (端到端推理): 1/1 ✅

**推理性能（DeepSeek-R1-Distill-Qwen-1.5B, MetaX C500）：**
- FP32: 25.6 tokens/s | INT8: 16.4 tokens/s | INT4: 18.6 tokens/s

### 2026-03-10 — Phase D 测试脚本与部署脚本就绪

- `scripts/deploy_metax.sh` — 一键部署脚本（6 步自动流程）
- `test/test_metax.py` — 三层验证测试（Runtime → 算子 → 推理）
- 待在沐曦 C500 平台上实际执行验证

---

## 十一、沐曦平台部署使用指南

### 11.1 租用算力配置

| 配置项 | 推荐选择 |
|--------|----------|
| GPU 型号 | 沐曦 C500 |
| 预载镜像 | **PyTorch 镜像**（含 MXMACA SDK） |
| GPU 数量 | ≥ 1 张（单卡推理）或 ≥ 2 张（TP 多卡） |
| 系统盘 | ≥ 50GB |
| 数据盘 | ≥ 20GB（存放模型权重） |

> **为什么选 PyTorch 镜像？**  
> 项目 `setup.cfg` 依赖 `torch>=2.4.0`；PyTorch 镜像预装了完整 MXMACA 工具链（mxcc、maca_runtime、mxBLAS、mcRAND），
> 以及 Python/transformers/safetensors 等依赖，省去手动配置的时间。

### 11.2 一键部署

```bash
# 1. SSH 登录沐曦平台后，克隆项目
git clone <仓库地址> && cd llaisys

# 2. 执行一键部署脚本
chmod +x scripts/deploy_metax.sh
./scripts/deploy_metax.sh ./quantized_model    # 使用 INT8 量化模型
# 或
./scripts/deploy_metax.sh ./quantized_model_int4  # 使用 INT4 量化模型
```

脚本自动完成：环境检测 → 安装 xmake → 编译 llaisys → 安装 Python 包 → 部署 .so → 验证 GPU

### 11.3 运行验证测试

```bash
# 三层验证测试
python3 test/test_metax.py                          # Level 1+2: Runtime + 算子
python3 test/test_metax.py --model ./quantized_model # Level 1+2+3: 含推理
python3 test/test_metax.py --full                    # 全量测试

# 单独测试 Runtime
python3 test/test_runtime.py --device metax

# 端到端推理对比
python3 test/test_infer.py --device metax --model ./quantized_model --test
```

### 11.4 启动推理服务

```bash
cd python

# 启动 Web 服务器
python3 -m server.app --model ../quantized_model --device metax

# 另一个终端：CLI 聊天
python3 -m server.chat_cli --url http://127.0.0.1:8000
```

### 11.5 常见问题排查

| 问题 | 排查方法 |
|------|----------|
| `mx-smi` 无法识别 GPU | 检查驱动是否安装：`lsmod \| grep maca` |
| 编译报错找不到 mxcc | 确认 `/opt/maca/mxgpu_llvm/bin/mxcc` 存在，或检查 `$MACA_PATH` 环境变量 |
| `libmcruntime.so` 找不到 | 设置 `export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH` |
| Runtime test 全部 skip | `getDeviceCount` 返回 0，检查驱动和 GPU 状态 |
| 推理结果全 0 或 NaN | 逐个算子测试定位问题算子 |

---

## 十二、Phase D 实际验证结果（2026-03-10）

### 12.1 验证环境

| 项目 | 实际值 |
|------|--------|
| GPU | MetaX C500, 65536 MiB VRAM |
| 驱动 | 3.0.11 |
| MACA SDK | 3.0.0.8, 路径 `/opt/maca/` |
| mxcc 编译器 | v1.0.0, 路径 `/opt/maca/mxgpu_llvm/bin/mxcc` |
| GPU 架构 | xcore1000 |
| PyTorch | 2.4.0+metax3.0.0.3 |
| Python | 3.10.10 |
| xmake | v3.0.7 |

### 12.2 API 命名差异修正

原计划假设 MACA API 使用 `maca*` 前缀，实际 SDK 使用 `mc*` 前缀。全部修正如下：

| 类别 | 原计划假设 | 实际 API |
|------|-----------|---------|
| Runtime | `macaMalloc`, `macaFree` | `mcMalloc`, `mcFree` |
| 头文件 | `maca_runtime_api.h` | `mc_runtime_api.h`（在 `mcr/` 子目录） |
| BLAS | `mxblasSgemm` | `mcblasGemmEx`, `mcblasSgemmStridedBatched` |
| BLAS 枚举 | `MXBLAS_OP_T` | `MCBLAS_OP_T`, `MCBLAS_OP_N` |
| BLAS 库 | `libmxblas.so` | `libmcblas.so` |
| Runtime 库 | `libmaca_runtime.so` | `libmcruntime.so` |
| BF16 类型 | `__nv_bfloat16` | `__maca_bfloat16` |
| 数据类型 | `CUDA_R_32F` | `MACA_R_32F` |
| 随机数 | `curandState*` | `mcrandState*` |
| SDK 路径 | `/opt/mxmaca/` | `/opt/maca/` |

### 12.3 构建系统修改

1. **文件扩展名**：`.cu` → `.mc`（避免 xmake 自动调用 CUDA 工具链）
2. **自定义编译规则**：`xmake/metax.lua` 使用 `on_build` 手动调用 mxcc 编译 `.mc` 文件
3. **mxcc 编译参数**：`--cuda-gpu-arch=xcore1000 -x maca -fPIC -std=c++17 -O2`
4. **链接策略**：使用 `add_shflags("-Wl,--whole-archive", "libllaisys-ops-metax.a", "-Wl,--no-whole-archive")` 解决链接顺序问题

### 12.4 代码变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/device/metax/metax_runtime_api.cpp` | API 重命名 | `maca*` → `mc*` 全部替换 |
| `src/ops/*/metax/*.mc` (10 个文件) | API 重命名 + 扩展名 | CUDA API → MACA API，`.cu` → `.mc` |
| `xmake/metax.lua` | 重写 | 自定义 `on_build` 编译规则，正确 SDK 路径 |
| `xmake.lua` | 修改 | MetaX 链接配置，`add_shflags` 解决 archive 链接 |
| `include/llaisys/ops.h` | 新增 | `llaisysDequantize`, `llaisysDequantizeInt4` 声明 |
| `src/llaisys/ops.cc` | 新增 | dequantize C wrapper 实现 |
| `python/llaisys/ops.py` | 修复+新增 | `linear` None bias 修复，`dequantize` 方法 |
| `python/llaisys/libllaisys/ops.py` | 新增 | dequantize ctypes 绑定 |

### 12.5 测试结果

**三种模型格式均在 MetaX C500 上通过 17/17 测试：**

| 测试级别 | 测试项 | FP32 | INT8 | INT4 |
|---------|--------|------|------|------|
| Level 1 | getDeviceCount | ✅ | ✅ | ✅ |
| Level 1 | setDevice | ✅ | ✅ | ✅ |
| Level 1 | malloc_device | ✅ | ✅ | ✅ |
| Level 1 | malloc_host | ✅ | ✅ | ✅ |
| Level 1 | memcpy H2D→D2H | ✅ | ✅ | ✅ |
| Level 1 | free_device/host | ✅ | ✅ | ✅ |
| Level 2 | add | ✅ | ✅ | ✅ |
| Level 2 | argmax | ✅ | ✅ | ✅ |
| Level 2 | embedding | ✅ | ✅ | ✅ |
| Level 2 | rms_norm | ✅ | ✅ | ✅ |
| Level 2 | swiglu | ✅ | ✅ | ✅ |
| Level 2 | rope | ✅ | ✅ | ✅ |
| Level 2 | linear | ✅ | ✅ | ✅ |
| Level 2 | dequantize | ✅ | ✅ | ✅ |
| Level 2 | sample | ✅ | ✅ | ✅ |
| Level 2 | self_attention | ✅ | ✅ | ✅ |
| Level 3 | 端到端推理 | ✅ | ✅ | ✅ |

### 12.6 推理性能（DeepSeek-R1-Distill-Qwen-1.5B）

| 模型格式 | 权重大小 | 压缩比 | 吞吐量 (tokens/s) | 输出质量 |
|---------|---------|--------|-------------------|---------|
| FP32 原始 | 3.3 GB | 1.0x | **25.6** | 最佳 |
| INT8 量化 | 2.4 GB | 1.4x | **16.4** | 良好 |
| INT4 量化 | 0.8 GB | 3.8x | **18.6** | 可接受 |

> 注：吞吐量包含 prefill + decode 阶段，测试 prompt 为 "What is 1+1?"，生成 32 新 tokens。

---

## 十三、沐曦平台操作手册（完整步骤）

> 本章提供从零开始在沐曦 C500 算力平台上编译、测试、部署 llaisys 的完整操作指南。
> 无需 AI Agent 辅助，按步骤执行即可。

### 13.1 前提条件

- 已租用沐曦 C500 算力实例（推荐 PyTorch 镜像）
- 已通过 SSH 或 VS Code Remote SSH 连接到实例
- 项目代码已克隆到实例上

### 13.2 环境检查

```bash
# 1. 检查 GPU 是否可用
mx-smi
# 应看到 "MetaX C500" 设备信息，显示显存 65536 MiB

# 2. 检查 MACA SDK
ls /opt/maca/mxgpu_llvm/bin/mxcc
# 应存在 mxcc 编译器

echo $MACA_PATH
# 应输出 /opt/maca

# 3. 检查 Python 和 PyTorch
python3 --version                    # 应 >= 3.8
python3 -c "import torch; print(torch.__version__)"  # 应包含 "metax"

# 4. 检查必要 Python 包
pip list | grep -E "transformers|safetensors|huggingface"
# 应有 transformers, safetensors, huggingface_hub
```

### 13.3 安装 xmake（构建工具）

```bash
# 安装 xmake（如果尚未安装）
curl -fsSL https://xmake.io/shget.text | bash

# 激活 xmake（每次新开终端都需要）
source ~/.xmake/profile

# 验证
xmake --version
```

> **注意：** 如果以 root 用户运行，可能需要设置 `export XMAKE_ROOT=y`。

### 13.4 编译 llaisys（MetaX GPU 后端）

```bash
cd /path/to/llaisys

# 配置构建（启用 MetaX GPU 支持）
xmake f -c --metax-gpu=true

# 编译（使用所有 CPU 核心）
xmake -j$(nproc)

# 预期输出：
# - "compiling.maca src/ops/*/metax/*.mc" 共 10 条
# - "archiving.release libllaisys-ops-metax.a"
# - "linking.release libllaisys.so"
# - "[100%]: build ok"

# 安装共享库
xmake install
# 会将 libllaisys.so 复制到 python/llaisys/libllaisys/ 目录

# 安装 Python 包
pip install ./python/
```

### 13.5 运行测试

#### 基础测试（不需要模型权重）

```bash
# Runtime API 测试 — 测试 GPU 内存分配、拷贝、设备管理
python3 test/test_runtime.py --device metax

# 完整算子测试（Level 1 + Level 2）
python3 test/test_metax.py
# 应看到 16/16 通过（Level 3 跳过）
```

#### 下载模型权重

```bash
# 方式 1：使用 Python 下载（推荐，自动处理所有文件）
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                  local_dir='./models/DeepSeek-R1-Distill-Qwen-1.5B')
"
# 约 3.3 GB，需要几分钟

# 方式 2：如果 HuggingFace 被墙，使用镜像
# export HF_ENDPOINT=https://hf-mirror.com
# 然后再运行上面的命令
```

#### 端到端推理测试

```bash
# 使用 FP32 原始模型（最高精度，约 25 tokens/s）
python3 test/test_metax.py --model ./models/DeepSeek-R1-Distill-Qwen-1.5B

# 应看到 17/17 全部通过
```

### 13.6 模型量化（可选，减小显存占用）

```bash
# INT8 量化（压缩约 1.4x）
python3 scripts/quantize.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output ./models/DeepSeek-R1-Distill-Qwen-1.5B-INT8 \
    --bits 8

# INT4 量化（压缩约 3.8x，适合显存较小的场景）
python3 scripts/quantize.py \
    --model ./models/DeepSeek-R1-Distill-Qwen-1.5B \
    --output ./models/DeepSeek-R1-Distill-Qwen-1.5B-INT4 \
    --bits 4

# 量化后测试
python3 test/test_metax.py --model ./models/DeepSeek-R1-Distill-Qwen-1.5B-INT8
python3 test/test_metax.py --model ./models/DeepSeek-R1-Distill-Qwen-1.5B-INT4
```

### 13.7 启动推理服务器

#### 安装服务器依赖

```bash
pip install uvicorn fastapi sse-starlette
```

#### 启动服务器

```bash
cd python

# 使用 FP32 模型（最高精度）
python3 -m server.app \
    --model ../models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device metax \
    --host 0.0.0.0 \
    --port 8000

# 或使用 INT8 量化模型（节省显存）
python3 -m server.app \
    --model ../models/DeepSeek-R1-Distill-Qwen-1.5B-INT8 \
    --device metax \
    --host 0.0.0.0 \
    --port 8000

# 或使用 INT4 量化模型（最小显存）
python3 -m server.app \
    --model ../models/DeepSeek-R1-Distill-Qwen-1.5B-INT4 \
    --device metax \
    --host 0.0.0.0 \
    --port 8000
```

#### 后台运行（关闭终端不影响）

```bash
cd python
nohup python3 -m server.app \
    --model ../models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device metax \
    --host 0.0.0.0 \
    --port 8000 \
    > ../server.log 2>&1 &

# 查看日志
tail -f ../server.log

# 停止服务器
kill $(pgrep -f "server.app")
```

#### 验证服务器是否正常

```bash
# 查询可用模型
curl http://localhost:8000/v1/models

# 发送聊天请求
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-distill-qwen-1.5b",
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
    "max_tokens": 100
  }'
```

### 13.8 从主机浏览器访问 Web UI

服务器运行在算力平台上，需要将端口映射到本地才能从主机浏览器访问。

#### 方法 1：VS Code 端口转发（推荐，最简单）

如果你通过 **VS Code Remote SSH** 连接到算力平台：

1. VS Code 通常会**自动检测**端口 8000 并提示转发
2. 如果没有自动检测，手动操作：
   - 按 `Ctrl+Shift+P`，输入 **"Forward a Port"**
   - 或在底部面板点击 **"端口 (PORTS)"** 标签页
   - 输入 `8000`，回车
3. 在主机浏览器中打开：**http://localhost:8000**
4. 你会看到一个聊天界面，可以直接与模型对话

#### 方法 2：SSH 端口转发（通用方法）

在你的**本地主机**上打开终端：

```bash
# 语法：ssh -L 本地端口:远程地址:远程端口 用户@服务器
ssh -L 8000:localhost:8000 root@<算力平台IP> -p <SSH端口>

# 示例（假设算力平台 SSH 为 123.45.67.89:22222）
ssh -L 8000:localhost:8000 root@123.45.67.89 -p 22222
```

然后在浏览器中打开 **http://localhost:8000**。

> **注意：** 保持这个 SSH 连接不要关闭，关闭后端口映射也会断开。

#### 方法 3：算力平台自带端口映射

部分算力平台（如 AutoDL、恒源云等）提供自带的端口映射功能：

1. 登录算力平台控制台
2. 找到你的实例 → **端口映射** 或 **网络设置**
3. 添加映射：内部端口 `8000` → 外部端口（平台自动分配）
4. 平台会提供一个公网地址，如 `http://xxx.platform.com:12345`
5. 在浏览器中打开该地址即可

### 13.9 使用 CLI 聊天（无需浏览器）

```bash
cd python

# 确保服务器已在另一个终端运行
python3 -m server.chat_cli --url http://127.0.0.1:8000
# 然后直接在终端中输入消息进行对话
```

### 13.10 使用 OpenAI 兼容 API

服务器提供 OpenAI 兼容的 API，可以用任何支持 OpenAI API 的客户端连接：

```python
# Python 示例
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # 本地运行不需要 API key
)

response = client.chat.completions.create(
    model="deepseek-r1-distill-qwen-1.5b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=200
)
print(response.choices[0].message.content)
```

### 13.11 常见问题排查

| 问题 | 解决方法 |
|------|----------|
| `mx-smi` 无法识别 GPU | 检查驱动：`lsmod \| grep maca`，联系平台方 |
| `xmake` 命令找不到 | 执行 `source ~/.xmake/profile` |
| 编译报 "mxcc not found" | 检查 `/opt/maca/mxgpu_llvm/bin/mxcc` 是否存在 |
| 链接报 `libmcruntime.so` 找不到 | `export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH` |
| `import llaisys` 报错 | 确认已执行 `xmake install && pip install ./python/` |
| 服务器启动后无法访问网页 | 检查端口映射是否设置正确（见 13.8） |
| 推理结果全 0 或 NaN | 运行 `python3 test/test_metax.py` 逐算子排查 |
| HuggingFace 下载超时 | 设置 `export HF_ENDPOINT=https://hf-mirror.com` |
| 重新编译后测试失败 | 重新执行 `xmake install && pip install ./python/` |

### 13.12 完整操作流程速查

```bash
# ===== 一键部署流程 =====
cd /path/to/llaisys

# 1. 环境检查
mx-smi && echo "GPU OK"

# 2. 安装 xmake（仅首次）
curl -fsSL https://xmake.io/shget.text | bash
source ~/.xmake/profile

# 3. 编译 + 安装
xmake f -c --metax-gpu=true
xmake -j$(nproc)
xmake install
pip install ./python/

# 4. 下载模型
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir='./models/DeepSeek-R1-Distill-Qwen-1.5B')"

# 5. 验证测试
python3 test/test_metax.py --model ./models/DeepSeek-R1-Distill-Qwen-1.5B

# 6. 启动服务器
cd python
python3 -m server.app --model ../models/DeepSeek-R1-Distill-Qwen-1.5B --device metax --host 0.0.0.0 --port 8000

# 7. 在主机浏览器打开 http://localhost:8000（需端口映射）
```
