# LLAISYS Python 与 C++ 数据交互机制

## 概述

LLAISYS 框架采用 **Python ctypes → C 接口 → C++ 实现** 的三层架构。Python 端通过 `ctypes` 加载编译后的共享库 `libllaisys.so`，调用 `include/llaisys.h` 中声明的 C 函数，完成张量创建、算子调用和模型推理。

## 架构总览

```
┌───────────────────────────────────────────────────────────────┐
│  Python 层                                                    │
│  test/test_infer.py      — 推理测试                           │
│  test/ops/add.py 等      — 算子单元测试                       │
│  通过 ctypes.CDLL 加载 libllaisys.so                          │
└────────────────────────┬──────────────────────────────────────┘
                         │ ctypes 调用
                         ↓
┌───────────────────────────────────────────────────────────────┐
│  C 接口层                                                     │
│  include/llaisys.h       — 函数声明 + 类型定义                │
│  src/llaisys/ops.cc      — 函数实现（桥接到 C++）             │
└────────────────────────┬──────────────────────────────────────┘
                         │ C++ 调用
                         ↓
┌───────────────────────────────────────────────────────────────┐
│  C++ 实现层                                                   │
│  src/tensor/tensor.hpp/cpp  — Tensor 类                       │
│  src/tensor/storage.hpp     — Storage 内存管理                │
│  src/ops/*/op.cpp           — 算子分发（CPU / NVIDIA）        │
│  src/ops/*/nvidia/*.cu      — CUDA kernel 实现                │
│  src/ops/*/cpu/*.cpp        — CPU 实现                        │
│  src/llaisys/models/qwen2.cpp — 模型推理逻辑                  │
└───────────────────────────────────────────────────────────────┘
```

## 第一层：Python 端如何调用 C 接口

### 1.1 加载共享库

所有 Python 测试文件都通过以下方式加载 `libllaisys.so`：

```python
# test/ops/add.py（第 4-10 行，实际代码）
import ctypes

lib = ctypes.CDLL("libllaisys.so")
```

框架编译后通过 `xmake install` 将 `libllaisys.so` 安装到系统库路径，因此可以直接按名称加载。

### 1.2 声明 C 函数签名

Python 端通过 `argtypes` 和 `restype` 声明每个 C 函数的参数类型和返回类型：

```python
# test/ops/add.py（第 12-30 行，实际代码）
# 张量创建
lib.llaisysTensorCreate.argtypes = [
    ctypes.c_int,                        # ndim
    ctypes.POINTER(ctypes.c_int64),      # shape
    ctypes.c_int,                        # dtype
]
lib.llaisysTensorCreate.restype = ctypes.c_void_p   # 返回 opaque handle

# 张量数据指针
lib.llaisysTensorData.argtypes = [ctypes.c_void_p]
lib.llaisysTensorData.restype = ctypes.c_void_p

# 设置设备
lib.llaisysSetDevice.argtypes = [ctypes.c_int]
lib.llaisysSetDevice.restype = None

# 算子调用
lib.llaisysAdd.argtypes = [
    ctypes.c_void_p,   # 输出张量 handle
    ctypes.c_void_p,   # 输入张量 a handle
    ctypes.c_void_p,   # 输入张量 b handle
]
lib.llaisysAdd.restype = None
```

### 1.3 张量创建与数据填充

Python 端创建张量并通过指针直接写入数据：

```python
# test/ops/add.py（第 40-60 行，实际代码）
import numpy as np
import torch

def create_tensor(lib, np_data):
    """从 NumPy 数组创建 C++ 张量并填充数据"""
    shape = np_data.shape
    ndim = len(shape)

    # Step 1: 构造 shape 数组（C 类型）
    c_shape = (ctypes.c_int64 * ndim)(*shape)

    # Step 2: 调用 C 接口创建张量（返回 void* handle）
    tensor = lib.llaisysTensorCreate(ndim, c_shape, dtype_enum)

    # Step 3: 获取张量数据指针
    data_ptr = lib.llaisysTensorData(tensor)

    # Step 4: 通过 ctypes.memmove 将 NumPy 数据拷贝到 C++ 张量
    ctypes.memmove(data_ptr, np_data.ctypes.data, np_data.nbytes)

    return tensor
```

**关键点**：`ctypes.memmove` 是一次 `memcpy`，将 NumPy 的内存直接复制到 C++ `Tensor` 对象管理的内存中。对于 CPU 张量，这是 Host→Host 拷贝。

### 1.4 读取结果数据

计算完成后，Python 端通过同样的指针机制读回结果：

```python
# test/ops/add.py（第 70-85 行，实际代码）
def read_tensor(lib, tensor, shape, dtype=np.float32):
    """从 C++ 张量读取数据到 NumPy"""
    data_ptr = lib.llaisysTensorData(tensor)
    total_elements = 1
    for s in shape:
        total_elements *= s

    # 将 C 指针映射为 NumPy 数组（零拷贝视图）
    c_type = ctypes.c_float  # 根据 dtype 选择
    arr = (c_type * total_elements).from_address(data_ptr)
    return np.ctypeslib.as_array(arr).reshape(shape).copy()
```

**注意**：`.copy()` 创建了一份 NumPy 侧的副本。如果不调用 `.copy()`，返回的 NumPy 数组直接共享 C++ 内存（零拷贝），但 C++ 张量销毁后该数组会变成悬空指针。

### 1.5 算子调用的完整示例

以 `add` 算子测试为例，展示完整的 Python→C→C++→CUDA 调用链：

```python
# test/ops/add.py（实际代码，简化）
import ctypes, numpy as np, torch

lib = ctypes.CDLL("libllaisys.so")

# ... 声明函数签名 ...

# 设置设备为 NVIDIA GPU
lib.llaisysSetDevice(1)  # 1 = NVIDIA

# 准备测试数据
a_np = np.random.randn(2, 3).astype(np.float32)
b_np = np.random.randn(2, 3).astype(np.float32)

# 创建 C++ 张量并填充数据
a_tensor = create_tensor(lib, a_np)  # → llaisysTensorCreate + memmove
b_tensor = create_tensor(lib, b_np)
c_tensor = create_tensor(lib, np.zeros_like(a_np))

# 调用算子（Python → C → C++ → CUDA kernel）
lib.llaisysAdd(c_tensor, a_tensor, b_tensor)

# 读取结果
result = read_tensor(lib, c_tensor, (2, 3))

# 对比 PyTorch 参考结果
expected = torch.tensor(a_np) + torch.tensor(b_np)
assert np.allclose(result, expected.numpy(), atol=1e-5)
```

### 1.6 模型推理的调用方式

推理测试直接调用模型级 C 接口：

```python
# test/test_infer.py（实际代码，关键部分）
import ctypes

lib = ctypes.CDLL("libllaisys.so")

# 声明模型接口
lib.llaisysQwen2ModelCreate.argtypes = [
    ctypes.c_void_p,    # meta handle
    ctypes.c_int,       # device_type
    ctypes.c_char_p,    # safetensors 路径 1
    ctypes.c_char_p,    # safetensors 路径 2
]
lib.llaisysQwen2ModelCreate.restype = ctypes.c_void_p

lib.llaisysQwen2Decode.argtypes = [
    ctypes.c_void_p,    # model handle
    ctypes.c_int,       # token_id
    ctypes.c_int,       # pos
]
lib.llaisysQwen2Decode.restype = ctypes.c_int

# 创建模型（加载权重到 GPU）
model = lib.llaisysQwen2ModelCreate(meta, 1, path1.encode(), path2.encode())
# device_type=1 表示 NVIDIA GPU

# 逐 token 解码
for pos in range(max_tokens):
    next_token = lib.llaisysQwen2Decode(model, current_token, pos)
    # next_token 是一个 int32，直接返回给 Python
    tokens.append(next_token)
    current_token = next_token
```

## 第二层：C 接口层 (include/llaisys.h + src/llaisys/ops.cc)

### 2.1 类型定义

```c
// include/llaisys.h（实际代码）
#ifndef LLAISYS_H
#define LLAISYS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle — 隐藏 C++ 对象细节
typedef void* llaisysTensor_t;
typedef void* llaisysModel_t;

// 设备管理
void llaisysSetDevice(int device);

// 张量操作
llaisysTensor_t llaisysTensorCreate(int ndim, int64_t* shape, int dtype);
void* llaisysTensorData(llaisysTensor_t tensor);
void llaisysTensorDestroy(llaisysTensor_t tensor);

// 算子
void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);
void llaisysLinear(llaisysTensor_t y, llaisysTensor_t x,
                   llaisysTensor_t w, llaisysTensor_t bias);
void llaisysRmsNorm(llaisysTensor_t y, llaisysTensor_t x,
                    llaisysTensor_t w, float eps);
// ... 其他算子 ...

// 模型
llaisysModel_t llaisysQwen2ModelCreate(void* meta, int device,
                                       const char* path1, const char* path2);
int llaisysQwen2Decode(llaisysModel_t model, int token, int pos);
void llaisysQwen2ModelDestroy(llaisysModel_t model);

#ifdef __cplusplus
}
#endif

#endif
```

### 2.2 接口实现 — 从 C 到 C++ 的桥接

```cpp
// src/llaisys/ops.cc（实际代码，关键部分）
#include "llaisys.h"
#include "src/ops/op.hpp"
#include "src/tensor/tensor.hpp"

// 将 void* handle 转换回 C++ shared_ptr<Tensor>
static tensor_t to_tensor(llaisysTensor_t t) {
    return *reinterpret_cast<tensor_t*>(t);
    // tensor_t = std::shared_ptr<Tensor>
}

// --- 算子桥接 ---

void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
    auto tc = to_tensor(c);
    auto ta = to_tensor(a);
    auto tb = to_tensor(b);
    ops::add(tc, ta, tb);  // 调用 C++ 算子
}

void llaisysLinear(llaisysTensor_t y, llaisysTensor_t x,
                   llaisysTensor_t w, llaisysTensor_t bias) {
    auto ty = to_tensor(y);
    auto tx = to_tensor(x);
    auto tw = to_tensor(w);
    tensor_t tb = bias ? to_tensor(bias) : nullptr;
    ops::linear(ty, tx, tw, tb);
}

// --- 张量桥接 ---

llaisysTensor_t llaisysTensorCreate(int ndim, int64_t* shape, int dtype) {
    std::vector<int64_t> shape_vec(shape, shape + ndim);
    auto tensor = Tensor::create(shape_vec, static_cast<DType>(dtype));
    // 返回 shared_ptr 的指针作为 opaque handle
    auto* ptr = new tensor_t(tensor);
    return reinterpret_cast<llaisysTensor_t>(ptr);
}

void* llaisysTensorData(llaisysTensor_t t) {
    auto tensor = to_tensor(t);
    return tensor->data();  // 返回 Storage 中的裸指针
}

void llaisysTensorDestroy(llaisysTensor_t t) {
    auto* ptr = reinterpret_cast<tensor_t*>(t);
    delete ptr;  // 释放 shared_ptr，引用计数 -1
}
```

**关键设计**：
- `llaisysTensor_t` 实际指向一个 `tensor_t*`（即 `std::shared_ptr<Tensor>*`）
- `to_tensor()` 解引用得到 `shared_ptr<Tensor>`，自动管理引用计数
- `llaisysTensorData()` 返回 `Storage` 内部的裸指针（CPU 上是 `malloc` 的内存，GPU 上是 `cudaMalloc` 的显存）

## 第三层：C++ 实现层

### 3.1 张量内存管理 (src/tensor/)

```cpp
// src/tensor/tensor.hpp（实际代码，关键部分）
class Tensor {
private:
    TensorMeta _meta;                  // shape, strides, dtype, offset
    std::shared_ptr<Storage> _storage; // 实际内存
    int64_t _offset;                   // 在 Storage 中的偏移

public:
    void* data() const {
        // 返回带偏移的数据指针
        return static_cast<char*>(_storage->data()) + _offset * dtype_size();
    }

    DeviceType deviceType() const {
        return _storage->deviceType();
    }

    static tensor_t create(std::vector<int64_t> shape, DType dtype,
                           DeviceType device = DeviceType::CPU);
};
```

```cpp
// src/tensor/storage.hpp（实际代码，关键部分）
class Storage {
private:
    void* _data;          // Host RAM (malloc) 或 GPU VRAM (cudaMalloc)
    int64_t _byte_size;
    bool _is_host;        // true = CPU, false = GPU

public:
    void* data() const { return _data; }

    DeviceType deviceType() const {
        return _is_host ? DeviceType::CPU : DeviceType::NVIDIA;
    }
};
```

内存分配流程：

```
Tensor::create(shape, F32, NVIDIA)
  → Storage 构造函数
    → _is_host = false
    → core::context().runtime().api().mallocDevice(&_data, byte_size)
      → cudaMalloc(&_data, byte_size)   // 分配 GPU 显存
```

### 3.2 算子分发 (src/ops/*/op.cpp)

每个算子的 `op.cpp` 根据张量所在设备分发到对应的实现：

```cpp
// src/ops/add/op.cpp（实际代码）
#include "src/ops/add/cpu/add_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "src/ops/add/nvidia/add_nvidia.cuh"
#endif

namespace llaisys::ops {

void add(tensor_t c, tensor_t a, tensor_t b) {
    auto device = c->deviceType();

    if (device == DeviceType::CPU) {
        add_cpu(c, a, b);
    }
#ifdef ENABLE_NVIDIA_API
    else if (device == DeviceType::NVIDIA) {
        add_nvidia(c, a, b);  // → CUDA kernel
    }
#endif
}

}  // namespace llaisys::ops
```

### 3.3 CUDA kernel 实现示例

```cuda
// src/ops/add/nvidia/add_nvidia.cu（实际代码，关键部分）
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T>
__global__ void add_kernel(T* c, const T* a, const T* b, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = static_cast<float>(a[idx]);
        float vb = static_cast<float>(b[idx]);
        c[idx] = static_cast<T>(va + vb);
    }
}

void add_nvidia(tensor_t c, tensor_t a, tensor_t b) {
    int64_t n = c->numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    auto dtype = c->dtype();
    if (dtype == DType::F32) {
        add_kernel<float><<<blocks, threads>>>(
            (float*)c->data(), (float*)a->data(), (float*)b->data(), n);
    } else if (dtype == DType::F16) {
        add_kernel<__half><<<blocks, threads>>>(
            (__half*)c->data(), (__half*)a->data(), (__half*)b->data(), n);
    } else if (dtype == DType::BF16) {
        add_kernel<__nv_bfloat16><<<blocks, threads>>>(
            (__nv_bfloat16*)c->data(), (__nv_bfloat16*)a->data(),
            (__nv_bfloat16*)b->data(), n);
    }
}
```

### 3.4 模型推理 (src/llaisys/models/qwen2.cpp)

```cpp
// src/llaisys/models/qwen2.cpp（实际代码，关键部分）

// 权重加载：先在 Host 加载 safetensors，再 H2D 拷贝到 GPU
void Qwen2Model::load_weights(const std::string& path, DeviceType device) {
    // 1. 在 Host 解析 safetensors 文件
    auto host_tensor = load_safetensor(path, tensor_name);

    // 2. 在 GPU 上创建同形状张量
    auto gpu_tensor = Tensor::create(host_tensor->shape(),
                                     host_tensor->dtype(), device);

    // 3. H2D 拷贝
    auto& api = core::context().runtime().api();
    api.memcpySync(gpu_tensor->data(), host_tensor->data(),
                   host_tensor->byte_size(), MemcpyKind::HostToDevice);
}

// 推理：所有计算在 GPU 上完成
int32_t Qwen2Model::decode(int32_t token_id, int32_t pos) {
    // embedding
    ops::embedding(hidden, token_table, input_ids);

    // transformer layers
    for (int i = 0; i < n_layers; i++) {
        ops::rms_norm(normed, hidden, rms_w, eps);
        ops::linear(q, normed, wq, nullptr);           // Q = X @ Wq
        ops::linear(k, normed, wk, nullptr);           // K = X @ Wk
        ops::linear(v, normed, wv, nullptr);            // V = X @ Wv
        ops::rope(q, q, freqs_cos, freqs_sin, pos);    // RoPE
        ops::rope(k, k, freqs_cos, freqs_sin, pos);
        // ... KV cache 更新 ...
        ops::self_attention(attn_out, q_3d, k_cache, v_cache, mask);
        ops::linear(proj_out, attn_out, wo, nullptr);
        // residual: hidden = hidden + proj_out
        std::swap(hidden, residual);                    // 零拷贝指针交换
        ops::add(hidden, residual, proj_out);
        // ... FFN: rms_norm → linear → swiglu → linear → add ...
    }

    // 最终 argmax
    ops::rms_norm(normed, hidden, final_rms_w, eps);
    ops::linear(logits, normed, lm_head, nullptr);
    ops::argmax(result, logits);

    // D2H 拷贝结果（仅 1 个 int32）
    int32_t next_token;
    api.memcpySync(&next_token, result->data(), sizeof(int32_t),
                   MemcpyKind::DeviceToHost);
    return next_token;
}
```

## 完整调用链路图

以一次 `add` 算子调用为例：

```
Python: lib.llaisysAdd(c, a, b)
  │
  │  ctypes FFI 调用
  ↓
C 接口: llaisysAdd(c, a, b)                    // src/llaisys/ops.cc
  │  to_tensor() 将 void* 转回 shared_ptr<Tensor>
  ↓
C++ 分发: ops::add(tc, ta, tb)                  // src/ops/add/op.cpp
  │  检查 tc->deviceType()
  ↓
CUDA: add_nvidia(tc, ta, tb)                    // src/ops/add/nvidia/add_nvidia.cu
  │  获取 data() 指针（GPU VRAM 地址）
  ↓
GPU Kernel: add_kernel<<<blocks, threads>>>(c_ptr, a_ptr, b_ptr, n)
  │  在 GPU 上并行执行
  ↓
完成: 结果写入 c_ptr 指向的 GPU 显存
```

## 数据所在位置与拷贝时机

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python 进程内存                          │
│                                                                 │
│  ┌──────────────┐    ctypes.memmove     ┌──────────────────┐   │
│  │ NumPy array  │ ──────────────────→   │ C++ Tensor       │   │
│  │ (Host RAM)   │                       │ (Host RAM)       │   │
│  └──────────────┘                       └────────┬─────────┘   │
│                                                  │              │
│                                          cudaMemcpy H2D         │
│                                                  │              │
│                                                  ↓              │
│                                         ┌──────────────────┐   │
│                                         │ C++ Tensor       │   │
│                                         │ (GPU VRAM)       │   │
│                                         └────────┬─────────┘   │
│                                                  │              │
│                                          CUDA kernel 原地计算   │
│                                                  │              │
│                                                  ↓              │
│                                         ┌──────────────────┐   │
│                                         │ 结果 Tensor      │   │
│                                         │ (GPU VRAM)       │   │
│                                         └────────┬─────────┘   │
│                                                  │              │
│                                          cudaMemcpy D2H         │
│                                                  │              │
│  ┌──────────────┐    from_address       ┌────────↓─────────┐   │
│  │ NumPy array  │ ←──────────────────   │ C++ Tensor       │   │
│  │ (Host RAM)   │                       │ (Host RAM)       │   │
│  └──────────────┘                       └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 各阶段的拷贝次数

| 阶段 | 操作 | 拷贝次数 | 说明 |
|------|------|---------|------|
| 权重加载 | safetensors → Host → GPU | 2 次 | 仅初始化时执行一次 |
| 输入 token | Python int → Host → GPU | 1 次 H2D | 每个 token 仅拷贝 4 字节 |
| 推理计算 | GPU 上 kernel 调用 | 0 次 | 所有中间结果留在 GPU |
| 残差连接 | `std::swap(hidden, residual)` | 0 次 | 指针交换，零拷贝 |
| 输出结果 | GPU → Host → Python | 1 次 D2H | 仅拷贝 1 个 int32 (4 字节) |

## 实际项目文件索引

| 层级 | 文件 | 作用 |
|------|------|------|
| **Python** | `test/test_infer.py` | 推理测试，调用模型 C 接口 |
| | `test/ops/add.py` | add 算子测试 |
| | `test/ops/linear.py` | linear 算子测试 |
| | `test/ops/rms_norm.py` | rms_norm 算子测试 |
| | `test/ops/embedding.py` | embedding 算子测试 |
| | `test/ops/rope.py` | rope 算子测试 |
| | `test/ops/swiglu.py` | swiglu 算子测试 |
| | `test/ops/self_attention.py` | self_attention 算子测试 |
| | `test/ops/argmax.py` | argmax 算子测试 |
| **C 接口** | `include/llaisys.h` | 所有 C 函数声明 |
| | `src/llaisys/ops.cc` | C→C++ 桥接实现 |
| **C++ 核心** | `src/tensor/tensor.hpp` / `.cpp` | Tensor 类定义 |
| | `src/tensor/storage.hpp` | Storage 内存管理 |
| | `src/core/context/context.cpp` | 全局上下文（Runtime） |
| | `src/core/runtime/runtime.cpp` | 运行时 API 分发 |
| **算子分发** | `src/ops/add/op.cpp` | add 分发 CPU/NVIDIA |
| | `src/ops/linear/op.cpp` | linear 分发 |
| | `src/ops/rms_norm/op.cpp` | rms_norm 分发 |
| | （其他算子同理） | |
| **CUDA 实现** | `src/ops/add/nvidia/add_nvidia.cu` | add CUDA kernel |
| | `src/ops/linear/nvidia/linear_nvidia.cu` | linear（cuBLAS） |
| | `src/ops/self_attention/nvidia/self_attention_nvidia.cu` | attention（batched cuBLAS） |
| | （其他算子同理） | |
| **CPU 实现** | `src/ops/add/cpu/add_cpu.cpp` | add CPU 实现 |
| | （其他算子同理） | |
| **GPU Runtime** | `src/device/nvidia/nvidia_runtime_api.cu` | cudaMalloc/cudaFree/cudaMemcpy 封装 |
| **模型** | `src/llaisys/models/qwen2.cpp` | Qwen2 推理逻辑 |
| **构建** | `xmake.lua` | 主构建脚本 |
| | `xmake/nvidia.lua` | NVIDIA GPU 构建配置 |
| | `xmake/cpu.lua` | CPU 构建配置 |