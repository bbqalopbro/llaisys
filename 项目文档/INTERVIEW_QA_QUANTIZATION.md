# 面试深挖记录：极致显存压缩与量化落地

> **简历原文**：极致显存压缩与量化落地：实现 INT8/INT4 Weight-Only 量化链路，手写高效 INT4 反量化 CUDA Kernel（实现 3.76 倍显存压缩）；在权重加载端无缝兼容 GPTQ/AWQ 格式，实现底层 C++ 算子的零侵入式量化推理。

---

## 第一轮：量化方案选型与基本原理

### Q1: Weight-Only 量化 vs 全量化（W8A8）

**问题**：为什么选择 Weight-Only 量化而不对激活值也做量化？

**标准答案**：

- **WO 优势**：
  - 权重是静态的，量化一次即可；激活值每次推理都变化，量化开销大
  - 小 batch 推理（尤其 batch=1 的 decode 阶段）是 **memory-bound**，瓶颈在于把权重从显存搬到计算单元。WO 量化把权重从 FP16 压到 INT4，**显存带宽需求降了 4 倍**，所以反而比 W8A8 更快
  - 实现简单，不需要校准激活值的量化范围

- **WO 劣势**：
  - 反量化后仍然用 FP32/FP16 做 GEMM，**无法利用 INT8 Tensor Core**
  - 大 batch / prefill 阶段是 **compute-bound**，此时 W8A8 可以利用 INT8 Tensor Core 获得更高吞吐

- **关键结论**：batch=1 decode 是 memory-bound → WO 更快；大 batch prefill 是 compute-bound → W8A8 更快

---

### Q2: 3.76 倍压缩率怎么算？

**问题**：FP32→INT4 理论 8 倍，FP16→INT4 理论 4 倍，3.76 怎么来的？

**标准答案（必须能当场算）**：

| 项目 | 字节/参数 |
|------|----------|
| 原始 FP16 权重 | 2.0 |
| INT4 量化权重 (4 bit) | 0.5 |
| Scale (per-group, group_size=128, FP32) | 4 / 128 = 0.03125 |
| **总计** | **0.53125** |

**压缩率 = 2.0 / 0.53125 ≈ 3.76 倍**（基准是 FP16，不是 FP32）

实际项目数据验证（quant_config.json）：
- FP32 总字节：3,087,138,816
- 量化后总字节：820,021,248
- 从 FP32 压缩：3,087,138,816 / 820,021,248 ≈ 3.76（恰好因为 FP32 baseline 和辅助数据占比吻合）

⚠️ **注意**：面试时务必说清楚 baseline 是 FP16 还是 FP32。

---

### Q3: INT8 与 INT4 量化粒度为什么不同？

**问题**：INT8 用 per-channel，INT4 用 per-group，为什么？

**标准答案**：

- **INT8** 有 256 个可表示值（-128~127），表示能力足够，per-channel（每行一个 scale）精度已经可以接受
- **INT4** 只有 16 个可表示值（-8~7），表示能力极弱。如果用 per-channel，一行里几千个元素只共享一个 scale，极端值会把 scale 撑大，让大部分正常值的量化精度很差
- **per-group**（group_size=128）是更细的粒度：每 128 个元素独立计算一个 scale，能更好地适应权重分布的局部特征
- group_size=128 是业界标准平衡点：太大精度不够，太小辅助数据开销大

**易错点**：per-group 比 per-channel **更细**，不是更粗。

---

### Q4: 为什么用对称量化？

**问题**：对称 vs 非对称，你怎么选的？为什么？

**标准答案**：

- **对称量化**：zero-point = 0，公式 `q = round(w / scale)`，反量化 `w ≈ q × scale`
- **非对称量化**：有 zero-point，公式 `q = round(w / scale) + zp`，反量化 `w ≈ (q - zp) × scale`

**选对称的原因**：
1. 反量化只需**一次乘法**，非对称需要一次减法+一次乘法，计算量多一步
2. 不需要存 zero-point 向量，**节省存储**
3. LLM 的权重分布通常近似关于 0 对称（正态分布），对称量化的表示效率足够

**劣势**：当权重分布严重偏移（如全正或偏态）时，对称量化会浪费约一半表示范围。但实际 LLM 中这种情况很少。

---

## 第二轮：GPTQ/AWQ 格式兼容

### Q1: GPTQ 和 AWQ 的算法原理区别

**问题**：这两种方法在减少量化误差上做了什么？

**标准答案**：

| | GPTQ | AWQ |
|--|------|-----|
| **核心思想** | 二阶误差补偿 | 激活感知等价变换 |
| **方法** | 基于 Hessian 逐列量化，量化一列后把误差补偿到后续未量化列 | 观察激活值分布找出重要权重通道，量化前放大、量化后缩小 |
| **理论基础** | 最优脑量化（OBQ）/ 层级重建 | 等价变换不改变输出 |
| **精度** | 依赖权重间相关性 | 依赖激活值统计 |

**面试加分说法**：GPTQ 是"训练后量化的逐层重建（layer-wise reconstruction）"；AWQ 是"激活感知的等价变换（equivalent transformation）"。

---

### Q2: 存储格式（打包方向）差异

**问题**：GPTQ 和 AWQ 的 safetensors 里 qweight 有什么区别？

**标准答案**：

| | GPTQ | AWQ |
|--|------|-----|
| **打包维度** | 输入维度（行向） | 输出维度（列向） |
| **qweight shape** | `[in_features//8, out_features]` int32 | `[in_features, out_features//8]` int32 |
| **bit shifts** | 顺序：`[0,4,8,12,16,20,24,28]` | 交错：`[0,16,4,20,8,24,12,28]` |
| **zero-point 偏移** | +1（AutoGPTQ 约定） | 无偏移 |

**关键区别**：打包方向完全不同，GPTQ 沿输入维压缩，AWQ 沿输出维压缩。AWQ 的交错 bit 排列是为了优化 GPU warp 级内存访问。

---

### Q3: 自动检测模型格式

**问题**：用户怎么无感知地加载不同格式？

**标准答案**：

加载时按以下优先级自动检测：
1. 读取模型目录下的 `quantize_config.json` 文件
2. 如果没有，读取 `config.json` 中的 `quantization_config` 字段
3. 检查 `quant_method` 字段值：
   - `"gptq"` → 走 GPTQ 加载分支
   - `"awq"` → 走 AWQ 加载分支
   - 自有格式标识 → 走 INT8/INT4 native 分支
4. 用户接口不需要指定量化类型，**一个 `--model` 路径搞定**

---

### Q4: 兼容时遇到的最大的坑

**问题**：实际遇到什么技术难题？

**标准答案（两个真实踩坑）**：

**坑1：GPTQ 零点 +1 偏移**
- AutoGPTQ 存储零点时做了 +1 偏移（历史约定）
- 反量化时如果忘了加这个 1，整层偏置错误，输出直接乱码
- 排查花了很多时间，最终通过对比 AutoGPTQ 源码发现

**坑2：AWQ 双重量化**
- 最初尝试：AWQ INT4 → 解包为 FP32 → 用自有 INT4 方案重新量化 → 推理
- 这导致**双重量化误差叠加**：单层相对误差 9.4%，28 层后信号几乎全丢（6.3% 保留率）
- 解决方案：放弃二次量化，AWQ 权重直接解包为 FP16 送入推理（精度几乎无损），或者保持原始 INT32 packed 格式用 GPU kernel 动态反量化

---

## 第三轮：零侵入式量化推理架构

### Q1: "零侵入"是什么意思？

**标准答案**：

**零侵入** = 模型推理的前向传播代码（attention、FFN 等）不需要为了支持量化做任何修改。无论权重是 FP32、FP16、INT8 还是 INT4，模型层代码调用的都是同一个 linear 接口。

**对比"有侵入"的做法**：在每个 Transformer 层代码里写 `if quant_type == INT4: ... elif quant_type == INT8: ... else: ...`，把量化逻辑散落在模型各处——这就是侵入式的，维护成本高、改一处量化方案要改 N 处代码。

---

### Q2: 零侵入的架构边界

**标准答案**：

零侵入发生在 **C++ 模型层 ↔ C++ 算子层** 之间。

中间有一个适配函数 `linear_maybe_dequant`（定义在 `qwen2.cpp` 第 276 行）：
- 它检查权重的 `dtype`：
  - `LLAISYS_DTYPE_I32 + qzeros` → AWQ 路径：`dequantize_awq_int4()` + `linear()`
  - `LLAISYS_DTYPE_I8` → INT8 路径：`dequantize()` + `linear()`
  - `LLAISYS_DTYPE_U8` → INT4 路径：`dequantize_int4()` + `linear()`
  - 其他 → FP32/FP16 直接 `linear()`
- 模型前向代码只调用 `linear_maybe_dequant`，**完全不感知量化的存在**

**代码证据**（qwen2.cpp 第 481-483 行）：
```cpp
// QKV Linear — 无论什么精度，写法完全一致
model->linear_maybe_dequant(model->q, model->norm_out, 
    weights.attn_q_w[i], weights.attn_q_w_scale[i], 
    weights.attn_q_b[i], weights.attn_q_w_qzeros[i]);
```

---

### Q3: 天然支持混合精度

**标准答案**：

因为每层权重各自携带 dtype 标记，`linear_maybe_dequant` 逐层独立判断走哪条路径。同一模型中：
- 第 0 层可以是 INT4
- 第 1 层可以是 INT8
- 第 2 层可以是 FP16

这正是零侵入设计的附加好处，天然支持混合精度。

---

### Q4: 零侵入方案的性能代价

**标准答案**：

**核心代价**：反量化和 GEMM 是两个独立 kernel launch：
```cpp
ops::dequantize_int4(dq_buf, w, sc, group_size);  // Kernel 1: 写 FP32 到显存
ops::linear(out, in, dq_buf, b);                    // Kernel 2: 从显存读 FP32
```

中间 `dq_buf` 需要完整地写入显存再读出，产生额外的**全矩阵显存搬运**。

**量化代价估算**（以 `[1536, 8960]` 权重为例）：
- 反量化输出：1536 × 8960 × 4 bytes ≈ 52MB
- 一次写 + 一次读 ≈ 104MB 额外带宽
- 28 层 × 7 个 linear ≈ **约 20GB 额外显存搬运**

**对比 fused dequant-GEMM**：
- 在 GEMM Kernel 的 tile 加载权重时，直接在 **寄存器/shared memory 中反量化**
- 不需要把完整 FP32 矩阵写回显存
- 这是 TensorRT-LLM、vLLM 的做法

**Trade-off**：零侵入方案牺牲部分性能，换来了实现简单、可维护性好、支持热插拔任意量化格式。

---

### Dequant 缓冲区设计

```cpp
// key = (rows << 32) | cols，按 shape 复用缓冲区
tensor_t get_dequant_buf(size_t rows, size_t cols) {
    uint64_t key = ((uint64_t)rows << 32) | (uint64_t)cols;
    auto it = dequant_cache.find(key);
    if (it != dequant_cache.end()) return it->second;
    auto buf = Tensor::create({rows, cols}, FP32, device_type, device_id);
    dequant_cache[key] = buf;
    return buf;
}
```

- **Lazy 分配**：首次遇到新 shape 才创建缓冲区，避免初始化 OOM
- **形状复用**：相同 shape 的权重共享缓冲区，减少显存占用
- **设备感知**：自动在正确的设备上分配

---

## 第四轮：CUDA Kernel 设计

### Q1: INT4 数据打包方式

**标准答案**：

量化值范围 `-8 ~ +7`（有符号 4-bit），打包时：
1. **加 8 偏移**，变成无符号 `0~15`
2. 偶数列值存 **低 4 位**：`(val_even + 8) & 0xF`
3. 奇数列值存 **高 4 位**：`(val_odd + 8) << 4`
4. 打包公式：`byte = ((val_odd + 8) << 4) | ((val_even + 8) & 0xF)`

**解包**（反量化时）：
- 偶数值：`(byte & 0x0F) - 8`
- 奇数值：`(byte >> 4) - 8`

---

### Q2: Kernel 并行策略

**标准答案**：

```cpp
const int threads = 256;  // 每 block 256 线程
const int blocks = (rows * packed_cols + 255) / 256;  // 按 packed 字节总数算
```

- 每个线程处理 **1 个 packed byte**，输出 **2 个 FP32** 值
- 线程总数 = rows × packed_cols（即 packed 矩阵的元素数）
- 256 线程/块是经典配置，保证 SM 占用率
- 相比 INT8 kernel（1 线程 1 元素），**元素效率翻倍**

---

### Q3: 完整反量化步骤（每线程 7 步）

```
① tid = blockIdx.x * blockDim.x + threadIdx.x  // 全局线程 ID
② row = tid / packed_cols;  pc = tid % packed_cols  // 定位行和 packed 列
③ byte = weight[tid]  // 读取 1 个 packed 字节
④ val_even = (byte & 0x0F) - 8  // 解包低 4 位，减 8 还原有符号
⑤ val_odd  = (byte >> 4) - 8  // 解包高 4 位，减 8 还原有符号
⑥ group_even = col_even / group_size  // 计算所属 group 索引
⑦ out[row, col_even] = float(val_even) × scale[row, group_even]  // 反量化输出
   out[row, col_odd]  = float(val_odd) × scale[row, group_odd]
```

---

### Q4: Scale 的 per-group 寻址

**标准答案**：

Scale 张量是 2D：`[rows, num_groups]`，其中 `num_groups = cols / group_size`。

```cpp
const float* srow = scale + row * num_groups;  // 当前行的 scale 基址
int64_t group_idx = col / group_size;           // 列号 ÷ group_size = 组索引
float s = srow[group_idx];                      // 查表得到对应 scale
```

---

### Q5: 性能瓶颈分析

**标准答案**：

该 kernel 是 **带宽 bound（memory-bound）** 的。

**分析**：
- 每线程计算量：位操作 × 几次 + 浮点乘法 × 2 ≈ ~10 FLOP
- 每线程访存量：读 1 byte + 读 ~1 float scale ≈ 5 bytes；写 2 floats = 8 bytes
- **算术强度** ≈ 10 FLOP / 13 bytes ≈ 0.77 FLOP/byte
- A100 的计算/带宽平衡点 ≈ ~100 FLOP/byte
- 远低于平衡点 → **带宽瓶颈**

**优化方向**（面试加分）：
1. **向量化访存**：用 `float4` / `uint4` 一次读写 16 字节，减少访存指令数
2. **Shared memory 预加载 scale**：同一 group 的多个线程共享 scale，先加载到 shared memory
3. **Warp 级协同**：利用 `__shfl_sync` 在 warp 内广播 scale
4. **终极优化**：Fuse 到 GEMM kernel 内，反量化在 register/shared memory 完成，避免写出 FP32 矩阵

---

## AWQ CUDA Kernel 补充

AWQ kernel 与自有 INT4 kernel 有重要区别：

| | 自有 INT4 Kernel | AWQ Kernel |
|--|-----------------|------------|
| 打包格式 | uint8 (2×4-bit) | int32 (8×4-bit) |
| 每线程输出 | 2 个 FP32 | **8 个 FP32** |
| 量化类型 | 对称（无 zero-point） | 非对称（有 zero-point） |
| Bit shifts | 无（位掩码） | 交错：`[0,16,4,20,8,24,12,28]` |
| 输出布局 | 正常 `[row, col]` | **转置** `[out_col, in_idx]` |
| 循环展开 | 无 | `#pragma unroll` 8 次 |

AWQ kernel 的转置写入 `out[out_col][in_idx]` 是为了让后续 `ops::linear` 按行优先访问权重矩阵时获得连续内存访问。

---

## 薄弱点清单

| 编号 | 薄弱点 | 严重程度 | 备注 |
|------|--------|---------|------|
| 1 | 不知道 memory-bound vs compute-bound 对 WO 量化的影响 | ⚠️ 高频考点 | 已讲解 |
| 2 | 3.76 倍压缩率算不出来 | ⚠️ 必须能当场算 | 2.0 / 0.53125 ≈ 3.76 |
| 3 | group_size 答错（说 4，实际 128） | ❌ 核心参数 | 记住 128 |
| 4 | 对称 vs 非对称的选择原因说不清 | ⚠️ | 简单高效 + LLM 分布近似对称 |
| 5 | GPTQ/AWQ 打包方向区别不知道 | ❌ 核心区别 | 行向 vs 列向 + 交错 shifts |
| 6 | 怎么自动检测模型格式不知道 | ❌ 自己写的代码 | 读 quant_config.json |
| 7 | 零侵入的架构边界答错（说 Python→C++） | ⚠️ | C++ 模型层 ↔ 算子层 |
| 8 | CUDA kernel 的并行策略答不上来 | ❌ 自己写的代码 | 256 threads, 1 byte/thread, 2 outputs |
| 9 | kernel 反量化步骤说太笼统 | ❌ 必须烂熟 | 7 步完整流程 |
| 10 | 性能瓶颈分析缺失 | ⚠️ | memory-bound, 算术强度 ≈ 0.77 |

---

## 面试建议

1. **数字要准**：3.76倍、group_size=128、INT4 范围 -8~7、256 threads/block — 这些必须脱口而出
2. **自己的代码必须熟**：`linear_maybe_dequant` 的四条路径、kernel 的 7 步流程、打包公式
3. **Memory-bound vs Compute-bound** 是高频考点，要能解释在什么场景下 WO 比 W8A8 快
4. **踩坑经验**要能讲故事：GPTQ 的 +1 偏移、AWQ 的双重量化都是好素材
5. **知道自己方案的局限**：非 fused kernel 的带宽代价，对称量化对不对称分布的表示损失
