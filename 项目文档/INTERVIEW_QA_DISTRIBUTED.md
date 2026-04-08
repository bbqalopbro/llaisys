# 面试深挖问答记录：分布式推理与高并发服务

> 简历片段：分布式推理与高并发服务：基于 NCCL/MPI 构建通信抽象层，实现底层算子级张量并行（TP）与 KV-Cache 切分，并在 8×A100 集群完成端到端验证；自建请求调度队列与前缀树（Prefix-Tree）KV-Cache 池，结合 FastAPI 提供支持多会话管理的高效流式对话服务。

---

## 一、通信抽象层设计（NCCL/MPI）

### Q1: 你提到"通信抽象层"这个概念。请你先解释一下：在你的系统中，这个抽象层的职责是什么？它对上层推理逻辑屏蔽了什么？你为什么需要同时支持 NCCL 和 MPI 两种后端，而不是只用 NCCL？

**参考答案**：

通信抽象层的三个核心职责：

1. **屏蔽实现差异**：NCCL 在 GPU 显存上直接操作（GPU-to-GPU），MPI 需要经 CPU 中转（GPU→CPU→MPI→CPU→GPU）。上层推理代码只调用统一接口（如 `allReduceSum`），不关心底层用哪种库。

2. **NCCL 适合 GPU 多卡场景**（NVLink/PCIe 直通），**MPI 适合 CPU 或跨节点场景**。

3. **有了抽象层，算子代码只写一次，切换后端只改配置。**

抽象层对上层屏蔽了：
- 通信库的初始化/销毁差异（NCCL 需要 UniqueId 交换，MPI 需要 MPI_Init）
- 数据位置差异（NCCL 直接操作 GPU 内存，MPI 需要 CPU 缓冲区中转）
- 同步语义差异（NCCL 用 CUDA stream 异步，MPI 是阻塞调用）

---

## 二、并行策略选择

### Q2: 张量并行和数据并行（DP）、流水线并行（PP）的区别是什么？你为什么在推理场景下选择了 TP 而不是其他并行策略？

**参考答案**：

#### 三种并行策略的区别（面试必背）

| 策略 | 切分维度 | 典型场景 | 通信模式 |
|------|---------|---------|---------|
| **DP（数据并行）** | 切分 **batch**：每张卡持有完整模型，各自处理不同数据 | **训练**（多 batch） | 梯度 allReduce |
| **PP（流水线并行）** | 切分 **层**：不同层放不同卡，数据像流水线一样逐层流过 | 超大模型训练 | 层间点对点传输 |
| **TP（张量并行）** | 切分 **同一层内的矩阵**：同一个 Linear 的权重按行/列切开，每张卡算一部分 | **推理**（低延迟） | allReduce（每层内） |

#### 为什么推理选 TP？

关键原因不只是显存：

1. **推理时 batch_size 通常很小（甚至为 1）**，DP 没有意义——你切 batch 没东西可切
2. **PP 会引入气泡（bubble）**：层 1 算完才能给层 2，延迟高，不适合对 latency 敏感的在线推理
3. **TP 把单层计算分到多卡并行执行**，延迟几乎线性降低（通信开销远小于计算），最适合在线推理场景

> 面试加分点：如果你还能提到 "TP 的通信量只跟 hidden_size 相关，和 sequence_length 无关，所以在长序列推理中通信开销占比很小"，面试官会印象深刻。

---

## 三、算子级张量并行——Column Parallel 与 Row Parallel

### Q3: 你说 TP 是"算子级"的并行。具体到 Transformer 的一个 Attention 层，哪些矩阵乘法（Q/K/V 投影、O 投影）适合按列切分（Column Parallel），哪些适合按行切分（Row Parallel）？为什么？

**参考答案**：

TP 切分的对象是 **投影层的权重矩阵**（Linear 层），不是 Attention 里的 Q×K^T 那步矩阵乘法本身。

| 投影层 | 权重形状 | 切分方式 | 原因 |
|--------|---------|---------|------|
| **Q/K/V 投影** | `[hidden_size, out_dim]` | **Column Parallel（按输出维/列切）** | 输出维就是 head 维，按 head 切最自然——每个 rank 分到各自的 head，各算各的，**不需要通信** |
| **O 投影** | `[n_head×head_dim, hidden_size]` | **Row Parallel（按输入维/行切）** | 每个 rank 持有部分输入（对应本地 head 的输出），各自做局部矩阵乘、**allReduce 求和**得到完整的 hidden_size 结果 |

#### 从计算流推导

```
输入 x: [seq, hidden_size]     ← 所有 rank 都有完整的 x

Q/K/V 投影 (Column Parallel):
  每个 rank: x × W_q_local = Q_local    ← W_q 按列切, 输出是局部 head
  → 不需要通信（各 rank 独立完成本地 head 的投影）

Attention 计算:
  每个 rank 用本地 Q_local, K_local, V_local 独立计算 → attn_local

O 投影 (Row Parallel):
  每个 rank: attn_local × W_o_local = partial_output
  → allReduce(sum) → 完整 output    ← 这里需要通信！
```

**核心逻辑**：Column→独立计算→Row→allReduce。这样**每个 Transformer 层只需要 1 次 allReduce**（在 O 投影之后），通信量最小。

同理，MLP 中 gate/up 是 Column Parallel，down 是 Row Parallel，也是每层 1 次 allReduce。

所以 **每层 2 次 allReduce**（Attention 一次 + MLP 一次），28 层就是 56 次。

#### 用极简例子理解 Column 和 Row

假设有一个 Linear 层：`Y = X × W`

- X 的形状：`[1, 4]`（1 个 token，hidden_size=4）
- W 的形状：`[4, 6]`（输入 4 维，输出 6 维）
- Y 的形状：`[1, 6]`

现在你有 **2 张 GPU**（tp_size=2），要把这个 Linear 的计算量切成两半。

**方法 A：Column Parallel（按列切 W）**

把 W **按输出维（列）** 切成两半：

```
GPU 0 拿到 W 的前 3 列: W0 = [4, 3]
GPU 1 拿到 W 的后 3 列: W1 = [4, 3]

GPU 0: Y0 = X × W0 = [1, 3]  ← Y 的前 3 个元素
GPU 1: Y1 = X × W1 = [1, 3]  ← Y 的后 3 个元素
```

**结果**：每个 GPU 得到 Y 的**一部分**（不完整）。
**不需要通信**，因为后续如果只对本地那部分做计算就行了。

**方法 B：Row Parallel（按行切 W）**

把 W **按输入维（行）** 切成两半：

```
GPU 0 拿到 W 的前 2 行: W0 = [2, 6]
GPU 1 拿到 W 的后 2 行: W1 = [2, 6]

但 X 也要切！
GPU 0 拿 X 的前 2 列: X0 = [1, 2]
GPU 1 拿 X 的后 2 列: X1 = [1, 2]

GPU 0: Y0 = X0 × W0 = [1, 6]  ← 部分和
GPU 1: Y1 = X1 × W1 = [1, 6]  ← 部分和
```

**结果**：每个 GPU 得到的 Y0、Y1 都是 `[1, 6]`，但都只是 **部分和**（不完整）。
**必须 allReduce(sum)**：`Y_完整 = Y0 + Y1`，才得到正确的输出。

#### 套到 Transformer 上

```
       ┌── Q/K/V 投影: Column Parallel (按列切)
       │     每个 GPU 得到局部 head 的 Q, K, V
       │
       │── Attention: 每个 GPU 用本地 head 独立计算
       │     （因为 attention 是 per-head 的，不需要别的 GPU 的 head）
       │     得到局部 attn_out
       │
       └── O 投影: Row Parallel (按行切)
             每个 GPU 拿本地 attn_out（部分输入）乘 W_o 的一部分
             得到部分和 → allReduce(sum) → 完整输出
```

**一句话总结：Column 在前面切开，各自独立算，Row 在后面收拢（allReduce 合并）。**

---

## 四、TP 下 KV-Cache 快照一致性

### Q4: 在张量并行场景下，每个 GPU 只持有部分 head 的 KV-Cache。当你需要对某个会话做 KV-Cache 快照保存和恢复时，你怎么保证多个 GPU 上的 KV-Cache 状态是一致的？如果 tp=2 时保存的快照，能恢复到 tp=1 的模型上吗？为什么？

**参考答案**：

**问题核心**：tp=2 时，GPU 0 存了 head 0~5 的 KV-Cache，GPU 1 存了 head 6~11 的。它们各自独立保存快照。

#### 一致性怎么保证？

- **不需要额外通信**！因为每个 rank 只 save/restore **自己本地的 KV-Cache**
- 关键前提：所有 rank 在**同一时刻**做 save/restore（由 TP Coordinator 广播指令同步）
- 只要所有 rank 的 `cache_pos` 一致，恢复后就能正确继续推理

#### tp=2 快照能恢复到 tp=1 吗？

- **不能**。原因：tp=2 的快照只包含一半的 head；tp=1 需要所有 head 的数据。维度不匹配，无法兼容。
- 这就是为什么快照里要**记录 tp_size 和 tp_rank**，恢复时检查是否匹配，不匹配就拒绝。

> 面试简洁回答："每个 rank 独立保存本地 KV-Cache 分片，恢复时通过 TP Coordinator 保证所有 rank 同步操作。快照元数据中记录了 tp_size/tp_rank，跨 TP 配置恢复会被拒绝。"

---

## 五、连续批处理（Continuous Batching）

### Q5: 什么是连续批处理（Continuous Batching）？它和传统的静态 batching 有什么区别？为什么在线推理服务需要连续批处理？

**参考答案**：

#### 背景：LLM 推理的特殊性

LLM 生成文本是**逐 token 的**。不同用户的请求长度不同，结束时间也不同：

```
用户 A: "你好" → 生成 50 个 token 后结束
用户 B: "写一篇论文" → 生成 500 个 token 后结束
用户 C: "1+1=" → 生成 5 个 token 后结束
```

#### Static Batching（传统做法，差）

把多个请求组成一个 batch，**等最慢的那个请求结束才处理下一批**：

```
时间 →
用户A ████████████░░░░░░░░░░░░  (50 tok，但要等用户 B 完成)
用户B ████████████████████████  (500 tok)
用户C ███░░░░░░░░░░░░░░░░░░░░  (5 tok，但也要等用户 B 完成)
                               ↑
                        这些 ░ 全是 GPU 空转浪费！
```

问题：短请求早结束了但占着 slot 不释放 → GPU 利用率低，用户 C 本来 5 个 token 就完了，却等了 500 token 的时间。

#### Continuous Batching（连续批处理，好）

**每一步 decode 之后检查**：谁完成了就踢出去，有新请求就立即插入空位：

```
时间 →
step 1:  [A, B, C]  → decode 一步
step 2:  [A, B, C]  → decode 一步
...
step 5:  [A, B, ✓C完成] → C 的 slot 释放！
step 6:  [A, B, D(新来的)] → D 插入空位！ D 先做 prefill
...
step 50: [✓A完成, B, D]
step 51: [E(新来), B, D] → E 插入
...
```

#### 核心区别

| 维度 | Static Batching | Continuous Batching |
|------|----------------|-------------------|
| slot 释放时机 | 整个 batch 最长请求结束 | **每步** decode 后立即释放 |
| 新请求插入 | 等整个 batch 完成 | 有空 slot 就**立即**插入 |
| GPU 利用率 | 低（短请求空转等长请求） | 高（slot 始终被填满） |
| 用户延迟 | 短请求被拖慢 | 短请求及时返回 |

#### 项目中的实现对应

`InferenceEngine._worker_loop()` 就是一个 continuous batching 循环：

1. **Admit 阶段**：检查空闲 slot → 从队列取新请求 → prefill → 插入运行集合
2. **Decode 阶段**：对所有活跃 slot 执行一步 batch decode
3. **Finish 阶段**：检查谁生成了 EOS 或达到 max_tokens → 释放 slot

这三步**每轮循环**都执行，所以完成一个请求能立刻接上新请求。

> 面试简洁回答："Continuous Batching 是每步 decode 后都可以踢出已完成的请求并插入新请求，避免短请求等长请求导致的 GPU 空转。我的实现中 worker 循环每轮包含 admit→decode→finish 三个阶段。"

---

## 六、前缀树 KV-Cache 池

### Q6: 你简历提到"前缀树（Prefix-Tree）KV-Cache 池"。请问：这个前缀树解决的是什么问题？它的工作机制是什么？跟直接为每个请求从头做 prefill 相比，优势在哪里？

**参考答案**：

#### 问题：重复 prefill 的浪费

在多用户对话服务中，不同请求经常**共享相同的前缀**：

```
用户 A: "你是一个AI助手。请问什么是机器学习？"
用户 B: "你是一个AI助手。请解释深度学习。"
用户 C: "你是一个AI助手。今天天气怎么样？"
```

这三个请求的前缀 `"你是一个AI助手。"` 对应的 token 序列**完全相同**。如果每次都从头 prefill，同一段前缀的 KV-Cache 会被**反复计算**。

#### 前缀树工作机制

前缀树（Trie）是一种**按 token 逐级存储**的树结构：

```
根
└── "你"
    └── "是"
        └── "一个"
            └── "AI"
                └── "助手"
                    └── "。"  ← 这个节点存了一份 KV-Cache 快照！
                        ├── "请问" → "什么" → ...  (用户 A 的路径)
                        ├── "请" → "解释" → ...    (用户 B 的路径)
                        └── "今天" → ...            (用户 C 的路径)
```

**每个节点可以挂一份 KV-Cache 快照**，记录"到这个位置为止的 KV-Cache 状态"。

#### 处理流程

1. **新请求来了**，先把 token 序列在前缀树中查找最长匹配
2. **命中前缀**（比如匹配了 6 个 token）→ 直接**恢复**对应的 KV-Cache 快照
3. **只 prefill 剩余部分**（未匹配的 token）

```
请求 A: 无缓存 → 完整 prefill 20 个 token → 存快照到树中
请求 B: 命中前缀 6 个 token → 恢复快照 → 只 prefill 剩余 8 个 token  ← 省了 6 个 token 的计算！
请求 C: 命中前缀 6 个 token → 恢复快照 → 只 prefill 剩余 7 个 token
```

#### 优势

| 对比 | 无前缀树 | 有前缀树 |
|------|---------|---------|
| 请求 B 的 prefill 计算量 | 14 tokens | 8 tokens（省掉 6 个共享前缀） |
| System prompt 1000 tokens 场景 | 每个请求都算 1000 | 第一次算 1000，之后直接跳过 |
| 核心收益 | — | **prefill 延迟下降，GPU 计算量减少** |

> **面试简洁回答**："前缀树按 token 粒度存储历史请求的 KV-Cache 快照。新请求先查最长前缀匹配，命中后直接恢复 KV-Cache，只对未匹配的部分做 prefill，避免重复计算共享前缀。在多用户共享 system prompt 的场景下收益最大。"

---

## 七、流式服务数据通路

### Q7: 你简历写"结合 FastAPI 提供高效流式对话服务"。请问：你的流式输出是怎么实现的？从后台推理线程生成一个 token，到用户浏览器收到这个 token，中间经过了哪些环节？

**参考答案**：

一个 token 从生成到用户收到，经过 **5 个环节**：

```
┌─────────────────────────────────────────────────────────┐
│  后台推理线程 (Python threading.Thread)                    │
│                                                         │
│  worker_loop 中 decode 得到 next_token                    │
│       │                                                 │
│       ▼                                                 │
│  ① loop.call_soon_threadsafe(queue.put, token)          │
│     ↑ 这是关键！线程安全地往 asyncio 世界投递数据          │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  asyncio 事件循环 (FastAPI 运行在这里)                      │
│                                                         │
│  ② asyncio.Queue.put() 被触发                             │
│       │                                                 │
│       ▼                                                 │
│  ③ async generator (stream_tokens) 中                    │
│     token = await queue.get()                           │
│     text = tokenizer.decode(token)                      │
│       │                                                 │
│       ▼                                                 │
│  ④ yield f"data: {json}\n\n"    ← SSE 格式               │
│     FastAPI 的 EventSourceResponse 逐块发送               │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  ⑤ HTTP 响应 (SSE: Server-Sent Events)                   │
│     Content-Type: text/event-stream                     │
│     浏览器/客户端逐条读取 data: {...}                      │
└──────────────────────────────────────────────────────────┘
```

#### 逐步解释

| 环节 | 位置 | 说明 |
|------|------|------|
| **①** 推理线程产出 token | 后台 Thread | `_worker_loop` 中 `batch_ctx.decode()` 返回 next_token |
| **②** 线程→协程 跨界传递 | `call_soon_threadsafe` | 这是**唯一的线程安全桥梁**：推理线程不能直接 `await`，必须通过 `loop.call_soon_threadsafe` 把数据投到 asyncio 的 Queue 里 |
| **③** async generator 消费 | FastAPI 协程 | `await queue.get()` 拿到 token → decode 成文本 |
| **④** SSE 格式 yield | FastAPI 响应流 | 按 OpenAI 格式包装成 `data: {"choices":[{"delta":{"content":"你"}}]}\n\n` |
| **⑤** HTTP 推送到客户端 | 网络层 | `Transfer-Encoding: chunked`，浏览器逐条收到、逐字显示 |

#### 面试关键点

面试官最想听的是 **环节②** —— **线程和协程之间怎么安全通信**：

> "推理引擎跑在独立的 Python 线程中，FastAPI 跑在 asyncio 事件循环中。两者通过 `asyncio.Queue` 桥接：推理线程用 `loop.call_soon_threadsafe(queue.put_nowait, token)` 将 token 投递到 asyncio 事件循环中，协程端通过 `await queue.get()` 非阻塞消费。这样推理线程不会阻塞事件循环，事件循环也不会阻塞推理。"

#### SSE（Server-Sent Events）补充

- SSE 是 HTTP 协议的一部分：服务端保持连接不关闭，持续往客户端推数据
- 每条数据格式：`data: {json}\n\n`，结束标记：`data: [DONE]\n\n`
- 跟 WebSocket 的区别：SSE 是**单向**的（只有服务端→客户端），更简单

---

## 八、多会话管理与开销

### Q8: 当用户 A 正在对话、用户 B 也发来请求时，涉及会话切换。切换会话意味着什么？会话切换有什么开销？如果有 1000 个活跃会话，你怎么处理？

**参考答案**：

#### 会话切换意味着什么？

每个会话有自己独立的**对话历史**和对应的 **KV-Cache 状态**。当用户 A 在对话时，GPU 上的 KV-Cache 存的是用户 A 的上下文。用户 B 发来请求时，如果用的是**同一个模型实例**，就需要：

```
1. 把用户 A 的 KV-Cache 从 GPU 拷贝到 CPU（save 快照）
2. 把用户 B 的 KV-Cache 从 CPU 恢复到 GPU（restore 快照）
3. 在用户 B 的上下文上继续推理
```

这就是**会话切换**——本质是 **KV-Cache 的换入换出（swap）**。

#### 开销有多大？

以 DeepSeek-R1-Distill-Qwen-1.5B 为例算一笔账：

```
28 层 × 2(K+V) × 序列长度 × kv_heads × head_dim × 4 bytes(FP32)

假设序列长度 = 2048：
28 × 2 × 2048 × 2 × 64 × 4 = 约 58 MB
```

每次切换 = **一次 58MB GPU→CPU 拷贝 + 一次 58MB CPU→GPU 拷贝**。

A100 上 PCIe 带宽约 25 GB/s，单次切换延迟 ≈ 116MB / 25GB/s ≈ **4.6 ms**。
看起来不大，但如果**频繁切换**，积累起来就很可观。

#### 1000 个活跃会话怎么办？

1000 个会话的快照全存在 CPU 内存里 → 1000 × 58MB ≈ **58 GB CPU 内存**。

**实际方案**：用 Continuous Batching 避免了频繁切换！

```
旧做法 (单 slot):
  用户 A 占 GPU → 用户 B 来了 → 换出 A、换入 B → B 完成 → 换出 B、换入 A
  → 频繁 swap，开销大

Continuous Batching (多 slot):
  slot 0: 用户 A  ← 各 slot 有独立的 KV-Cache
  slot 1: 用户 B  ← 同时在 GPU 上！
  slot 2: 用户 C
  slot 3: 用户 D
  → 4 个用户同时推理，不需要切换！
```

BatchContext 预分配了 `max_batch_size` 个 slot，每个 slot 有独立的 KV-Cache。**在 batch 容量内的并发请求不需要任何切换**。

超出容量时的优化手段：

| 策略 | 说明 |
|------|------|
| **LRU 淘汰** | 最久未使用的会话快照优先释放 |
| **分层存储** | 热会话在 GPU → 温会话在 CPU → 冷会话写磁盘 |
| **前缀树复用** | 共享 system prompt 的快照只存一份 |
| **量化快照** | 将 FP32 KV-Cache 以 FP16/INT8 存储，减半内存占用 |

> 面试简洁回答："会话切换的本质是 KV-Cache 的 swap in/out。我的方案用 Continuous Batching 的多 slot 机制，在 batch 容量内实现零切换并发。超出容量时可以用 LRU 淘汰 + 分层存储来管理大量会话。"

---

## 九、FastAPI 选型与 async 机制

### Q9: FastAPI 和 Flask 有什么区别？你为什么选 FastAPI？它的 async 特性在你的推理服务中具体起到了什么作用？

**参考答案**：

#### FastAPI vs Flask

| 维度 | Flask | FastAPI |
|------|-------|---------|
| 并发模型 | 同步（多线程/多进程） | **异步（asyncio 事件循环）** |
| 处理等待 | 线程阻塞，干等 | `await` 让出，去处理别的请求 |
| 并发能力 | 受线程数限制（通常 4~16） | **单线程处理上千并发**（I/O 密集型） |
| 流式响应（SSE） | 需要额外库，较麻烦 | **原生支持** `StreamingResponse` |
| 类型校验 | 手动写 | **自动**（基于 Pydantic） |
| API 文档 | 手动写 | **自动生成** Swagger UI（访问 /docs） |

#### 为什么选 FastAPI？

1. **流式输出**：LLM 逐 token 生成，需要 SSE 持续推送。FastAPI 原生支持 `EventSourceResponse`，Flask 做起来很麻烦。
2. **高并发**：多用户同时对话，需要同时持有上百个 HTTP 连接。async 模型比多线程高效得多。
3. **推理是异步的**：请求提交到队列后需要等，`await` 让等待期间不浪费资源。

#### async 在推理服务中的具体作用

```
                ┌──────────────────────────┐
                │    FastAPI (asyncio)      │  ← 1 个线程，处理所有 HTTP 请求
                │                          │
用户A POST ──→  │  async def chat():       │
用户B POST ──→  │    req = engine.submit() │  ← 提交到队列（瞬间完成）
用户C POST ──→  │    result = await future │  ← 非阻塞等待
                │    return response       │
                └──────────┬───────────────┘
                           │
              engine.submit() 把请求放入队列
                           │
                           ▼
                ┌──────────────────────────┐
                │  InferenceEngine (独立线程) │  ← 1 个后台线程，独占 GPU
                │                          │
                │  while running:           │
                │    取请求 → prefill       │
                │    batch decode           │
                │    完成 → set_result      │  ← 通过 call_soon_threadsafe 通知 FastAPI
                └──────────────────────────┘
```

分工：
- **FastAPI 线程**：只负责接收请求、返回响应，**不做任何推理计算**
- **Engine 线程**：独占 GPU，做所有推理计算

> **面试简洁回答**："选 FastAPI 是因为三点：(1) 原生支持 async，单线程就能处理大量并发连接；(2) 原生支持 SSE 流式响应，适合 LLM 逐 token 推送；(3) 推理引擎是异步的（提交→等待→返回），async/await 模型天然匹配。"

---

## 十、asyncio.Queue、asyncio.Future 与线程-协程桥接

### 补充知识：asyncio.Future —— "一个承诺"

比喻：你去餐厅点了一碗面。服务员给你一个**取餐号码牌**——这就是 Future。

- 面还没做好 → Future 还没有结果（pending）
- 面做好了，厨房叫号 → Future 被 `set_result()`（resolved）
- 你拿着号牌等叫号 → `await future`（等待结果）

**特点**：只能 set_result **一次**，适合 **"一次性返回全部结果"** 的场景（非流式）。

### 补充知识：asyncio.Queue —— "一条传送带"

比喻：你在吃旋转寿司。厨师一个一个做好放上传送带，你看到了就拿——这就是 Queue。

**特点**：可以 put **多次**，适合 **"逐个产出、逐个消费"** 的场景（流式）。

### 两者在项目中的分工

```
                    非流式请求                       流式请求

用户: "给我答案"                     用户: "逐字输出"
  │                                   │
  ▼                                   ▼
engine.submit(stream=False)        engine.submit(stream=True)
  │                                   │
  ▼                                   ▼
创建 Future                         创建 Queue
  │                                   │
  ▼                                   ▼
推理线程完整生成所有 token            推理线程每生成一个 token
  │                                   │
  ▼                                   ▼
future.set_result(全部tokens)       queue.put(单个token)
  │                                   │
  ▼                                   ▼
await future → 一次性返回            async for token in queue:
                                      yield → 逐个推送给用户
```

### `loop.call_soon_threadsafe` 的原理

推理线程不能直接操作 asyncio 的 Queue/Future（不是线程安全的）。`call_soon_threadsafe` 的工作原理：

```
推理线程                           事件循环线程
    │                                  │
    │ call_soon_threadsafe(fn, arg)     │
    │──→ [加锁] 放入待办列表 ──→         │
    │    写一个字节到 eventfd ──→        │ ← 被唤醒！
    │                                  │ 从待办列表取出 (fn, arg)
    │                                  │ 执行 fn(arg) → queue.put_nowait(token)
    │                                  │ 继续处理其他协程
```

**核心思想**：不让推理线程直接操作 asyncio 对象，而是**委托给事件循环线程去做**。事件循环线程是唯一有权操作 asyncio 对象的线程。

#### 为什么叫 "threadsafe" ？

因为 `call_soon_threadsafe` 自身内部有锁：

```python
# CPython asyncio 源码简化版：
def call_soon_threadsafe(self, callback, *args):
    self._ready.append((callback, args))   # 这个列表的写入是加了锁的 ✅
    self._write_to_self()                  # 写一个字节唤醒 select/epoll
```

而普通的 `call_soon()` 没有加锁，只能在事件循环线程内调用。

> **面试简洁回答**："非流式用 `asyncio.Future` 一次性返回结果，流式用 `asyncio.Queue` 逐 token 传递。推理线程通过 `loop.call_soon_threadsafe` 把数据安全地投递到 asyncio 事件循环中——它内部通过加锁的待办队列 + eventfd 唤醒机制保证线程安全。"
