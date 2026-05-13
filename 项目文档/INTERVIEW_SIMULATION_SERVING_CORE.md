# Serving Core 简历逐句模拟面试记录

> 记录方式：每题包含面试官问题、我的原回答、评估、推荐回答和可能追问。

## 句子 1：Python/C/C++ 三层跨平台推理框架

简历原句：

> 搭建 Python/C/C++ 三层跨平台推理框架，基于动态 Runtime 路由与编译级设备隔离支持设备无缝切换。

### Q1：为什么要设计 Python / C / C++ 这三层？每一层分别负责什么？

我的原回答：

> Cpp/cuda 是计算层，里面包括了推理、算子等底层的计算，C 是 cpp 和 python 的胶水层，通过函数指针表让 python 层调用，Python ctypes 直接调用 C 接口，python 层进行了模型加载等（python 层的主要工作我不记得了）

评估：

- 方向正确：能说出 C++/CUDA 是计算层，Python 通过 `ctypes` 调 C 接口。
- 需要修正：“C 通过函数指针表让 Python 调用”不够准确。Python 主路径是通过 `ctypes` 调 shared library 导出的 C ABI 函数；函数指针表主要用于 Runtime API 对设备操作的抽象。
- Python 层职责需要记牢，否则容易被追问卡住。

推荐回答：

> 我把框架分成 Python、C API 和 C++/CUDA 三层。Python 层负责工程灵活性比较强的部分，比如读取 HuggingFace 模型配置和权重、封装 Qwen2 模型对象、提供 CLI/server 接口，以及通过 `ctypes` 调用底层动态库。C 层是稳定 ABI 边界，对外暴露 `llaisysQwen2ModelCreate`、`llaisysQwen2LoadWeightByName`、`llaisysQwen2ModelInfer` 这类接口，避免 Python 直接绑定复杂 C++ 类型。C++/CUDA 层负责真正的高性能计算，包括 Tensor/Storage、Runtime、算子实现、KV Cache、Qwen2 forward 和 CUDA kernels。
>
> 这样分层的好处是 Python 层方便接生态和做服务，C ABI 保持跨语言稳定，C++/CUDA 层可以专注性能和设备后端。

### Q2：你说“动态 Runtime 路由”，具体是什么意思？一次 Tensor 分配或算子调用，怎么根据 CPU/NVIDIA/MetaX 走到不同实现？

我的原回答：

> runtime 路由是每个设备的上下文 context 持有一个 runtime 的 api，用于调控不同设备之间的运行，通过函数指针表进行一些基本函数的黑盒的封装，例如 malloc、free 等操作

评估：

- 核心方向正确：`Context -> Runtime -> RuntimeAPI`，Runtime API 用函数指针表封装设备操作。
- 可以再补两点：第一，Runtime API 路由的是内存、stream、memcpy、set_device 这类基础能力；第二，算子路由通常是在 `ops` 层根据 `device_type` 分发到 CPU/NVIDIA/MetaX 实现。
- “调控不同设备之间的运行”略泛，可以说成“屏蔽不同设备后端的基础运行时差异”。

推荐回答：

> 动态 Runtime 路由指的是上层不直接调用 CUDA 或某个设备厂商 API，而是通过当前 `Context` 里激活的 `Runtime` 获取一张 `LlaisysRuntimeAPI` 函数表。这个函数表封装了 `set_device`、`malloc_device`、`free_device`、`memcpy_async`、stream 创建和同步等基础操作。创建 Tensor 或分配 Storage 时，会根据 `device_type` 找到对应 Runtime API，比如 CPU 走 CPU runtime，NVIDIA 编译开启时走 CUDA runtime，MetaX 编译开启时走 MetaX runtime。
>
> 算子层也是类似思想：统一的 `ops::linear`、`ops::rope`、`ops::rms_norm` 入口会根据 tensor 所在设备或传入的 `device_type` 分发到 CPU/NVIDIA/MetaX 的具体实现。这样上层模型 forward 不需要关心底层设备 API，只需要传递 device 信息。

可能追问：

> Runtime 路由和编译级设备隔离有什么区别？

### Q3：动态 Runtime 路由和编译级设备隔离分别解决什么问题？它们有什么区别？

我的原回答：

> 编译级设备隔离是在 xmake 中进行了，在编译期间就会针对不同的设备例如 cpu 或者 gpu 进行编译隔离

评估：

- 方向正确：能指出编译级隔离是在 `xmake` 里通过不同选项控制 CPU/GPU 后端是否参与编译。
- 回答还不完整：需要明确它和 Runtime 路由的分工。编译级隔离解决“能不能编译、链接哪些后端代码”的问题；Runtime 路由解决“运行时当前请求/张量使用哪个后端 API”的问题。
- 建议举具体例子：`nv-gpu` 开启时定义 `ENABLE_NVIDIA_API` 并编译 CUDA 文件；不开启时不编译 CUDA 算子，也不会强依赖 CUDA/cuBLAS/NCCL。

推荐回答：

> 编译级设备隔离主要解决构建和依赖问题。比如项目用 `xmake` 配置 `nv-gpu`、`metax-gpu`、`dist-nccl`、`dist-mpi` 等开关。开启 `nv-gpu` 时才定义 `ENABLE_NVIDIA_API`，编译 `src/device/nvidia/*.cu` 和各个 `ops/*/nvidia/*.cu`，并链接 `cublas`、`cudart`；不开启时这些 CUDA 文件不会进入编译，机器没有 CUDA 环境也能构建 CPU 版本。
>
> Runtime 路由解决的是运行时选择问题。编译出来的二进制如果包含多个后端，上层根据 `device_type` 获取对应的 Runtime API，比如 CPU/NVIDIA/MetaX，然后走对应的 malloc、memcpy、stream 和算子实现。
>
> 所以区别是：编译级隔离决定“哪些设备后端被编进来”，Runtime 路由决定“运行时当前用哪个后端”。前者偏构建和依赖隔离，后者偏执行路径选择。

可能追问：

> 如果没有开启 `nv-gpu` 编译选项，但运行时用户传了 `LLAISYS_DEVICE_NVIDIA`，会发生什么？

### Q4：如果没有开启 `nv-gpu` 编译选项，但运行时用户传了 `LLAISYS_DEVICE_NVIDIA`，框架会怎么处理？

我的原回答：

> 会报错，并默认按照 cpu 进行编译有哪些

评估：

- “会报错”是对的。
- “默认按照 CPU 进行编译”要说清楚：CPU 后端在 `xmake.lua` 中默认总是 include/编译；但运行时如果用户显式选择 NVIDIA，不会自动 fallback 到 CPU 执行，而是走 unsupported runtime API，调用设备操作时抛出 unsupported device。
- 面试时不要说“传 NVIDIA 后默认走 CPU”，这会让人觉得设备语义不清晰。

推荐回答：

> 如果编译时没有开启 `nv-gpu`，`ENABLE_NVIDIA_API` 不会被定义，NVIDIA runtime 和 CUDA 算子都不会编进来。运行时如果用户仍然传 `LLAISYS_DEVICE_NVIDIA`，`getRuntimeAPI` 会返回 unsupported runtime API；后续调用 `set_device`、`malloc_device`、`memcpy` 这类函数时会报 unsupported device。也就是说，CPU 后端是默认编译的，但显式选择 NVIDIA 不会悄悄 fallback 到 CPU，这样可以避免用户以为自己在跑 GPU，实际却跑了 CPU。

可能追问：

> 为什么不做自动 fallback 到 CPU？这样不是更友好吗？

### Q5：为什么不自动 fallback 到 CPU？如果用户没装 CUDA，直接跑 CPU 不是更友好吗？

我的原回答：

> fallback 到 cpu 会有很大的 memcpy 开销即带宽压力

评估：

- 提到了一个真实问题：如果运行过程中从 GPU fallback 到 CPU，确实会有数据迁移和带宽开销。
- 但这不是最核心理由。更关键的是：自动 fallback 会静默改变用户的执行语义和性能预期，用户以为在跑 GPU，实际跑 CPU，benchmark 和排错都会被误导。
- 还可以补充：不同设备的 tensor、storage、stream、算子支持集合不同，不是所有状态都能无损迁移；对推理框架来说，显式失败比隐式慢跑更安全。

推荐回答：

> 不做静默 fallback 主要是为了避免改变执行语义。用户显式传了 `LLAISYS_DEVICE_NVIDIA`，说明他期望使用 GPU。如果框架悄悄切到 CPU，结果虽然可能能跑，但性能会完全不符合预期，benchmark 和线上排错都会被误导。
>
> 另外，真正从 GPU fallback 到 CPU 也不是简单换一个函数入口。Tensor storage、KV cache、临时 buffer 可能已经在设备侧，迁移到 CPU 会产生大量 memcpy 和带宽开销；有些 CUDA 专用路径，比如 CUDA Graph、paged attention device pointer、cuBLAS workspace，也没有直接的 CPU 等价状态。因此当前设计选择显式报错，让用户修正编译选项或设备配置。

可能追问：

> 那你说“支持设备无缝切换”，和“不自动 fallback”是不是矛盾？

### Q6：“支持设备无缝切换”和“不自动 fallback”是不是矛盾？你怎么解释？

我的原回答：

> 这里表述是有误的，我的无缝切换是指编译后，在运行时能通过多设备的 runtime 路由进行不同设备的函数调用

评估：

- 这个回答边界感很好：主动澄清“无缝切换”不是运行失败时自动 fallback。
- 可以进一步补一句前提：对应设备后端必须已经在编译时启用并正确链接。
- “不同设备的函数调用”可以具体化为 Runtime API 和 ops dispatch 两层：基础内存/stream 操作走 Runtime API，算子走对应后端实现。

推荐回答：

> 不矛盾，但“无缝切换”这个表述需要限定前提。我的意思不是运行失败后自动 fallback，也不是运行中把 GPU 上的状态透明迁移到 CPU；而是在对应设备后端已经编译进来的前提下，上层 Python/C API 和模型 forward 基本不变，运行时通过 `device_type` 选择不同 Runtime API 和算子后端。
>
> 具体来说，Tensor 分配、memcpy、stream 等基础操作通过 Runtime API 路由到 CPU/NVIDIA/MetaX；`linear`、`rope`、`rms_norm` 这类算子则在 ops 层分发到对应设备实现。所以“无缝”是对上层调用接口而言的，不是对错误配置做静默兜底。

可能追问：

> 你这个三层框架和直接用 PyTorch 扩展相比，有什么收益和代价？

### Q7：这个 Python/C/C++ 三层框架和直接写 PyTorch extension 相比，有什么收益和代价？Python 调 C++ 除了 ctypes 还有哪些方式？

我的原回答：

> 我的底层的 tensor、meta 还有 storage 等都是 cpp 实现的，而不是直接调用 pytorch 的 tensor 等，因为这是项目最开始给的框架，我是在这个 cpp 框架上进行的升级，Python 调 C++ 除了 ctypes 还有 pyblind 等，ctypes 主要的优点是标准库自带，实现简单

评估：

- 回答真实可信：说明了项目背景是已有 C++ Tensor/Storage/Runtime 框架，你是在这个框架上做升级。
- `pyblind` 应改为 `pybind11`。
- 需要补充收益和代价：收益是可控 runtime、独立 Tensor/Storage、跨设备后端隔离、服务侧 C ABI；代价是要自己维护 Tensor、allocator、dispatcher、算子和绑定，工程复杂度高于 PyTorch extension。
- 不要让回答听起来像“因为老师给的框架所以只能这样”。可以先讲项目定位，再讲历史背景。

推荐回答：

> 这个项目本身定位更接近一个独立推理 runtime，而不是 PyTorch 的单个算子扩展。底层的 Tensor、Storage、Runtime、模型元信息和算子分发都在 C++ 框架里实现，我是在这个框架上继续升级 PagedAttention、CUDA kernels、TP 通信和量化推理。所以它的收益是执行链路更可控，不强依赖 PyTorch Tensor 和 dispatcher，更方便做 Runtime 后端隔离、C ABI 暴露、KV cache 管理、CUDA Graph 这类底层优化。
>
> 代价也很明显：PyTorch 已经有成熟的 Tensor、allocator、stream 管理、算子生态和调试工具；自己做 runtime 就要维护这些基础设施的最小子集，工程复杂度更高，功能完整性也远不如 PyTorch。
>
> Python 调 C/C++ 的方式除了 `ctypes`，还有 `pybind11`、Cython、Python C API、cffi，以及 PyTorch C++/CUDA extension。我这里使用 `ctypes`，主要因为它是标准库自带，部署简单，只要底层暴露稳定的 C ABI，就可以从 Python 加载 `.so` 调用。但缺点是类型安全弱，复杂对象和生命周期管理需要自己处理；如果要暴露大量 C++ 类，`pybind11` 会更舒服。

可能追问：

> 既然 ctypes 类型安全弱，那你们怎么管理 C++ 对象生命周期？比如模型对象、tensor 对象在 Python 和 C++ 之间怎么传？

### Q8：既然 ctypes 类型安全弱，那你们怎么管理 C++ 对象生命周期？模型对象、权重对象、Tensor 对象在 Python 和 C++ 之间怎么传？

我的原回答：

> 主要是通过 RAII 原则来进行 tensor 对象的管理，每个 tensor 对象都由 shared 指针进行管理（具体我说不清楚），传递也不清楚

评估：

- RAII 和 `shared_ptr` 方向正确，说明你知道 C++ 侧对象不是裸指针满天飞。
- 需要补清楚 Python/C 边界：Python 通过 `ctypes.c_void_p` 保存 C++ 对象的不透明句柄，不能直接访问 C++ 对象内部。
- 模型对象的生命周期是显式 create/destroy；权重 Tensor 一般由 C++ 模型内部的 `resources` 容器持有 `shared_ptr`，C 结构体里保存的是裸 handle，避免权重加载函数返回后 Tensor 被释放。
- 输入权重数据通常是 Python 侧 numpy/torch tensor 的数据指针临时传入，C++ 侧创建自己的 Tensor 并拷贝到设备/CPU storage，因此不会长期依赖 Python 原始 buffer。

推荐回答：

> Python 和 C++ 之间主要传的是不透明句柄，也就是 `ctypes.c_void_p`。比如 Python 调 `model_create` 后拿到的是 `LlaisysQwen2Model*`，Python 不直接访问对象内部，只在后续 `load_weight`、`infer`、`destroy` 时把这个 handle 传回 C API。
>
> 生命周期上，模型对象是显式管理的：C API 里 `llaisysQwen2ModelCreate` 用 `new` 创建，Python wrapper 保存 handle，结束时调用 `llaisysQwen2ModelDestroy` 释放。C++ 对象内部再用 RAII 管理资源，比如 Tensor/Storage 用 `shared_ptr`，BlockAllocator 析构时释放 device pool，CUDA Graph runner 析构时销毁 graph。
>
> 权重加载时，Python 把权重名、shape、dtype 和数据指针传给 C API。C++ 侧会创建自己的 `Tensor`，把 Python 传入的数据拷贝进框架管理的 storage，然后把这个 `tensor_t` 放进模型的 `resources` 容器里延长生命周期；权重结构体里保存 `llaisysTensor_t` handle 指向这些 Tensor。这样 Python 原始 buffer 只需要在调用期间有效，模型推理长期依赖的是 C++ 自己持有的 Tensor。
>
> 这套做法的不足是 `ctypes` 类型安全弱，handle 传错或者重复 destroy 编译器管不了，所以需要在 Python wrapper 里配置 `argtypes/restype`，并尽量把对象创建和销毁封装在类里。

可能追问：

> Python 传权重数据指针给 C++ 时，怎么保证数据布局和 dtype 是对的？

### Q9：Python 传权重数据指针给 C++ 时，怎么保证数据布局和 dtype 是对的？

我的原回答：

> 这个问题我也不清楚，实际上 python 的代码我没怎么看过，后续需要仔细看看 python 层的代码

评估：

- 这是一个真实知识缺口，需要补 Python 权重加载链路。
- 面试中不能只说“不清楚”，至少要能讲出：Python 读取 config/权重，转换为连续数组，传入 data pointer、shape、dtype；C++ 按传入 shape/dtype 创建 Tensor 并 load。
- 如果不确定某些细节，可以诚实限定：“我这块主要做底层 C++/CUDA，Python 权重加载是基于已有框架，不过我理解它通过 dtype/shape 显式传参保证 C ABI 边界的信息完整。”

推荐回答：

> Python 侧会先读取模型的 `config.json` 和 safetensors 权重，得到每个权重 tensor 的 shape 和 dtype。调用 C API 时不会只传一个裸指针，而是同时传 `name`、`data`、`ndim`、`shape` 和 `dtype`。C++ 侧 `llaisysQwen2LoadWeightByName` 根据这些元信息创建框架自己的 Tensor，并把 Python 传入的数据拷贝到 C++ 管理的 storage 中。
>
> 数据布局上，Python 侧需要确保传给 C++ 的数组是 contiguous 的，并且 dtype 映射到框架里的 `llaisysDataType_t`。C++ 侧再根据权重名决定它属于 Q/K/V/O projection、MLP、embedding 还是量化 scale/qzeros，并在 TP 模式下做对应维度切分。
>
> 这套方式的风险是 C ABI 本身不会自动检查 dtype 和 shape 语义是否匹配，所以 Python wrapper 需要负责把 HuggingFace 权重转换成框架期望的布局；C++ 侧只能做一部分维度和 dtype 校验。这个地方如果工程化继续推进，应该补更严格的 shape validation 和错误信息。

可能追问：

> 那 C API 为什么要传权重名？不能只按固定顺序加载吗？

### Q10：C API 为什么要传权重名？不能只按固定顺序加载吗？另外怎么做 shape/dtype validation？

我的原回答：

> 因为 hg 下载的权重就是按名字进行排列的，我将名字记录下后，才能在推理过程中正确地使用对应的权重，例如 qkv 的映射和 mlp 映射等，可以，但是这样更稳定，也更利于后续对其他模型的扩展，要做 shape/dtype validation，需要 python 端传递步长或者先进行连续化处理

评估：

- “按名字路由到 Q/K/V/MLP 等权重字段”回答正确。
- “更稳定、更利于其他模型扩展”也正确，因为 safetensors/HuggingFace 权重 key 天然是 name-based，不应依赖文件内部顺序。
- 需要修正：shape/dtype validation 主要是检查传入 shape 和 dtype 是否符合模型 meta 预期；传 strides 或做 contiguous 是数据布局问题，不等同于 shape validation。
- 可以说：当前 C API 假设传入 buffer 是 contiguous；如果要支持 non-contiguous，才需要额外传 strides 或 Python 侧先 `.contiguous()`。

推荐回答：

> C API 传权重名是因为 HuggingFace/safetensors 权重本身就是按 key 组织的。C++ loader 需要根据名字判断这个 tensor 应该放到哪个字段，比如 `self_attn.q_proj.weight` 放到某一层的 Q projection，`k_proj/v_proj/o_proj` 分别放到对应 attention 权重，`mlp.gate_proj/up_proj/down_proj` 放到 MLP，`.scale`、`.qzeros` 则进入量化相关字段。
>
> 固定顺序理论上也可以，但它会强依赖权重文件遍历顺序，不同模型或不同量化格式一旦 key 有增减就容易错位。按名字加载更稳定，也更利于兼容 HuggingFace、AWQ/GPTQ 和后续其他模型。
>
> shape/dtype validation 可以在 C++ loader 里根据模型 meta 和权重名做检查。例如 `q_proj.weight` 预期是 `[hidden_size, hidden_size]`，`k_proj.weight` 预期是 `[num_kv_heads * head_dim, hidden_size]`，`down_proj.weight` 预期是 `[hidden_size, intermediate_size]`，同时检查 dtype 是否属于 FP16/FP32/INT8/INT4 等支持类型。如果是 TP 模式，还要检查切分维度能被 `tp_size` 整除。
>
> contiguous 是另一个问题。当前 C ABI 传的是 data pointer、shape、dtype，默认 Python 侧传入连续 buffer。如果要支持 non-contiguous tensor，就需要额外传 strides，并在 C++ 侧按 strides 拷贝；更简单的做法是 Python 侧在传入前调用 `.contiguous()`。

可能追问：

> 好，三层框架这一句我们过了。下一句：你说“核心实现 vLLM 风格 PagedAttention 引擎”，PagedAttention 具体解决了什么问题？

## 句子 2：vLLM 风格 PagedAttention 引擎

简历原句：

> 核心实现 vLLM 风格 PagedAttention 引擎——手写 Online Softmax CUDA Kernel（单次遍历 KV，支持 FP16/BF16 混合精度与 GQA），配合 Block Allocator + Page Table 实现按需分页 KV-Cache，典型场景下显存节省 41%。

### Q11：PagedAttention 具体解决了什么问题？为什么传统 KV Cache 会浪费显存？

我的原回答：

> 解决的是最开始的 kvcache 一次性预分配一个最大上下文长度的显存，导致严重显存碎片化和浪费的问题，PagedAttention 用了类似虚拟页表的操作，分配了一块 blocktable 来记录显存中 block 的使用情况，利用碎片化的 blok 而不是大块的连续显存

评估：

- 核心方向正确：传统 KV Cache 按最大上下文长度预分配，短请求浪费；PagedAttention 借鉴虚拟内存/页表思想，用 block 粒度管理 KV。
- 术语需要更准确：这里主要是“预留浪费”和“最后一个 block 的内部碎片”，不是传统意义上的外部碎片化。
- `block table` 不是记录全局 block 使用情况；全局空闲/已用由 `BlockAllocator` 管，`PageTable/block table` 记录的是每个序列“逻辑 block → 物理 block”的映射。
- “利用碎片化的 block”可以改成“允许逻辑连续的 KV 存在物理不连续的 block 中，通过间接寻址访问”。

推荐回答：

> PagedAttention 主要解决 KV Cache 显存预留浪费的问题。传统实现通常给每个请求或 slot 按 `max_seq_len × num_kv_heads × head_dim × num_layers` 预分配连续 KV Cache，但真实请求长度差异很大。比如 max context 是 4096，很多请求只生成几十到几百个 token，那么大部分预留 KV 空间都没有被使用。
>
> PagedAttention 借鉴操作系统虚拟内存的思想，把 KV Cache 切成固定大小的 block。逻辑上一个序列的 KV 仍然是连续 token，但物理上可以分散在 block pool 的不同 block 里。每个序列维护一张 page table 或 block table，记录逻辑 block 到物理 block id 的映射；全局 `BlockAllocator` 管理哪些物理 block 空闲、哪些被占用。这样请求只按实际 token 数增长分配 block，浪费主要限制在最后一个未填满 block。
>
> 所以它的收益不是让 attention 计算复杂度下降，而是显著提高 KV Cache 显存利用率，并为 continuous batching、prefix cache、preemption 这类 serving 能力提供底层内存管理基础。

可能追问：

> 你项目里的 BlockAllocator 和 PageTable 分别负责什么？能结合代码说一下吗？

### Q12：项目里的 BlockAllocator 和 PageTable 分别负责什么？请结合代码结构讲一下。

我的原回答：

> BlockAllocator 记录 block 的使用情况，是否使用，table 是用于对实际物理显存位置的映射，用于查找真实的 block 位置

评估：

- 分工大方向正确：`BlockAllocator` 管物理 block，`PageTable` 管序列到 block 的映射。
- 需要更精确：当前 `BlockAllocator` 不是完整记录 block 元数据，而是维护 K/V block pool 和 `_free_list`。它只知道哪些 block id 空闲，缺少 ref count、LRU、computed/cached 等工业级元数据。
- `PageTable` 不是直接保存显存地址，而是保存物理 `block_id` 列表。真正的地址由 `BlockAllocator::get_k_ptr/get_v_ptr` 根据 `block_id`、`layer`、`block_stride`、`layer_stride` 计算。

推荐回答：

> `BlockAllocator` 负责全局物理 KV block 的管理。代码里它初始化时一次性分配两块 device pool：`_pool_K` 和 `_pool_V`，布局是 `[num_blocks, nlayer, block_size, nkvh, dh]`。同时维护一个 `_free_list`，`alloc()` 从 free list 弹出一个可用 `block_id`，`free(block_id)` 把 block id 放回去。它还提供 `get_k_ptr(block_id, layer)` 和 `get_v_ptr(block_id, layer)`，根据 stride 计算某个物理 block 在某一层的 K/V 起始地址。
>
> `PageTable` 是每个 sequence 或 slot 自己维护的逻辑映射表。它保存一个有序的 `block_ids` 数组，`token_pos / block_size` 得到逻辑 block index，再从 `block_ids` 中查出物理 block id；`token_pos % block_size` 得到 block 内 offset。
>
> 所以两者分工是：`PageTable` 解决“这个 token 属于哪个物理 block”，`BlockAllocator` 解决“物理 block 的内存在哪里，以及哪些 block 可分配”。当前实现是 naive 版，只用 free list，没有 ref count、LRU 和 prefix-cache 相关元数据。

可能追问：

> 你说 block pool 布局是 `[num_blocks, nlayer, block_size, nkvh, dh]`，为什么把 nlayer 放在 block 里面？地址怎么计算？

### Q13：block pool 布局是 `[num_blocks, nlayer, block_size, nkvh, dh]`，为什么这样设计？地址怎么计算？

我的原回答：

> 因为每一层都需要进行 attenion 的计算，地址是 addr = num_block * blockstride + nlayer * layerstride

评估：

- 思路正确：每一层 attention 都会产生自己的 K/V cache，所以 block pool 需要区分 layer。
- 公式里的变量需要修正：应该是 `block_id * block_stride + layer_idx * layer_stride`，不是 `num_block` 和 `nlayer`。`num_blocks/nlayer` 是总数量，`block_id/layer_idx` 才是当前访问的索引。
- 如果要定位到某个 token offset，还要再加 `offset_in_block * nkvh * dh * elem_size`。

推荐回答：

> 因为 Transformer 每一层都有独立的 K/V cache，同一个 token 在不同 layer 的 K/V 不一样，所以 block pool 需要同时区分物理 block 和 layer。当前实现把 K 和 V 分成两个 pool，每个 pool 的逻辑布局是 `[num_blocks, nlayer, block_size, nkvh, dh]`。
>
> 地址计算分两级 stride：`layer_stride = block_size * nkvh * dh * elem_size`，表示某一层里一个 block 的字节数；`block_stride = nlayer * layer_stride`，表示一个物理 block 跨所有层的总字节数。所以某个 `(block_id, layer_idx)` 的起始地址是：
>
> ```text
> base + block_id * block_stride + layer_idx * layer_stride
> ```
>
> 如果要定位某个 token 在 block 内的 K/V 行，还要加：
>
> ```text
> offset_in_block * nkvh * dh * elem_size
> ```
>
> 因此完整地址类似：
>
> ```text
> base
> + block_id * block_stride
> + layer_idx * layer_stride
> + offset_in_block * nkvh * dh * elem_size
> ```

可能追问：

> 为什么 K 和 V 要分两个 pool？能不能放在一个 interleaved pool 里？

### Q14：为什么 K 和 V 要分成两个 pool？能不能放在一个 interleaved pool 里？

我的原回答：

> 不行，因为后续要进行 attention 的计算，而且在 vllm 中，二者 block 的维度都不一样

评估：

- “attention 后续会分别读取 K 和 V”这个方向是对的。
- 但“不能放在一个 interleaved pool”说得太绝对。K/V 可以分开存，也可以放在同一个大 buffer 的不同维度或 interleaved layout 中，关键是 kernel 和 stride 描述要匹配。
- 在本项目里 K/V 分两个 pool 是实现选择：地址计算简单，`k_pool`/`v_pool` 分别传给 native paged attention 和 FlashInfer adapter，也便于分别写入和调试。
- “vLLM 中二者 block 维度都不一样”要谨慎。标准 MHA/GQA 中 K/V 的 token、kv_head、head_dim 通常相同；某些后端或模型可能有不同布局，但不要把它作为 K/V 必须分离的核心理由。

推荐回答：

> K 和 V 分成两个 pool 不是理论上必须，而是当前实现选择了比较简单的 SoA 布局。Attention 计算分两步：先用 Q 和 K 做 dot product 得到 score，再用 softmax 后的权重去加权 V。Kernel 访问 K 和 V 的时机、访问模式不一样，所以把 `k_pool` 和 `v_pool` 分开传入，地址计算和代码实现更直接。
>
> 当然，也可以把 K/V 放在同一个大 buffer 里，比如加一个 KV 维度 `[2, num_blocks, nlayer, block_size, nkvh, dh]`，或者做 token 内 interleaved。这样可能减少 allocator 对象数量，某些 kernel 也可能更喜欢统一描述符。但代价是 stride 更复杂，kernel 需要按新的 layout 解析，和 FlashInfer/vLLM backend 对接时也要匹配它们期望的 paged KV layout。
>
> 所以我的项目里分成两个 pool，主要是为了实现简单和访问清晰：`BlockAllocator` 分别维护 `_pool_K`、`_pool_V`，`paged_attention` 也分别接收 `k_pool`、`v_pool`。这不是唯一正确布局，而是一个适合原型实现的保守选择。

可能追问：

> 你手写的 paged attention kernel 是怎么通过 page table 访问非连续 KV 的？

### Q15：手写 paged attention kernel 是怎么通过 page table 访问非连续 KV 的？从 token_pos 到真实 K/V 地址的过程是什么？

我的原回答：

> 通过之前的地址计算进行访存

评估：

- 回答太短，缺少 kernel 执行流程。
- 面试中需要说清楚：kernel 不是先对每个 token 单独查 page table，而是按逻辑 block 遍历；每个 block 从 `block_tables` 取物理 `block_id`，再遍历 block 内 token。
- 要提 GQA：`head_idx / group_size` 得到 KV head。
- 要提 stride：`block_id * pool_block_stride + layer_idx * pool_layer_stride + token_offset * num_kv_heads * head_dim + kv_head_idx * head_dim`。

推荐回答：

> 我的 native paged attention kernel 里，一个 CUDA block 负责一个 `(batch_idx, head_idx)`。首先根据 `seq_lens[batch_idx]` 算出这个序列有多少个逻辑 block：`num_blocks = ceil(seq_len / block_size)`。然后循环遍历逻辑 block index `bi`，通过：
>
> ```text
> block_id = block_tables[batch_idx * max_blocks_per_seq + bi]
> ```
>
> 拿到这个逻辑 block 对应的物理 block id。
>
> 接着根据物理 block id 和 layer 计算当前 block 的 K/V 起始地址：
>
> ```text
> k_base = k_pool + block_id * pool_block_stride + layer_idx * pool_layer_stride
> v_base = v_pool + block_id * pool_block_stride + layer_idx * pool_layer_stride
> ```
>
> 在 block 内再遍历 token offset `t`。GQA 下多个 query head 共享 KV head，所以先算：
>
> ```text
> group_size = num_heads / num_kv_heads
> kv_head_idx = head_idx / group_size
> ```
>
> 然后某个 token 的 K/V 向量地址是：
>
> ```text
> k_vec = k_base + t * num_kv_heads * head_dim + kv_head_idx * head_dim
> v_vec = v_base + t * num_kv_heads * head_dim + kv_head_idx * head_dim
> ```
>
> 这样逻辑上连续的历史 KV 可以分散在不同物理 block 中，kernel 通过 `block_tables` 做一次间接寻址，把每个 block 逐个扫过。

可能追问：

> 你这个 kernel 里的 Online Softmax 是怎么做的？为什么不需要存完整 attention scores？

### Q16：kernel 里的 Online Softmax 是怎么做的？为什么不需要存完整 attention scores？

我的原回答：

> 是通过得到一个 tile 的局部最大值和总的最大值，进行一个 resacle。因为有 casual mask，列数大于行数的部分可以省去

评估：

- “局部最大值/全局最大值/rescale”方向接近 online softmax/FlashAttention 的思想，但和本项目 native decode kernel 的具体实现不完全一致。
- 本项目的 paged decode kernel 不是完整矩阵 prefill attention 的 tile 版，而是一个 query token 对历史 KV 做 decode attention。它逐 token score 更新 `m`、`l`、`acc`，不需要保存完整 scores。
- `causal mask` 这句不适合这里作为核心解释。decode 阶段 query 是当前位置 token，只需要 attend 到 `seq_len` 范围内的历史 KV；kernel 通过 `seq_lens` 和 `tokens_in_block` 控制遍历范围，天然不访问未来 token。
- 拼写：`causal mask`，不是 `casual mask`；`rescale`，不是 `resacle`。

推荐回答：

> Online Softmax 的核心是边遍历 score，边维护当前最大值 `m`、softmax 分母 `l` 和加权输出累加 `acc`。我的 kernel 遍历 page table 里的每个 block，再遍历 block 内 token。对每个 token 先算：
>
> ```text
> score = dot(q, k) * scale
> ```
>
> 然后用新的 score 更新：
>
> ```text
> m_new = max(m, score)
> p = exp(score - m_new)
> correction = exp(m - m_new)
> l = correction * l + p
> acc = correction * acc + p * v
> m = m_new
> ```
>
> 这里 `correction` 的作用是：如果新的最大值变大了，之前累积的分母和输出都要按新的最大值重新缩放，保证数值等价于最后统一做 `softmax(scores) @ V`。
>
> 因为我们只保留 `m/l/acc` 这三个运行状态，所以不需要把所有 attention scores 存下来，也不需要 materialize `[query_len, kv_len]` 的 attention 矩阵。对 decode 来说 query_len 通常是 1，这样非常适合逐 block 扫描 paged KV。
>
> causal 语义在 decode 里主要体现为只遍历当前 `seq_len` 内已经写入的 KV，不会访问未来 token；不是靠保存一个完整 mask 矩阵来做。

可能追问：

> 为什么 Online Softmax 要维护最大值 m？直接累加 exp(score) 不行吗？

### 面试口径：native paged attention kernel vs FlashInfer backend

推荐表述：

> 项目里有两条 paged attention 路径。第一条是我手写的 native paged attention kernel，主要用于学习、验证和 fallback，覆盖 page table 间接寻址、GQA 映射、FP16/BF16 I/O、FP32 累加和 Online Softmax。但它不是生产级实现，并行粒度、访存模式和调度都比较 naive。
>
> 第二条是 FlashInfer adapter，用来对齐更工业化的 paged decode backend。本项目负责 KV block pool、PageTable/BlockAllocator、KV 写入、block table 构建，以及把本项目的 dense block table 转成 FlashInfer 需要的 `indptr/indices/last_page_len` 等元数据；FlashInfer 负责真正高性能的 paged decode attention kernel。
>
> 所以面试中如果讲工业性能，我重点讲 FlashInfer 接入和 KV 管理；如果讲底层原理，我可以解释 native kernel 的 Online Softmax 和 page table 访存，但也会主动说明它的 naive 限制。

native kernel 的不足：

- 一个 `(batch, head)` 一个 CUDA block，粒度简单。
- 没有 FlashAttention 式 tile 级 IO 优化。
- 没有复杂 split-KV、warp specialization、persistent scheduling。
- batch metadata 每次构建和 host/device 拷贝路径仍有优化空间，虽然已有 device-pointer variant 用于 CUDA Graph。
- 更适合作为教学/fallback，而不是对标 vLLM/FlashInfer 的生产 kernel。

### 常见追问：你是单独写了一个 softmax kernel，还是写了 fused attention kernel？

推荐回答：

> 我不是单独写一个 softmax kernel，而是写了一个 native fused paged attention kernel。Online Softmax 是这个 attention kernel 里面的数值稳定归一化策略。
>
> 这个 kernel 融合了三件事：第一，按 page table 遍历 paged KV block，计算 `Q · K` score；第二，边遍历边用 Online Softmax 维护 `m/l`，避免保存完整 attention scores；第三，同时累加 `softmax(score) * V`，最后写出 attention output。所以它不是单独的 softmax 算子，而是 fused attention decode kernel。
>
> 不过我也会说明，这个 native kernel 是原型/fallback，性能上比较 naive。真正工业性能路径我接了 FlashInfer paged decode backend，本项目负责 block pool、page table、metadata 转换和后端接入。

短版：

> 不是单独 softmax kernel，是 fused paged attention kernel。Online Softmax 只是里面的 softmax 计算方式，和 QK dot、page table 访存、V 累加在同一个 kernel 里完成。native 版本用于验证和 fallback，高性能 decode 走 FlashInfer adapter。

### Q18：项目自己的 PageTable/block table 格式和 FlashInfer 需要的 paged KV metadata 有什么差别？你是怎么适配的？

推荐回答：

> 项目内部每个 request/slot 有自己的 `PageTable`，本质是一个有序的 `block_ids` 数组。batch decode 前，项目会把多个 slot 拼成 dense 格式：
>
> ```text
> block_tables: [batch_size, max_blocks_per_seq]
> seq_lens:     [batch_size]
> ```
>
> 例如：
>
> ```text
> seq0 blocks = [5, 8, 2]
> seq1 blocks = [7]
> seq2 blocks = [4, 9]
>
> dense block_tables =
> [
>   [5, 8, 2],
>   [7, 0, 0],
>   [4, 9, 0]
> ]
> ```
>
> 这里 padding 的 `0` 不是有效 block，只是为了形成固定二维表。
>
> FlashInfer paged decode 更偏 CSR/压缩格式，需要：
>
> ```text
> indices:       所有有效 physical block id 连续拼接
> indptr:        每个 sequence 在 indices 里的起止位置
> last_page_len: 每个 sequence 最后一个 block 实际有多少 token
> ```
>
> 上面的例子会转成：
>
> ```text
> indices = [5, 8, 2, 7, 4, 9]
> indptr  = [0, 3, 4, 6]
> ```
>
> `indptr[0:2]` 描述 seq0 使用 `indices[0:3]`，`indptr[1:3]` 描述 seq1 使用 `indices[3:4]`，以此类推。`last_page_len` 根据 `seq_len` 和 `block_size` 计算，比如：
>
> ```text
> last_page_len[b] = (seq_len[b] - 1) % block_size + 1
> ```
>
> 我的 adapter 做的就是把项目内部 dense `block_tables + seq_lens` 转成 FlashInfer 需要的 `indices/indptr/last_page_len`，然后构造 FlashInfer 的 paged KV 描述符，让 FlashInfer kernel 读取同一份 K/V block pool。

不足：

> 当前适配还有 naive 的地方：metadata 转换主要在 host 侧完成，plan/cache 管理也比较简单。更工业化的做法是让 scheduler 直接生成 backend 所需 metadata，减少每轮 dense table 到 CSR metadata 的转换开销。

记忆句：

> 项目自己的格式是 dense `[B, max_blocks]`；FlashInfer 要 CSR 风格的 `indptr + indices + last_page_len`。adapter 的核心工作就是 metadata 转换，并把同一份 paged KV block pool 描述给 FlashInfer。

### Q19：简历里说“典型场景下显存节省 41%”，这个数怎么来的？怎么计算或验证 Paged KV Cache 相比连续 KV Cache 的显存收益？

推荐回答：

> 这个收益主要来自 KV Cache 的分配粒度变化。连续 KV Cache 通常按每个 slot 的最大序列长度预留：
>
> ```text
> memory_contiguous =
>   batch_size * max_seq_len * nlayer * 2(K,V) * num_kv_heads * head_dim * elem_size
> ```
>
> Paged KV Cache 则按实际已经使用的 token 数分配 block：
>
> ```text
> blocks_i = ceil(seq_len_i / block_size)
> memory_paged =
>   sum_i(blocks_i) * block_size * nlayer * 2(K,V) * num_kv_heads * head_dim * elem_size
> ```
>
> 因为 paged cache 的浪费主要是每个请求最后一个 block 的 padding，理论上每个请求最多浪费 `block_size - 1` 个 token 的 KV，而连续 cache 会浪费 `max_seq_len - actual_seq_len` 个 token 的 KV。
>
> 所以节省比例可以按：
>
> ```text
> saving = 1 - memory_paged / memory_contiguous
> ```
>
> 来算。比如在一个多请求 workload 里，连续方案给每个 slot 都预留 `max_seq_len`，但实际请求平均长度明显小于 `max_seq_len`，Paged KV 只按实际 token 数向上取整到 block，就会节省显存。简历里的 41% 应该来自某个典型 workload 下，用实际 request length 分布代入上述公式或通过 benchmark 统计 block allocator 使用量得到。
>
> 验证时我会统计两类指标：第一是理论估算，根据 batch 内每个请求的 `seq_len`、`block_size`、`nlayer/nkvh/head_dim/elem_size` 算 contiguous 和 paged 的 KV memory；第二是运行时统计 `BlockAllocator` 的 `num_total - num_free`，得到实际使用 block 数，再乘以每个 block 的 K/V 字节数。两者对齐后，再用 `nvidia-smi` 或 CUDA memory API 看整体显存变化。

需要诚实补充：

> 当前项目有些路径仍是原型实现。比如单请求 GPU cache 初始化时会预分配所有 block；batch context 才更接近多 slot 共享 block pool 的 paged KV 形态。所以面试时我会把 41% 说成“在按需 block 分配模型或 batch context workload 下的统计结果/估算结果”，而不是说当前所有路径都必然节省 41%。

简短版：

> 连续 KV 按 `batch_size * max_seq_len` 预留；Paged KV 按 `sum(ceil(seq_len_i / block_size)) * block_size` 分配。节省比例是 `1 - paged / contiguous`。Paged 的浪费主要是每个请求最后一个 block 的 padding，而连续方案浪费的是每个 slot 未使用的 maxseq 空间。运行时可以用 allocator 的 used blocks 乘每 block 字节数验证。

### Q20：一次 batch decode 的完整流程是什么？从 active slots、current tokens，到写 KV cache、调用 paged attention、输出 next tokens，中间经过哪些步骤？

推荐回答：

> 在我的项目里，batch decode 指的是多个 active slot 各生成一个 token，把这些请求合成一个 batch 一起执行一次 decode forward。单个请求仍然是 query_len=1，但 batch_size 是 `B=num_active`。
>
> 完整流程可以分成几步：
>
> 1. **准备 active slots 和 current tokens**
>
>    Python/server 或上层调度逻辑传入：
>
>    ```text
>    active_slots:    当前参与 decode 的 slot id 列表
>    current_tokens:  每个 slot 上一步生成、这一步要输入的 token
>    ```
>
>    每个 slot 内部维护自己的 `PageTable` 和 `current_pos`。
>
> 2. **为当前 token 分配 KV block**
>
>    对每个 active slot，检查它的 `PageTable::needs_new_block()`。如果当前 position 正好落在新 block 开头，就从共享 `BlockAllocator` 里 `alloc()` 一个物理 block，并 append 到这个 slot 的 page table。
>
> 3. **构造 batch attention metadata**
>
>    遍历所有 active slot，拿到每个 slot 的 `block_ids` 和 `current_pos`，构造：
>
>    ```text
>    block_tables: [B, max_blocks_per_seq]
>    seq_lens:     [B]
>    ```
>
>    其中 `seq_lens[i] = slot.current_pos + 1`，因为当前 token 的 K/V 也会写入 cache 并参与本步 attention。
>
> 4. **上传输入 token 和 position**
>
>    把 `current_tokens` 拷贝到 `batch_input_ids`，把每个 slot 的 `current_pos` 拷贝到 `batch_pos_ids`。
>
> 5. **执行 decode forward 的 embedding 和每层 transformer**
>
>    先做 embedding 得到 `[B, hidden]`，然后逐层执行：
>
>    ```text
>    RMSNorm
>    Q/K/V Linear
>    RoPE(q, k)
>    写当前 token 的 K/V 到 paged block pool
>    PagedAttention 读取历史 paged KV
>    O projection + residual
>    MLP norm + gate/up/down + residual
>    ```
>
> 6. **把当前 token 的 K/V 写入 block pool**
>
>    对 batch 中每个 slot，根据 `current_pos` 计算：
>
>    ```text
>    block_id = page_table.get_block_for_token(current_pos)
>    offset   = page_table.get_offset_in_block(current_pos)
>    ```
>
>    然后通过 `BlockAllocator::get_k_ptr/get_v_ptr(block_id, layer)` 找到该层该 block 的起始地址，再加上 `offset * kv_row_bytes`，把当前 token 的 K/V 写进去。
>
> 7. **调用 paged attention**
>
>    每层写完当前 token K/V 后，调用 `paged_attention`。它接收：
>
>    ```text
>    query:        [B, local_nh, head_dim]
>    k_pool/v_pool
>    block_tables
>    seq_lens
>    block_size
>    layer_idx
>    ```
>
>    kernel 根据 `block_tables + seq_lens` 对每个请求读取不同长度、物理不连续的历史 KV，输出 `[B, local_nh, head_dim]` attention result。
>
> 8. **LM head 和采样**
>
>    所有层结束后做 final RMSNorm 和 LM head 得到 `[B, vocab]` logits。然后对每个 slot 分别做 argmax 或 sample，得到 `output_tokens[i]`。
>
> 9. **更新 slot 状态**
>
>    对每个 active slot：
>
>    ```text
>    page_table.inc_num_tokens()
>    current_pos += 1
>    ```
>
>    这样下一轮 decode 时，它的 KV cache 长度就增长了一个 token。
>
> 总结一句：batch decode 不是一个请求一次生成多个 token，而是多个请求各生成一个 token；PagedAttention 负责在同一个 batch 中处理不同请求各自不同长度、不同 block 映射的历史 KV。

不足：

> 当前实现仍比较 naive：active slots 由上层传入，没有完整的 token-budget/KV-budget scheduler；`block_tables` 是 host 侧 dense metadata，短请求会有 padding；prefill 也不是 chunked prefill。因此它是 batch decode 原型，不是完整 vLLM continuous batching scheduler。

### Q21：batch decode、continuous batching、chunked prefill 分别是什么？项目目前实现到了哪一步，没实现哪些？

推荐回答：

> 这三个概念层级不一样。
>
> **batch decode** 是一个计算执行形态：多个已经完成 prefill、正在生成的请求，每个请求各输入一个 current token，一起组成 batch 做一次 decode forward，输出每个请求的 next token。单个请求仍然是 query_len=1，只是 batch_size 是多个请求。我的项目已经实现了这个原型：`active_slots + current_tokens -> batch_decode_impl -> output_tokens`。
>
> **continuous batching** 是 serving 调度策略：系统持续接收新请求，每一轮 iteration 都动态决定哪些请求做 decode、哪些请求做 prefill、哪些请求等待或被抢占。它不是固定 batch 跑完再换下一批，而是请求可以动态进入和退出 batch。工业实现通常会基于 token budget、KV block budget、优先级、TTFT/TPOT 目标来调度。
>
> **chunked prefill** 是 continuous batching 里的一个关键能力：长 prompt 不一次性 prefill 完，而是拆成多个 chunk。每轮只处理一小段 prompt，并允许在两个 prefill chunk 之间插入其他请求的 decode，从而避免一个超长 prompt 长时间阻塞 decode，改善在线 serving 的 TPOT 和 tail latency。
>
> 我的项目目前实现的是 batch decode 原型和 batch context：有多个 slot、PageTable、BlockAllocator、paged attention，一轮可以让多个 active slot 各 decode 一个 token；也有 batch prefill，但它是一次性处理完整 prompt，不是 chunked prefill。
>
> 还没有完整实现 vLLM 风格 continuous batching scheduler：没有 WAITING/PREFILLING/DECODING/FINISHED 状态机，没有 token budget/KV budget admission control，没有 preemption 的调度闭环，也没有真正的 chunked prefill。因此我会把当前实现描述为 serving core 的基础执行层，而不是完整 scheduler。

一句话记忆：

> batch decode 是“一轮里多个请求各生成一个 token”；continuous batching 是“每轮动态调度 prefill/decode 请求”；chunked prefill 是“把长 prompt 拆 chunk，让 decode 能插队”。

### Q22：项目里的 snapshot restore / prefix cache 原型是怎么做的？它和 vLLM 的 block-level prefix cache 有什么区别？

我的原回答：

> 我是通过哈希表实现的前缀树，通过检查提示词的序列，如果不匹配就进行计算，匹配的就用之前计算好的结果，快照是将 gpu 计算好的结果放到 cpu 中

评估：

- 方向正确：项目里确实有 Trie/前缀树原型，用 token 序列做前缀匹配；snapshot 是把已计算 KV 从设备侧拷贝到 CPU 侧保存。
- 要说得更精确：TrieNode 的 children 是 `unordered_map<token, child>`，节点上可以挂一个 `LlaisysQwen2CacheSnapshot`。
- 当前项目是“整段 KV snapshot 复用”原型，不是 vLLM 的 block-level automatic prefix caching。
- vLLM 风格 prefix cache 通常以完整 KV block 为粒度，用 block hash 查找和复用，命中后增加 block ref count，而不是把整段 KV 深拷贝到 CPU 再恢复。

推荐回答：

> 项目里 prefix cache 是一个原型实现。Python 或上层可以把一段 token prefix 对应的 KV cache 保存成 snapshot，然后插入到一个 Trie 里。Trie 的每条边是一个 token，节点用 `unordered_map<int64_t, unique_ptr<TrieNode>> children` 存孩子；某个节点如果对应一个可复用前缀，就挂一个 `LlaisysQwen2CacheSnapshot*`。
>
> lookup 时沿着输入 prompt 的 token 序列在 Trie 里向下走，记录最长的、带 snapshot 的节点。如果命中，就拿到这段前缀的 KV snapshot，调用 restore 把 KV cache 恢复到模型或 batch slot，然后只计算剩余未命中的 token；如果不命中，就正常 prefill。
>
> snapshot 本身是一个深拷贝。GPU 路径下 save 时会按 layer、按 token 从 paged block pool 读出 K/V，把它们拷贝到 CPU vector 里；restore 时再从 CPU vector 拷回设备侧 KV cache/block pool。
>
> 这和 vLLM 的 block-level prefix cache 有明显区别。vLLM 更工业化的做法是只缓存完整 block，对每个 block 计算 hash，比如 `hash(parent_hash, block_tokens, extra_keys)`，命中后直接复用物理 KV block，并增加 ref count。这样不需要把整段 KV 拷到 CPU，也能让多个请求共享同一批物理 blocks，还能配合 LRU eviction。我的项目目前是整段 snapshot + Trie 的原型，功能上能演示 prefix 复用思路，但显存复用、引用计数、淘汰策略和 block-level 复用都还没做完整。

一句话记忆：

> 我的实现是“token Trie + 整段 KV snapshot 深拷贝”；vLLM 是“block hash + 物理 KV block 共享 + ref count/LRU”。

可能追问：

> 为什么 snapshot 放 CPU 会慢？如果要改成 vLLM 风格 block-level prefix cache，你会怎么设计？

### Q23：为什么 snapshot 放 CPU 会慢？如果要改成 vLLM 风格 block-level prefix cache，你会怎么设计？

我的原回答：

> snapshot 放 CPU 会有带宽压力，本身 decode 就是 memory bound，如果改成 vllm 形式，paged 的设计就需要加上 ref count 和 LRU

评估：

- 方向正确：CPU snapshot 会带来 host-device 拷贝带宽压力；block-level prefix cache 需要 ref count 和 LRU。
- 需要补充：慢的关键在于 GPU↔CPU PCIe 拷贝和同步，不只是 decode memory-bound。snapshot save/restore 会把多层 K/V 大量数据深拷贝到 CPU，再拷回 GPU，影响 TTFT 和调度。
- vLLM 风格设计还需要 block hash、cached/free/active 状态、只缓存完整 block、hash key 包含额外上下文信息、eviction 只淘汰 ref_count=0 的 cached blocks。

推荐回答：

> snapshot 放 CPU 慢，主要是因为它把 KV cache 做了深拷贝。对一个 prefix，需要按 layer 把 K/V 从 GPU block pool 拷到 CPU vector；restore 时又要从 CPU 拷回 GPU。这个过程会占 PCIe/Host-device 带宽，并且通常伴随同步，容易拉高 TTFT。KV cache 本身数据量又很大，层数、head 数、prefix 长度一上来，拷贝成本就很明显。
>
> 如果改成 vLLM 风格 block-level prefix cache，我会把 `BlockAllocator + PageTable` 升级成显式 `KVCacheManager`。每个物理 block 有元数据：
>
> ```text
> block_id
> ref_count
> num_tokens
> computed/cached 状态
> block_hash
> last_access_time
> ```
>
> prefix cache 只缓存完整 block。每个完整 block 计算：
>
> ```text
> block_hash = hash(parent_hash, block_tokens, extra_keys)
> ```
>
> `extra_keys` 里要包含 model/tokenizer/lora/cache_salt 等，避免不同上下文误复用。全局维护 `hash -> block_id` 的表。新请求 admit 时，按 token block 逐个查 hash，命中的 block 增加 `ref_count`，直接挂到该请求的 page table；未命中的部分继续 prefill 并写入新 block。
>
> 请求结束时，不立即清空可缓存 block，而是把 `ref_count` 减一。`ref_count == 0` 且 cached 的 block 可以进入 LRU 队列；当 free block 不够时，只淘汰 LRU 中 `ref_count == 0` 的 cached block。这样 KV 仍在 GPU block pool 里，不需要 CPU snapshot 的大规模拷贝，也能让多个请求共享同一批物理 KV blocks。

一句话记忆：

> CPU snapshot 是“KV 深拷贝到 CPU 再拷回 GPU”，慢在 PCIe/同步；vLLM 风格是“完整 block hash 命中后直接共享 GPU 物理 block”，核心是 block hash、ref count 和 LRU。

### Q24：prefix cache 为什么一般只缓存完整 block？最后一个 block 只填了一半，为什么不缓存？

我的原回答：

> 不知道

评估：

- 这是 vLLM 风格 prefix cache 的关键点，需要掌握。
- 核心原因：完整 block 的 hash、复用、引用计数和淘汰语义更稳定；半满 block 还可能继续追加 token，内容不稳定，容易造成错误复用或复杂的 copy-on-write。

推荐回答：

> 一般只缓存完整 block，是为了让 cache key 和 block 内容稳定。prefix cache 的核心是对一个完整 token block 计算 hash，比如：
>
> ```text
> block_hash = hash(parent_hash, block_tokens, extra_keys)
> ```
>
> 只有 block 填满后，`block_tokens` 才固定下来，hash 才稳定，可以安全地加入 `hash -> block_id` 表，并被其他请求复用。
>
> 最后一个半满 block 通常不缓存，因为它后面可能继续 append 新 token，内容还会变化。如果把半满 block 放进 cache，就会遇到几个问题：第一，hash 会随着追加 token 改变，cache key 不稳定；第二，多个请求共享一个半满 block 后，如果其中一个请求继续写入，就需要 copy-on-write，否则会污染另一个请求；第三，半满 block 的匹配粒度更细，会让元数据、引用计数和淘汰逻辑复杂很多。
>
> 所以工业实现通常只缓存完整 block。未满的 tail block 可以留给当前请求继续写，但不进入 prefix cache。这样每个 cached block 都是不可变的、可 hash 的、可被多个请求安全共享的单位。

一句话记忆：

> 完整 block 内容不可变，hash 稳定，能安全共享；半满 block 还会继续写，hash 不稳定，复用时需要 copy-on-write，复杂且容易错。

### Q25：PagedAttention 里 block size 怎么选？block size 太大或太小分别有什么问题？

我的原回答：

> 我是用的64，太大会容易出现碎片化问题，太小会有更多的launch开销

评估：

- “太大会浪费更多 tail padding”方向正确。
- 项目代码里默认 block size 多处是 16，例如 `SINGLE_BLOCK_SIZE = 16`、`BatchContext block_size(16)`；如果面试说 64，需要确认是否是某次实验配置，否则容易被代码追问打穿。
- “太小会有更多 launch 开销”不够准确。PagedAttention 通常不是每个 block 单独 launch 一个 kernel，而是一个 attention kernel 内遍历多个 KV blocks。block size 太小主要导致 page table 更长、metadata 更多、间接寻址次数更多、kernel 内循环/调度开销更大。

推荐回答：

> block size 是一个折中参数。它决定一个物理 KV block 里放多少个 token。block size 太大时，每个请求最后一个 block 的 padding 浪费会变大，尤其是大量短请求或长度分布很散时，KV 利用率下降；prefix cache 也只能缓存完整 block，block 太大还会降低 prefix cache 的命中粒度。
>
> block size 太小时，tail padding 浪费会变小，但每个序列需要更多 block，page table 更长，attention kernel 需要更多次间接寻址和更多 metadata 处理；allocator 管理的 block 数也会增加，调度和维护成本上升。它不一定直接增加 kernel launch 次数，因为通常是一个 kernel 内遍历多个 blocks，但会增加 kernel 内部 page/block 级遍历开销。
>
> 我的项目里默认采用 16-token block，这是 vLLM/PagedAttention 里比较常见的折中：tail 浪费最多 15 个 token，同时 block 内 K/V 仍然有一定连续性，适合 kernel 顺序读取。实际生产中 block size 需要结合模型 head_dim、batch 长度分布、prefix cache 命中粒度和后端 kernel 实现一起 benchmark。

一句话记忆：

> block 大：metadata 少、连续性好，但 tail padding 浪费和 prefix cache 粒度变粗；block 小：浪费少、粒度细，但 page table 更长、间接寻址和管理开销更高。

### Q26：项目里的 PagedAttention / KV Cache 管理有哪些明显不足？如果继续优化，优先级是什么？

我的原回答：

> 首先是 table 是放在内存中而不是显存中，其实是功能很少，再然后是 attention 算子虽然和 vllm 一样用的 flashinfer，但是 token 的维度不是 csr，我会先先做 table 向 gpu 上的迁移，再从 allocator 的维度上向 vllm 靠齐，加上合适的 meta，拓展功能

评估：

- 方向正确：指出了 metadata 在 host 侧、功能少、FlashInfer metadata 适配不彻底、Allocator 需要加元数据。
- 需要修正术语：“token 的维度不是 csr”应改为“项目内部 block table 是 dense `[B, max_blocks]`，不是 FlashInfer/vLLM 常见的 CSR-style `indptr + indices + last_page_len` metadata”。
- “table 放在内存中而不是显存中”要具体：当前 batch decode 主路径构造 host 侧 `block_tables/seq_lens`；native `paged_attention()` 会在调用内部做 device copy。虽然已有 `paged_attention_device()`，但 batch decode 主路径还没有完全 device-side metadata。

推荐回答：

> 当前不足主要有三类。
>
> 第一，metadata 路径比较 naive。batch decode 里 block table 是 host 侧 dense `[B, max_blocks_per_seq]`，每次 decode 都要构造并传给 attention。相比 FlashInfer/vLLM 更常用的 `indptr + indices + last_page_len`，dense table 有 padding，短请求会浪费 metadata 空间，也不利于后端直接高效调度。后续我会让 scheduler 或 batch context 直接维护 device-side metadata，减少 host/device 搬运，并逐步改成 CSR-style metadata。
>
> 第二，Allocator 功能太少。现在 `BlockAllocator` 只有 K/V pool、stride 和 free list，缺少真正 KVCacheManager 的 block 元数据，比如 `ref_count`、`num_tokens`、`computed/cached` 状态、`block_hash`、`last_access_time` 和 LRU 队列。因此它还不能支持工业级 prefix cache、block sharing、eviction 和 preemption。
>
> 第三，attention backend 接入还不够彻底。虽然有 FlashInfer adapter，但项目内部仍先构造 dense block table，再由 adapter 转成 FlashInfer 需要的 `indices/indptr/last_page_len`。更好的做法是让上层 KV manager/scheduler 原生生成 FlashInfer 所需 metadata，减少转换成本；native kernel 则保留为 fallback。
>
> 优先级上，我会先做 device-side metadata 和 CSR-style block table，减少每轮 host/device metadata 开销；然后把 `BlockAllocator + PageTable` 升级成 `KVCacheManager`，增加 block meta、ref count、LRU 和 hash；最后再补 chunked prefill、prefix cache 和真正 token/KV budget scheduler。

短版：

> 当前主要问题是 metadata 在 host 侧且 dense、Allocator 只有 free list、FlashInfer 接入还要从 dense table 转 CSR。优化优先级是：先把 block table/seq_lens 迁到 device 并改成 CSR-style metadata，再升级 Allocator 为 KVCacheManager，补 ref count、LRU、block hash、cached 状态，最后接 chunked prefill 和完整 scheduler。

### Q27：PagedAttention 或推理链路里，FP16/BF16 混合精度具体体现在哪里？为什么 I/O 用半精度，但中间累加通常用 FP32？

我的原回答：

> 全链路中除了 linear 的 gemv 是 fp32，其他部分都是 fp16 进行的，主要是因为 fp32 精度更高

评估：

- “FP32 精度更高”方向对，但表述不准确。
- 不是“除了 linear GEMV 是 FP32，其他都是 FP16”。项目里 GPU 激活/KV I/O 多用 FP16，部分权重可能 FP16/FP32/INT4；但 attention dot、softmax、RMSNorm reduction、GEMV/GEMM compute、INT4 dequant 后累加等内部计算通常用 FP32 累加。
- logits 通常用 FP32，便于采样/top-k/top-p。
- BF16 的意义也要能说：和 FP16 一样 16-bit 存储带宽，但指数范围接近 FP32，更抗 overflow。

推荐回答：

> 混合精度主要体现在“存储和访存用低精度，归约和累加用 FP32”。在 GPU 推理路径里，激活、KV cache 和很多权重可以用 FP16/BF16 存储，这样显存占用和带宽压力都比 FP32 低。比如 PagedAttention 的 query、K/V pool、output 可以是 FP16 或 BF16 I/O。
>
> 但 attention 里的 `Q · K` 点积、softmax 的 `m/l`、以及 `softmax(score) * V` 的输出累加，我会转成 FP32 做。这是因为这些操作都是长向量归约或指数运算，如果直接用 FP16 累加，误差会随着 head_dim/seq_len 放大，softmax 还容易 overflow/underflow。FP32 累加可以显著改善数值稳定性，最后再 cast 回 FP16/BF16 输出。
>
> 同样，RMSNorm 会用 FP32 计算平方和，GEMM/GEMV 通常用 FP32 compute/accumulate，INT4 W4A16 kernel 也是 INT4 权重在寄存器内反量化后与 FP16 activation 相乘、FP32 累加。LM head 输出 logits 也保持 FP32，方便后续 argmax/top-k/top-p sampling。
>
> 所以不是某一个算子简单地“全 FP32”或“全 FP16”，而是 I/O/storage 用半精度省带宽，核心 reduction/accumulation 用 FP32 保精度。

一句话记忆：

> 半精度负责省显存和带宽，FP32 负责归约、累加和 softmax 稳定性；最后再根据输出需要 cast 回 FP16/BF16 或保留 FP32 logits。

可能追问：

> FP16 和 BF16 有什么区别？为什么有些模型更偏向 BF16？

### Q28：FP16 和 BF16 有什么区别？为什么有些模型更偏向 BF16？

我的原回答：

> BF16 范围和 FP32 一样，但是精度会低，FP16 是 1 符号 5 指数 10 尾数，BF16 是 1 符号 8 指数 7 指数，推理的精度主要还是和范围有关，精度实际上影响不大

评估：

- 基本正确：BF16 指数位和 FP32 一样是 8 bit，动态范围大；FP16 是 5 bit 指数、10 bit 尾数，尾数精度更高但范围更小。
- 小笔误：BF16 是 1 符号、8 指数、7 尾数，不是“7 指数”。
- “精度影响不大”要更谨慎。深度学习模型通常对尾数低几位比较容忍，但不是完全不影响；量化、归一化、logits、长序列累加仍可能受影响。

推荐回答：

> FP16 和 BF16 都是 16-bit 浮点，但位宽分配不同。FP16 是：
>
> ```text
> 1 sign + 5 exponent + 10 mantissa
> ```
>
> BF16 是：
>
> ```text
> 1 sign + 8 exponent + 7 mantissa
> ```
>
> BF16 的指数位和 FP32 一样，所以动态范围接近 FP32，不容易 overflow/underflow；但尾数只有 7 位，精度比 FP16 低。FP16 尾数更多，单个数的相对精度更高，但指数范围小，更容易在训练或某些推理中出现溢出。
>
> 深度学习里很多模型更偏向 BF16，是因为神经网络通常对尾数精度有一定容忍度，但对动态范围比较敏感。尤其是训练、长序列、normalization、attention score 或某些激活分布较大的场景，范围不足可能比尾数少几位更危险。
>
> 推理里如果模型本身是 FP16 权重，FP16 通常也能跑得很好；如果模型训练/权重发布就是 BF16，或者担心数值范围问题，BF16 会更稳。无论 FP16 还是 BF16，关键归约和累加通常仍用 FP32。

一句话记忆：

> FP16 尾数更长、精度更细，但范围小；BF16 范围接近 FP32，更抗溢出，但尾数更短。深度学习往往更怕范围不够，所以 BF16 更稳。

### Q29：GQA 和 MHA/MQA 的区别是什么？在 PagedAttention kernel 里，query head 和 KV head 是怎么对应的？

我的原回答：

> GQA 是一个 Q 头对应多个 KV 头，在我的 qwen2 的 1.5B 模型中是 1 个 Q 头对应 4 个 KV 头，MHA 和 MQA 我不了解，主要对应的方式是按顺序进行 padding

评估：

- 方向需要纠正：GQA 是多个 query heads 共享一个 KV head，不是一个 Q head 对应多个 KV heads。
- Qwen2-1.5B/DeepSeek-R1-Distill-Qwen-1.5B 配置里常见是 `num_attention_heads=12`、`num_key_value_heads=2`，因此 `group_size = 12 / 2 = 6`，即 6 个 Q heads 共享 1 个 KV head，不是 1 对 4。
- “padding”不准确。对应方式是整数分组映射：`kv_head_idx = query_head_idx / group_size`。

推荐回答：

> MHA、MQA、GQA 的区别在于 Q heads 和 KV heads 的数量关系。
>
> **MHA** 是标准 multi-head attention，`num_q_heads == num_kv_heads`，每个 Q head 有自己对应的一组 K/V head。
>
> **MQA** 是 multi-query attention，所有 Q heads 共享同一个 K/V head，也就是 `num_kv_heads = 1`。它能显著减少 KV cache 大小和 decode 阶段读 KV 的带宽。
>
> **GQA** 是 grouped-query attention，介于 MHA 和 MQA 之间。多个 Q heads 分成一组，共享一个 KV head。也就是 `num_q_heads > num_kv_heads`，并且：
>
> ```text
> group_size = num_q_heads / num_kv_heads
> kv_head_idx = q_head_idx / group_size
> ```
>
> 以 Qwen2-1.5B 的常见配置为例，`num_attention_heads = 12`，`num_key_value_heads = 2`，所以 `group_size = 6`。也就是 Q heads 0~5 共享 KV head 0，Q heads 6~11 共享 KV head 1。
>
> 在我的 paged attention kernel 里也是这样映射的：每个 CUDA block 负责一个 `(batch, q_head)`，先通过 `kv_head_idx = head_idx / group_size` 找到对应的 KV head，然后在 K/V block 内访问 `kv_head_idx * head_dim` 位置的 K/V 向量。

一句话记忆：

> MHA 是 Q/KV head 一一对应；MQA 是所有 Q 共享一个 KV；GQA 是一组 Q 共享一个 KV，kernel 里用 `kv_head = q_head / group_size` 映射。

可能追问：

> GQA 为什么能减少 KV Cache 显存和 decode 带宽？代价是什么？

### Q30：GQA 为什么能减少 KV Cache 显存和 decode 带宽？代价是什么？

我的原回答：

> 在 attnion 环节中，对于一个 score 的计算，可以多次复用一个 Q，对于 flashattention 来说，是先遍历 Q 的，Q 头的减少也可以减少 Q 矩阵的访问

评估：

- 需要纠正核心概念：GQA 通常不减少 Q heads，而是减少 KV heads。`num_attention_heads` 仍然是模型的 Q head 数，`num_key_value_heads` 更少。
- GQA 的主要收益来自 K/V cache 变小，以及 decode 阶段读取历史 K/V 的带宽下降，不是来自 Q 矩阵访问下降。
- “复用一个 Q”也不准确。实际是多个 Q heads 共享同一个 K/V head。

推荐回答：

> GQA 能省显存和带宽，是因为它减少了 KV heads 的数量，而 Q heads 数量通常保持不变。KV cache 的大小和 `num_kv_heads` 成正比：
>
> ```text
> KV cache memory
> = seq_len * nlayer * 2(K,V) * num_kv_heads * head_dim * elem_size
> ```
>
> 在 MHA 里 `num_kv_heads == num_q_heads`；在 GQA 里多个 Q heads 共享一个 KV head，所以 `num_kv_heads` 更少。比如 Qwen2-1.5B 是 12 个 Q heads、2 个 KV heads，那么 KV cache 相比 12 个 KV heads 的 MHA 理论上减少到 `2/12 = 1/6`。
>
> decode 阶段通常是 memory-bound，需要反复读取历史 K/V。GQA 减少了 K/V head 数，因此每个 token 的历史 K/V 读取量也下降，decode 带宽压力更小。
>
> 代价是表达能力可能下降。因为多个 Q heads 共享同一组 K/V，不如 MHA 每个 head 有独立 K/V 灵活；但比 MQA 所有 Q heads 共享一个 KV head 又更强。因此 GQA 是 MHA 质量和 MQA 性能之间的折中。

一句话记忆：

> GQA 不主要省 Q，而是省 KV：多个 Q heads 共享一个 KV head，所以 KV cache 和 decode 读 KV 带宽按 `num_kv_heads / num_q_heads` 下降；代价是 K/V 表达能力比 MHA 弱一些。

## 句子 3：CUDA 深度优化与 Qwen2 全链路推理

简历原句：

> CUDA 深度优化加速：开发全套推理流程的 CUDA 核心算子模块（RoPE/RMSNorm/SwiGLU/GEMV/Sampling 等）与 Qwen2-1.5B 全链路推理；通过 Nsight Systems 全链路 Profiling 驱动多轮优化——消除冗余 Kernel Launch、Prefill KV 直写 Block Pool（消除 post-prefill 连续→分页拷贝开销）与 CUDA Graph Decode（预分配 cuBLAS Workspace + per-thread 流适配），GPU 单次推理 118s → 2.73s（43 倍加速）。

### Q31：Qwen2 单层 forward 在 decode 阶段大概经过哪些算子？这些算子分别做什么？

我的原回答：

> 最开始是 RoPe，然后是 embeding、然后是 QKV 的映射用的 linear，然后是 attention 和 MLP，最后是 argmax 和 topk

评估：

- 覆盖了部分算子：embedding、QKV linear、RoPE、attention、MLP、argmax/top-k。
- 顺序需要修正：decode 首先是 token embedding；RoPE 发生在 Q/K linear 之后；每层还有 RMSNorm、O projection、residual add、MLP norm、SwiGLU、down projection；最后还有 final RMSNorm 和 LM head。
- `topk` 更准确说是 sampling 里的 top-k/top-p；greedy 时是 argmax。

推荐回答：

> Qwen2 是 decoder-only Transformer。decode 阶段对单个请求是一个 token，对 batch decode 是多个请求各一个 token。整体流程是：
>
> 1. **Embedding**
>
>    当前 token id 查 embedding table，得到 hidden state。
>
> 2. **每层 Transformer block**
>
>    每层大概是：
>
>    ```text
>    residual = hidden
>    attn_norm = RMSNorm(hidden)
>    q = Linear(attn_norm, q_proj)
>    k = Linear(attn_norm, k_proj)
>    v = Linear(attn_norm, v_proj)
>    q, k = RoPE(q, k, position)
>    write k/v to KV cache or paged block pool
>    attn_out = PagedAttention(q, KV cache)
>    hidden = Linear(attn_out, o_proj)
>    hidden = hidden + residual
>
>    residual = hidden
>    mlp_norm = RMSNorm(hidden)
>    gate = Linear(mlp_norm, gate_proj)
>    up   = Linear(mlp_norm, up_proj)
>    act  = SwiGLU(gate, up)
>    hidden = Linear(act, down_proj)
>    hidden = hidden + residual
>    ```
>
>    其中 RMSNorm 做归一化，Q/K/V linear 生成 attention 输入，RoPE 加位置信息，PagedAttention 读取历史 KV，O projection 把多头 attention 输出投回 hidden size，MLP 用 gate/up/down 三个 projection 和 SwiGLU 激活。
>
> 3. **输出层**
>
>    所有层结束后：
>
>    ```text
>    hidden = final RMSNorm(hidden)
>    logits = LMHead(hidden)
>    next_token = argmax 或 temperature/top-k/top-p sampling
>    ```
>
> 所以简历里列的 RoPE/RMSNorm/SwiGLU/GEMV/Sampling 都是这条 decode 链路上的核心 CUDA 算子。

一句话记忆：

> Embedding -> 每层 RMSNorm/QKV/RoPE/KV写入/PagedAttention/O投影/残差/MLP RMSNorm/gate-up/SwiGLU/down/残差 -> final norm -> LM head -> argmax 或 sampling。

可能追问：

> decode 阶段为什么很多 Linear 会退化成 GEMV？prefill 阶段为什么更像 GEMM？

### Q32：decode 阶段为什么很多 Linear 会退化成 GEMV？prefill 阶段为什么更像 GEMM？

我的原回答：

> prefill 是对所有的提示词中的 token 进行前向推理生成首个 token，进行的计算是多个向量形成的矩阵的，而 decode 阶段是单 token 生成，只有一个向量的计算

评估：

- 回答方向正确：prefill 一次处理 prompt 中多个 token，输入是矩阵；decode 每个请求每步只处理一个 token，输入是向量。
- 可以补充矩阵形状：linear 本质是 `Y = X W^T`。prefill 的 `X` 是 `[S, hidden]`，所以是 GEMM；decode 单请求的 `X` 是 `[1, hidden]`，所以退化为 GEMV。
- batch decode 时 `B` 个请求各一个 token，形状是 `[B, hidden]`；如果 B 较小，本质仍偏 GEMV/skinny GEMM，通常 memory-bound。

推荐回答：

> Linear 层本质上是：
>
> ```text
> Y = X W^T
> ```
>
> prefill 阶段一次处理 prompt 中的多个 token。假设 prompt 长度是 `S`，输入 hidden 是：
>
> ```text
> X: [S, hidden_size]
> W: [out_features, hidden_size]
> Y: [S, out_features]
> ```
>
> 这是矩阵乘矩阵，也就是 GEMM。GEMM 的算术强度更高，能更好利用 Tensor Core。
>
> decode 阶段对单个请求每步只输入当前一个 token：
>
> ```text
> X: [1, hidden_size]
> W: [out_features, hidden_size]
> Y: [1, out_features]
> ```
>
> 这就退化成向量乘矩阵，也就是 GEMV。GEMV 每个权重元素通常只读一次、用一次，算术强度低，更容易 memory-bound。
>
> batch decode 时是多个请求各一个 token，`X` 是 `[B, hidden_size]`。如果 B 不大，它仍然更像 skinny GEMM 或 batched GEMV，瓶颈也更偏权重读取带宽。这也是为什么 decode 阶段 weight-only INT4 和 fused dequant-GEMV 会比较有效。

一句话记忆：

> prefill 是多 token 矩阵输入 `[S,H]`，linear 是 GEMM；decode 是单 token 输入 `[1,H]`，linear 退化成 GEMV，通常更 memory-bound。

### Q33：用 Nsight Systems 主要看哪些指标或现象？怎么判断瓶颈来自 kernel launch、memcpy、cuBLAS，还是某个自定义 kernel？

我的原回答：

> 我不知道

评估：

- 这是需要补的知识点，因为简历中明确写了 Nsight Systems profiling。
- 面试不要求你像性能工程师一样背所有指标，但要能说明 Nsight Systems 看时间线、CPU/GPU overlap、kernel launch 间隙、memcpy、cuBLAS 调用、kernel 数量和耗时分布。

推荐回答：

> Nsight Systems 我主要用来看全链路 timeline，而不是单个 kernel 的微架构细节。重点看几类现象：
>
> 第一，看 CPU 和 GPU 时间线是否有大量空洞。如果 GPU stream 上 kernel 很短、kernel 之间有明显 gap，同时 CPU 线程上有很多 CUDA API launch 调用，说明瓶颈可能是 kernel launch overhead 或 CPU 调度开销。这也是 decode 阶段容易遇到的问题，因为每步 token 会触发很多小 kernel。
>
> 第二，看 CUDA API 区域里是否有频繁 `cudaMalloc/cudaFree`、`cudaMemcpy` 或同步 API。如果 profiling 里出现很多 `cudaMalloc`，说明临时 buffer 没有复用；如果 H2D/D2H memcpy 占比高，说明 host-device 数据搬运太多；如果 `cudaDeviceSynchronize` 或 stream synchronize 很多，说明异步流水被打断。
>
> 第三，看 GPU kernel 列表里耗时最长的是 cuBLAS GEMM/GEMV、attention kernel，还是自定义 kernel。cuBLAS 调用通常能在 timeline 里看到对应的 GEMM kernel；如果大部分时间在 cuBLAS，说明瓶颈在 linear 层。如果自定义 RoPE/RMSNorm/Sampling 等小 kernel 数量很多但单个很短，就可能需要融合或 CUDA Graph 减少 launch 开销。
>
> 第四，看 prefill 和 decode 分开分析。prefill 通常是大 GEMM/attention，GPU 利用率更高；decode 是单 token、小 batch，很多 linear 退化为 GEMV，更容易 memory-bound 和 launch-bound。
>
> 我根据这些现象做了几类优化：减少冗余同步和 kernel launch；让 prefill 的 KV 直接写入 block pool，避免连续 KV 到 paged KV 的 post-prefill 拷贝；decode 阶段用 CUDA Graph capture/replay 降低 launch overhead；并给 cuBLAS 预分配 workspace，避免 graph capture 期间内部 malloc。

Nsight Systems 和 Nsight Compute 的区别：

> Nsight Systems 看端到端时间线：CPU 调用、GPU kernel、memcpy、同步、stream overlap，适合找“时间花在哪里”。Nsight Compute 看单个 kernel 的微架构指标，比如 occupancy、memory throughput、warp stall、shared memory、L2 hit rate，适合优化某个具体 kernel。

一句话记忆：

> Nsight Systems 看 timeline：GPU 有没有空洞、CUDA API 有没有 malloc/memcpy/sync、kernel 数量和耗时分布、prefill/decode 谁是瓶颈；Nsight Compute 才是看单 kernel 微架构。

可能追问：

> 你说 CUDA Graph 能减少 launch overhead。CUDA Graph 具体解决什么问题？为什么 decode 阶段适合用？

### Q34：CUDA Graph 具体解决什么问题？为什么 decode 阶段适合用 CUDA Graph？

推荐回答：

> CUDA Graph 解决的是大量小 kernel 反复 launch 带来的 CPU 调度开销。正常 eager 模式下，每一步 decode 都要从 CPU 侧逐个发起 embedding、RMSNorm、QKV linear、RoPE、KV cache 写入、attention、MLP、LM head、argmax 等很多 CUDA kernel 或 cuBLAS 调用。单个 kernel 可能只有几微秒到几十微秒，但每次 launch 都有 CPU overhead，decode 每 token 重复一次，这个开销会被放大。
>
> CUDA Graph 的做法是：第一次运行时把这一串固定的 CUDA 操作 capture 成一张 graph，后续 decode 步骤直接 replay graph。这样 CPU 不需要每次逐个 launch kernel，而是一次 graph launch 提交整条执行图，降低 CPU launch overhead。
>
> decode 阶段适合 CUDA Graph，是因为它的计算拓扑相对固定。每一步都是一个 token，经过同样的模型层和同样的算子序列；输入 token、position、seq_len 会变，但 kernel DAG、buffer 地址和大部分 shape 可以保持稳定。只要提前预分配输入/输出 buffer、block table、seq_lens、cuBLAS workspace，就可以让 graph replay 复用同一套地址。
>
> 项目里为了适配 CUDA Graph，做了几个点：`d_block_tables` 和 `d_seq_lens` 放在 device 上，paged attention 有 device-pointer variant，避免 capture 中 malloc/free；cuBLAS handle 预分配 workspace，并使用 per-thread default stream，避免 cuBLAS 在 graph capture 中触发内部 allocation。

不足：

> 当前 CUDA Graph 主要用于单请求 greedy decode 路径。非 greedy sampling 还会在 graph 外重做 sampling；batch decode 的 dynamic metadata 和 active slots 变化也还没有完全 graph 化。因此这是 decode graph 的原型优化，不是完整 serving graph capture。

一句话记忆：

> CUDA Graph 把每 token 重复的小 kernel 序列 capture 后 replay，减少 CPU 逐 kernel launch 开销；decode 拓扑固定，所以适合 graph。

### Q35：什么是“Prefill KV 直写 Block Pool”？“post-prefill 连续→分页拷贝”是什么意思？

推荐回答：

> prefill 阶段会对 prompt 中所有 token 做一次前向，并在每一层生成这些 token 的 K/V。传统或早期实现里，prefill 往往先把 K/V 写到一块连续 KV cache 里，比如：
>
> ```text
> contiguous K/V: [seq_len, num_kv_heads, head_dim]
> ```
>
> 但后续 decode 的 PagedAttention 需要从 paged block pool 读取 KV，布局是：
>
> ```text
> block pool: [num_blocks, nlayer, block_size, num_kv_heads, head_dim]
> ```
>
> 如果 prefill 先写连续 KV，然后为了 decode 再把这段连续 KV 按 block 切开、拷贝到 paged block pool，这个额外转换就是“post-prefill 连续→分页拷贝”。它发生在 prefill 之后、decode 之前，属于额外的数据搬运，不产生新的数学计算，只是在改 KV cache 的存储布局。
>
> “Prefill KV 直写 Block Pool”就是避免这次额外拷贝：prefill 算出每层 K/V 后，直接根据 PageTable 把每个 token 或每个 block 的 K/V 写入最终的 paged block pool。这样 decode 阶段可以直接用同一份 paged KV cache，不需要再做连续布局到分页布局的转换。

一句话记忆：

> post-prefill 连续→分页拷贝，就是 prefill 先把 KV 写成连续 `[S,nkvh,dh]`，之后再切块搬到 paged block pool；直写 block pool 是 prefill 一开始就写到最终 paged KV 布局，省掉这次搬运。

可能追问：

> 那你改成直写 Block Pool 后，具体数据流发生了什么变化？代码里怎么做？

### Q36：改成 Prefill KV 直写 Block Pool 后，具体数据流发生了什么变化？代码里怎么做？

我的原回答：

> 不知道

评估：

- 需要补数据流，而不是只背概念。
- 注意诚实边界：当前项目 prefill attention 本身仍使用连续的 `k_3d/v_3d` 临时张量做普通 self attention；所谓“直写 Block Pool”主要指 prefill 产生的 K/V 会在每层直接 scatter/copy 到 paged block pool，供后续 decode 使用，避免额外 post-prefill layout conversion。

推荐回答：

> 原来的数据流可以理解成两段：
>
> ```text
> prefill QKV linear
> -> 得到连续 k_3d/v_3d [S, nkvh, dh]
> -> 用连续 KV 做 prefill attention
> -> prefill 后再把连续 KV 按 block 切分搬到 paged block pool
> -> decode 从 paged block pool 读 KV
> ```
>
> 直写 Block Pool 后，数据流变成：
>
> ```text
> prefill QKV linear
> -> 得到当前层连续 k_3d/v_3d 临时张量 [S, nkvh, dh]
> -> 立刻按照 PageTable 的 block_ids，把 k_3d/v_3d scatter/copy 到 BlockAllocator 的 k_pool/v_pool
> -> prefill attention 仍可用临时连续 k_3d/v_3d 保证实现简单
> -> decode 直接复用已经写好的 paged block pool
> ```
>
> 也就是说，当前实现不是 FlashAttention-style 的 paged prefill attention；它只是把 prefill 生成的 K/V 在每层及时写入最终 paged KV cache，避免 prefill 全部结束后再做一次连续布局到分页布局的整体转换。
>
> 代码上，batch prefill 会先根据 prompt 长度给 slot 分配足够 blocks，填好 `PageTable`。每层算出 `k_buf/v_buf` 并 reshape 成：
>
> ```text
> k_3d/v_3d: [S, local_nkvh, dh]
> ```
>
> 然后按 block 遍历：
>
> ```text
> block_id = slot.page_table.block_ids()[bi]
> tok_start = bi * block_size
> ntok = min(block_size, S - tok_start)
> k_dst = allocator.get_k_ptr(block_id, layer)
> v_dst = allocator.get_v_ptr(block_id, layer)
> copy k_3d[tok_start : tok_start + ntok] -> k_dst
> copy v_3d[tok_start : tok_start + ntok] -> v_dst
> ```
>
> 单请求 decode 路径里，当前 token 的 K/V 则通过 `reshape_and_cache` kernel 直接根据 device-side block table 和 position 写入 block pool。

不足：

> 当前 prefill 仍然会生成连续 `k_3d/v_3d` 临时张量，并用普通 self_attention 做 prefill attention，所以还不是完整的 paged/chunked prefill。更工业化的做法是 prefill attention backend 也直接消费 paged KV metadata，并支持 chunked prefill。

一句话记忆：

> 直写 Block Pool 不是说 prefill 完全没有连续临时 K/V，而是说每层 prefill 算出的 K/V 立即按 PageTable 写入最终 paged KV cache，decode 不再需要 post-prefill 的连续到分页整体搬运。

### Q37：为什么 CUDA Graph capture 期间 cuBLAS workspace 会成为问题？为什么要预分配 workspace，并使用 per-thread default stream？

我的原回答：

> 因为 cublas 不方便进行 cuda graph 内的 launch

评估：

- 方向太泛。cuBLAS 不是不能在 CUDA Graph 里用，而是要满足 graph capture 的约束。
- 关键问题是 capture 期间不允许出现不稳定的内存分配/释放、legacy default stream 交叉依赖等。cuBLAS 某些调用可能内部申请 workspace，如果 capture 时触发 `cudaMalloc`，会导致 capture 失败或 graph 不稳定。
- per-thread default stream 是为了避免 legacy NULL stream 的全局同步语义，更适合 graph capture 和多线程场景。

推荐回答：

> cuBLAS 本身可以被 CUDA Graph capture，但前提是执行过程中不能触发不稳定的 runtime 行为。一个常见问题是 cuBLAS 在某些 GEMM/GEMV 调用里可能需要 workspace，如果没有提前设置，它可能在第一次调用时内部申请临时显存。CUDA Graph capture 期间如果发生 `cudaMalloc/cudaFree` 这类 allocation，capture 可能失败，或者 graph 的地址依赖不稳定。
>
> 所以项目里给 cuBLAS handle 预分配了一块固定 workspace，然后调用 `cublasSetWorkspace(handle, workspace, size)`。这样 graph capture 时 cuBLAS 直接使用这块稳定地址的 workspace，不会在 capture 中临时 malloc。
>
> per-thread default stream 的原因是 legacy default stream 有全局同步语义，容易和其他 stream 产生隐式依赖，也不适合多线程 server 场景。项目里把 CUDA Graph capture、cuBLAS handle 和 kernel launch 都放到 per-thread default stream 上，使同一线程内的 H2D、kernel、cuBLAS 调用顺序稳定，也更容易被 graph capture。
>
> 简单说：预分配 workspace 是为了解决 cuBLAS 内部临时分配；per-thread stream 是为了避免 legacy default stream 的隐式同步和跨线程干扰，让 decode graph 的 capture/replay 更稳定。

一句话记忆：

> cuBLAS 可以进 CUDA Graph，但不能在 capture 里临时 malloc；所以提前 `cublasSetWorkspace`，并把 cuBLAS 绑定到 per-thread stream，保证地址和执行流稳定。

可能追问：

> CUDA Graph 为什么要求 buffer 地址稳定？如果 batch size 或 seq_len 变化怎么办？

### Q38：你项目里的 CUDA Graph Decode 是怎么实现的？

我的原回答：

> 包住了 decode 的逻辑。第一次会分配一块 graph 的最大内存池，如果池是空的，就会将 capture 的指针存进去，后续 replay 阶段就会直接按这个进行。不知道输入为什么能变，不知道和 vLLM 的差距。

评估：

- “包住 decode 逻辑”是对的，但要说清楚具体包住的是单请求 GPU decode 的 transformer layer 循环、KV 写入、PagedAttention、MLP、lm_head 和 greedy argmax。
- “第一次分配一块 graph 最大内存池”这个说法不准确。CUDA Graph 本身不是 KV pool，也不是显存 allocator。项目里的 `CUDAGraphRunner` 第一次执行时是 `cudaStreamBeginCapture -> 执行 decode_fn -> cudaStreamEndCapture -> cudaGraphInstantiate`，保存的是 `cudaGraph_t` 和 `cudaGraphExec_t`。
- Graph replay 复用的前提是 kernel 拓扑、参数形状、buffer 地址稳定；每步变化的数据通过固定地址的 device buffer 更新，例如 `input_ids_buf`、`pos_ids_buf`、`d_seq_lens`。
- 和 vLLM 相比，当前实现比较 naive：只支持单请求 decode，没有 batch/shape bucket，没有完整 metadata 设备化，没有 graph pool 管理，也没有把 sampling/调度完整纳入。

推荐回答：

> 我的 CUDA Graph Decode 是在单请求 GPU decode 路径上做的。`ntoken == 1` 时进入 decode，先在 Graph 外把当前 token、position 和 seq_len 拷贝到预分配好的 device buffer，比如 `input_ids_buf`、`pos_ids_buf`、`d_seq_lens`。然后把真正的 decode 计算封装成一个 `decode_fn`，交给 `CUDAGraphRunner::launch`。
>
> `decode_fn` 里面包括 embedding、每层的 RMSNorm、Q/K/V linear、RoPE、`reshape_and_cache` 把当前 token 的 KV 写入 paged block pool、`paged_attention_device` 从 block pool 读历史 KV 做 attention、attention output linear、MLP，最后是 final norm、lm_head 和 greedy argmax。
>
> 第一次调用时，`CUDAGraphRunner` 会在 per-thread default stream 上 `cudaStreamBeginCapture`，执行一遍 `decode_fn`，然后 `cudaStreamEndCapture` 得到 `cudaGraph_t`，再 `cudaGraphInstantiate` 得到可执行的 `cudaGraphExec_t`。后续 decode 如果 graph 已经 capture，就不再逐个 launch kernel，而是直接 `cudaGraphLaunch(_graph_exec)` replay 整个 decode 图。
>
> token、position、seq_len 每步能变化，是因为 Graph 捕获的是固定地址上的操作，而不是固定数值。也就是说，graph 里的 embedding kernel 读取的是 `input_ids_buf` 这个地址，RoPE 读取的是 `pos_ids_buf` 这个地址，PagedAttention 读取的是 `d_seq_lens` 和 `d_block_tables` 这些地址。每步 replay 前，我在 Graph 外把新 token/pos/seq_len 写到同一批 device buffer 里，地址不变，值变了，所以 Graph 可以复用。
>
> 当前实现的不足是，它只覆盖单请求 decode，batch decode 没有真正接入 CUDA Graph；Graph 外仍有 token/pos/seq_len 的 H2D 更新；非 greedy sampling 会在 Graph 外重做；而且没有像 vLLM 那样按 batch size/sequence length 做 CUDA Graph shape bucket，也没有把 block table、slot mapping、scheduler metadata 做成更完整的设备侧结构。因此它能减少 kernel launch overhead，但还不是工业级的 graph replay 调度体系。

一句话记忆：

> 我的 CUDA Graph 不是保存“显存池”，而是保存单请求 decode 的固定执行图；每步只更新固定地址里的 token/pos/seq_len，然后 replay 同一张图。

可能追问：

> 为什么 CUDA Graph 要求地址稳定？如果 batch size 或最大 block 数变化，当前实现会发生什么？

### Q39：为什么 CUDA Graph 要求 buffer 地址稳定？如果 batch size、seq_len 或 block table 形状变化，当前实现应该怎么处理？

我的原回答：

> 不知道

评估：

- 这是 CUDA Graph 的核心约束。Graph 捕获的是 kernel launch 拓扑和 launch 参数，其中包含输入输出指针、grid/block 配置、部分标量参数等。
- Graph replay 时不会重新走 C++/Python 侧的完整调度逻辑，所以不能依赖“每次重新创建 tensor、重新分配 buffer、重新选择 kernel 拓扑”。
- 变化的数据可以写入同一个 device buffer；但如果地址、shape、kernel 数量、grid 配置变化，就应该重新 capture 或选择另一张 graph。

推荐回答：

> CUDA Graph 要求 buffer 地址稳定，是因为 capture 阶段记录的是一串 CUDA 操作的执行图，里面包含 kernel 的输入输出指针、部分标量参数、grid/block 配置以及 kernel 之间的依赖关系。后续 replay 时，CUDA 不会重新执行上层调度逻辑，而是直接按照 capture 时记录的参数提交这张图。如果某个 tensor 重新分配了，地址变了，但 graph 里还拿着旧地址，就可能读写错误数据，甚至访问非法内存。
>
> 但是地址稳定不等于值不能变。比如我的 decode 每步 token、position、seq_len 都会变化，但它们写入的是固定的 `input_ids_buf`、`pos_ids_buf`、`d_seq_lens`。Graph 里的 kernel 每次都从这些固定地址读取，所以只要 replay 前把新值拷进去，就可以复用同一张 graph。
>
> 如果 batch size 变化，通常有两种做法：一种是像 vLLM/Nano-vLLM 那样按 batch size 做 graph bucket，比如 1、2、4、8、16、32 等，每个 bucket capture 一张 graph，实际 batch 选择能容纳它的 bucket；另一种是当前项目这种 naive 做法，只支持固定单请求 decode，batch decode 不 graph 化。
>
> 如果 seq_len 变化但没有改变 kernel 拓扑，可以把 seq_len 放在固定地址的 device metadata 里，让 kernel 运行时读取。我的单请求 decode 就是这样更新 `d_seq_lens`。但如果 seq_len 导致 block table 形状、max blocks、kernel grid 或 workspace 需求变化，就需要重新 capture，或者预先按 max shape 分配静态 buffer，用 bucket 方案复用。
>
> 如果 block table 形状变化，也类似：工业实现会预分配 `[max_batch, max_num_blocks]` 的 block table，然后每次只更新前面有效部分，地址和最大形状不变；当前项目单请求路径里 `d_block_tables` 在 cache 初始化时上传，block table 地址保持不变，所以能支持单请求 graph replay。但 batch 场景里 block table 是按活跃请求动态构造的，目前没有完整 graph 化。

一句话记忆：

> CUDA Graph 复用要求“地址和拓扑稳定、值可以变化”；shape 或拓扑变了就换 bucket graph 或重新 capture。

可能追问：

> 那为什么非 greedy sampling 在你项目里没有放进 CUDA Graph？如果要放进去，需要解决什么问题？

### Q40：为什么非 greedy sampling 在你项目里没有放进 CUDA Graph？如果要把 top-k/top-p sampling 也放进 Graph，需要解决什么问题？

我的原回答：

> 因为加入 temperature 后需要得到 topk 的结果

评估：

- 回答方向只说到一部分。temperature 本身只是对 logits 做缩放，不一定要求 top-k。
- 非 greedy sampling 难放进 Graph 的关键不是单纯 top-k，而是采样链路包含动态参数、随机数状态、top-k/top-p 候选集合、可能的排序/前缀和/截断，以及输出 token 回传到下一轮调度。
- 当前项目的实现里，Graph 内只做 greedy argmax；如果不是 greedy，会在 Graph 外基于 logits 重做 `ops::sample`。

推荐回答：

> 当前项目里 CUDA Graph 只把 greedy argmax 放进去了，因为 argmax 的执行拓扑固定：输入 logits，输出最大 token，kernel 形状稳定，比较适合 capture。
>
> 非 greedy sampling 没放进去，是因为 top-k/top-p/temperature sampling 的控制流更动态。temperature 本身只是 logits scaling，但 top-k 需要选出前 k 个候选，top-p 需要排序或近似排序后做 cumulative probability，再根据阈值截断候选集合；最后还要根据随机数采样。这些步骤可能涉及动态候选数量、随机数状态更新、不同采样参数，以及和 CPU 调度侧的交互。
>
> 在我的代码里，为了保证 Graph 简单稳定，Graph 内默认做 argmax。如果用户传入非 greedy 参数，比如 `top_k > 1` 或 `temperature > 0`，Graph 执行完后会在 Graph 外对 logits 调 `ops::sample` 重做采样。这意味着非 greedy 场景下会多一部分 Graph 外 kernel launch 和调度开销，所以 CUDA Graph 收益会进一步变小。
>
> 如果要把 top-k/top-p sampling 也放进 Graph，需要做几件事：第一，把采样参数 temperature、top_k、top_p 和随机种子/state 放到固定地址的 device buffer；第二，实现 graph-friendly 的 fused sampling kernel，尽量把 scaling、top-k/top-p 过滤、softmax/probability、随机采样融合在一个或少数固定拓扑 kernel 里；第三，避免 CPU 侧动态分配和动态 shape；第四，把输出 token 保留在 device buffer 中，最好和下一轮 decode 的 input buffer 更新衔接起来，减少 D2H/H2D 往返。

一句话记忆：

> greedy argmax 拓扑固定，适合进 Graph；top-k/top-p sampling 有动态候选集和随机状态，当前项目放在 Graph 外，若要 Graph 化需要固定 metadata buffer 和 fused device-side sampling。

可能追问：

> 如果把 sampling 放进 Graph，下一轮 decode 的 input token 能不能完全不回 CPU？这会带来什么调度问题？

### Q41：如果把 sampling 放进 CUDA Graph，下一轮 decode 的 input token 能不能完全不回 CPU？这样会带来什么调度问题？

我的原回答：

> 不知道

评估：

- 从纯计算角度，输出 token 可以留在 GPU 上，并直接作为下一轮 decode 的 input，减少 D2H/H2D 往返。
- 但服务端推理不是只有 kernel replay，还需要 CPU scheduler 管理请求状态、EOS 判断、流式返回、stop words、per-request sampling 参数、batch admission/preemption 等。
- 因此完全 device-side decode loop 可以减少 overhead，但会显著增加调度复杂度。

推荐回答：

> 理论上可以。如果 sampling kernel 在 GPU 上直接生成 `next_token`，下一轮 decode 的 embedding 也从同一个 device-side `input_ids` buffer 读取，那么 token 不需要每步 D2H 回 CPU 再 H2D 传回 GPU。这样可以减少每 token 的同步和 PCIe 拷贝，也更适合 CUDA Graph replay。
>
> 但问题是服务端调度通常需要 CPU 知道每个请求的生成状态。比如某个请求是否生成了 EOS，是否命中了 stop words，是否需要把 token 流式返回给用户，是否达到了 max_new_tokens，batch 里哪些 slot 要退出，哪些新请求可以 admission，哪些请求要 preempt。这些都是 scheduler 层面的决策。
>
> 如果 token 完全不回 CPU，调度器就不知道请求是否结束，也无法及时流式输出。因此工业实现通常会在减少同步和拷贝之间做折中：尽量把采样和部分状态判断放到 GPU 上，但仍需要周期性或按 token 把必要结果回传给 CPU，或者维护一套 device-side request state，再由 CPU 读取压缩后的状态。
>
> 对我这个项目来说，当前实现更简单：decode graph 里做 greedy argmax，之后 CPU 仍然能拿到 token 并驱动下一步推理。它的缺点是每步存在 Graph 外的 token/metadata 更新。如果要继续优化，可以把 sampling、EOS 判断、input token 更新做成 device-side pipeline，并让 scheduler 只读取必要的完成标记和输出 token buffer。

一句话记忆：

> token 留在 GPU 上可以省 D2H/H2D，但服务端 scheduler 仍要知道 EOS、stop、流式输出和 batch 变化，所以需要 device-side state 与 CPU 调度之间做折中。

可能追问：

> CUDA Graph Decode 为什么通常更适合固定 batch/固定 shape？Continuous Batching 的动态请求进出会怎么影响 Graph？

### Q42：CUDA Graph Decode 为什么通常更适合固定 batch / 固定 shape？Continuous Batching 里请求动态进出，会怎么影响 Graph？

我的原回答：

> 固定 batch 等更能保证地址的稳定性，不知道

评估：

- “固定 batch 更容易保证地址稳定”是对的，但还不够完整。
- CUDA Graph 还要求 kernel 拓扑、grid/block 配置、workspace、中间 tensor shape、metadata buffer shape 尽量稳定。
- Continuous Batching 的难点是每步都有请求完成、新请求进入、slot 变化、context length 变化、block table 变化，会破坏固定 shape 假设。

推荐回答：

> CUDA Graph 更适合固定 batch/固定 shape，是因为 capture 时记录的不只是指针地址，还包括 kernel launch 的拓扑、grid/block 配置、kernel 参数、中间 buffer 和依赖关系。如果 batch size 变化，很多 kernel 的 grid 维度会变化，比如 embedding、RMSNorm、linear、attention、sampling 都可能从 B=8 变成 B=13；block table 和 seq_lens 的 metadata 形状也可能变化。这种情况下原来的 graph 不一定还能正确 replay。
>
> Continuous Batching 里，请求是动态进出的。某一步有的请求生成 EOS 退出，有的新请求 admission 进来，不同请求的 context length 不同，block table 长度也不同。这样每步实际 batch size、slot mapping、seq_lens、block tables 都在变。如果每次变化都重新 capture，capture 成本会很高，也失去 CUDA Graph 的意义。
>
> 工业实现通常用 bucket 解决这个问题：预先为一组 batch size 捕获 graph，比如 1、2、4、8、16、32，或者每 16 一个 bucket。运行时真实 batch size 向上取整到某个 bucket，比如真实 B=13 用 B=16 的 graph。输入、position、slot_mapping、seq_lens、block_tables 都预分配到最大 bucket 形状，只更新有效请求部分。无效 slot 用 padding 或 mask 处理。
>
> 对我的项目来说，当前 CUDA Graph 只用于单请求 decode，所以避开了 Continuous Batching 的动态 shape 问题。batch decode 虽然有 batch context 和 slot/page table，但没有真正 graph 化。如果要向 vLLM 靠齐，需要做 batch size bucket、静态 metadata buffer、device-side block table/slot mapping，并在请求进出时选择合适的 graph，而不是每步重新 capture。

一句话记忆：

> Continuous Batching 每步 batch 和 metadata 都会变；CUDA Graph 喜欢固定拓扑，所以工业实现用 batch bucket + 静态 metadata buffer + padding/mask 来复用 graph。

可能追问：

> 你的项目里 CUDA Graph 为什么只适合单请求 decode？如果要支持 batch decode graph，你会怎么改？

### Q43：你的项目里 CUDA Graph 为什么目前只适合单请求 decode？如果要支持 batch decode graph，你会怎么改？

我的原回答：

> 需要做 Graph bucket，按照 batch 的序列从 bucket 中取 graph，而 graph bucket 直接 capture 一整个 bucket 的。

评估：

- 方向正确，核心就是 batch size bucket。
- 需要补充当前项目为什么只适合单请求：目前 `decode_graph.launch(decode_fn)` 只在单请求 `ntoken == 1` 且 `single_allocator` 的路径里；batch context 虽然有 `cuda_graph_session` 字段，但 batch decode 实际没有调用 graph。
- Graph bucket 不只是存 graph，还要为每个 bucket 配套静态 input/metadata/output buffer，并处理 padding slot。

推荐回答：

> 当前项目的 CUDA Graph 只适合单请求 decode，是因为 graph capture 只包住了单请求路径里的 `decode_fn`。这个路径 batch size 固定为 1，`input_ids_buf`、`pos_ids_buf`、`d_seq_lens`、`d_block_tables` 等 buffer 地址固定，layer 数和 kernel 拓扑也固定，所以比较容易 replay。
>
> batch decode 路径虽然有 batch context、slot、PageTable 和 `cuda_graph_session` 成员，但实际 `batch_decode_impl` 没有调用 CUDA Graph。它每步会根据 active slots 动态构造 block tables 和 seq_lens，实际 batch size 也会随着请求进出变化，所以不能直接复用单请求 graph。
>
> 如果要支持 batch decode graph，我会参考 vLLM/Nano-vLLM 做 graph bucket。预先定义一组 batch size bucket，比如 1、2、4、8、16、32，或者按 16 递增。每个 bucket capture 一张固定 shape 的 decode graph。运行时真实 batch size B 选择最小的、能容纳 B 的 bucket，例如 B=13 用 bs=16 的 graph。
>
> 对每个 bucket，需要预分配静态 buffer，包括 `input_ids[bs]`、`positions[bs]`、`slot_mapping[bs]`、`seq_lens/context_lens[bs]`、`block_tables[bs, max_num_blocks]`、`outputs/logits[bs, ...]`。每步调度器只更新前 B 个有效 slot 的 metadata，剩余 padding slot 用 mask 或无效 slot 处理，保证地址和最大 shape 不变。
>
> capture 时直接用这个 bucket 的最大形状跑完整 batch decode；replay 时只改 buffer 里的值，不改地址和拓扑。这样 Continuous Batching 中请求动态进出时，就不是每次重新 capture，而是根据当前 active batch size 选择已有 bucket graph。

一句话记忆：

> 单请求 graph 简单是因为 B=1 固定；batch graph 要靠 bucket，把动态 B 映射到固定 bucket shape，并配套静态 metadata buffer、padding/mask 和 device-side block table。

可能追问：

> Graph bucket 会不会浪费计算？比如真实 batch 是 13，却跑 bs=16 的 graph，这个 trade-off 怎么看？

### Q44：你简历里写“GPU 单次推理 118s → 2.73s，43 倍加速”，这个 43 倍主要来自哪些优化？CUDA Graph 在里面占多大贡献？

我的原回答：

> 有消除 cuBLAS 的句柄复用、CUDA Graph 等，CUDA Graph 有百分之十的提升。

评估：

- CUDA Graph 贡献约 5%～15% 这个说法可以保留。
- “43 倍”不能主要归因于 CUDA Graph。它更像多轮优化叠加，尤其是从非常 naive 的初始实现中移除了大量同步、重复分配、重复拷贝、低效 kernel launch 和非 fused 路径。
- “消除 cuBLAS 的句柄复用”表述不准，应该说“复用 cuBLAS handle / 预分配 workspace / 避免每次创建销毁 handle 或临时 workspace”。

推荐回答：

> 这个 43 倍不是单个 CUDA Graph 带来的，而是从非常 naive 的初始 GPU 路径逐步优化出来的结果。早期实现里有很多工程性开销，比如频繁创建/销毁 cuBLAS handle 或 workspace、decode 每步出现大量小 kernel launch、KV cache 在 prefill 后还要做连续布局到 paged block pool 的额外拷贝、部分路径存在重复 cudaMalloc/cudaFree、以及一些算子没有 fused，导致 CPU launch overhead 和显存搬运都很重。
>
> 后续优化主要包括几类：第一，核心算子 GPU 化和融合，比如 RoPE、RMSNorm、SwiGLU、GEMV、sampling，以及 INT4 W4A16 fused dequant-GEMV；第二，prefill KV 直写 Block Pool，避免 post-prefill 的连续 KV 到分页 KV 的 layout conversion；第三，复用 cuBLAS handle 并预分配 workspace，减少库调用和临时分配开销，也让 CUDA Graph capture 更稳定；第四，decode 阶段引入 CUDA Graph，把固定拓扑的一串 kernel launch 变成一次 graph launch。
>
> 从实测看，CUDA Graph 本身不是 43 倍的主要来源。它在稳定 decode 阶段大概带来 5%～15% 的提升，主要减少 CPU 侧 kernel launch overhead。43 倍更多来自整体路径从 naive 实现变成 GPU 常驻、KV 直写、减少分配/拷贝、核心算子融合后的累计效果。
>
> 所以面试里我会强调：CUDA Graph 是最后阶段的 decode launch 优化，不是最大贡献；最大收益来自消除早期实现中的冗余内存搬运、动态分配和低效算子路径。

一句话记忆：

> 43 倍是全链路从 naive 到 GPU 优化的累计收益；CUDA Graph 只是其中 decode 阶段约 5%～15% 的 launch overhead 优化。

可能追问：

> 你说消除了冗余 Kernel Launch，具体有哪些 kernel launch 是冗余的？是通过 fusion 消掉，还是通过 CUDA Graph 降低 launch overhead？

### Q45：你说“消除冗余 Kernel Launch”，具体哪些 launch 是冗余的？哪些是通过算子融合消掉的，哪些是通过 CUDA Graph 降低 launch overhead 的？

我的原回答：

> 例如 add 和 linear 进行算子融合，反量化和 GEMV 的算子融合，CUDA Graph 降低的是 CPU 端对 kernel 的 launch。

评估：

- 回答方向正确，已经区分了 fusion 和 CUDA Graph。
- 需要进一步强调：CUDA Graph 并没有消掉 graph 内 kernel 本身，而是把多个 kernel launch 的 CPU 提交开销合并成一次 `cudaGraphLaunch`。
- 真正“消掉”的 launch 通常来自算子融合、layout conversion 消除、内存分配/拷贝路径优化。

推荐回答：

> 我这里说的“消除冗余 Kernel Launch”分两类。第一类是真正通过算子融合或数据流改造把某些 kernel 去掉；第二类是 CUDA Graph 没有去掉 kernel 本身，而是降低 CPU 逐个 launch kernel 的 overhead。
>
> 通过 fusion 消掉的例子包括 linear + residual add。Transformer 里 attention output projection 和 MLP down projection 后面通常都要和 residual 相加，如果分开做就是一个 GEMV/GEMM kernel 加一个 add kernel。我的实现里在一些 decode 路径把 linear 的输出和 residual add 融合，避免额外 add kernel 和一次中间结果读写。
>
> 另一个例子是 INT4 W4A16 fused dequant-GEMV。naive 路径会先把 INT4 weight 反量化成 FP16/FP32 临时 buffer，再调用 GEMV/GEMM；这至少包含 dequant kernel、中间 buffer 写入/读取和后续 GEMV。fused kernel 在寄存器里把 INT4 解成 FP16，然后直接参与矩阵向量乘，消除了独立 dequant kernel 和中间 buffer。
>
> 还有一类是 KV cache 数据流优化，比如 prefill KV 直写 Block Pool。它不是简单融合两个数学算子，而是避免 prefill 后再做一次连续 KV 到 paged KV 的 layout conversion，从而减少额外拷贝 kernel 或 memcpy。
>
> CUDA Graph 处理的是另一类问题：decode 阶段每 token 有大量小 kernel，比如 RMSNorm、RoPE、reshape_and_cache、PagedAttention、GEMV、SwiGLU、argmax 等。Graph 不会让这些 kernel 消失，也不会让 GEMV 本身更快；它把固定拓扑的一串 kernel launch capture 成图，后续每 token 用一次 `cudaGraphLaunch` 提交，从而减少 CPU 侧 launch overhead。

一句话记忆：

> Fusion 是少跑 kernel、少写中间 buffer；CUDA Graph 是 kernel 还在，但 CPU 不再一个个 launch。

可能追问：

> INT4 W4A16 fused dequant-GEMV 为什么主要节省带宽？7.4× 带宽节省是怎么算的？

### Q46：INT4 W4A16 fused dequant-GEMV 为什么主要节省带宽？你简历里写 7.4× 带宽节省，这个数大概是怎么算出来的？

我的原回答：

> 因为从最开始的 FP32 的访存到 INT4 的访存减少了 8 倍，一次访存可以多放下八倍的数据，7.4 是因为加上了 scale。

评估：

- 方向正确：weight-only INT4 的主要收益来自权重读取带宽下降。
- 如果对比 FP32 权重，理论压缩是 32 bit / 4 bit = 8×；如果对比 FP16 权重，则理论是 16 bit / 4 bit = 4×。
- 7.4× 这个说法要明确是相对 FP32 权重读带宽，并且需要把 group-wise scale、zero-point/qzeros、对齐 padding、metadata 读取等额外开销算进去。
- fused dequant-GEMV 还节省了中间反量化 buffer 的写入/读取，这是比单纯 INT4 存储更重要的工程收益。

推荐回答：

> INT4 W4A16 GEMV 主要节省的是权重访存带宽。decode 阶段 batch 小，linear 基本是 GEMV，计算强度不高，通常是 memory-bound。每生成一个 token 都要读取大量权重，如果权重还是 FP32，每个 weight 要读 4 bytes；换成 INT4 后，每个 weight 理论上只需要 0.5 byte，所以单看权重本体，带宽需求是 4 / 0.5 = 8 倍下降。
>
> 但实际不是完整 8×，因为 INT4 weight-only 还需要读取量化元数据，比如 group-wise scale，有些格式还有 zero-point/qzeros，并且还会有对齐和 padding 开销。比如 INT4-g128 表示每 128 个 weight 共享一组 scale，读取权重本体之外还要额外读 scale。因此实际有效带宽节省会比理论 8× 略低，我这里统计大约是 7.4×。
>
> Fused dequant-GEMV 的另一个关键点是没有中间反量化 buffer。naive 实现会先把 INT4 权重 dequantize 成 FP16/FP32 临时矩阵，再调用 GEMV，这会产生额外的全局内存写入和读取。我的 fused kernel 在寄存器里把 INT4 解包、乘 scale 转成 FP16/FP32 参与累加，直接完成 GEMV，所以同时减少了权重读取带宽和中间 buffer 带宽。

一句话记忆：

> 理想 FP32→INT4 是 8× 权重带宽下降；实际加上 scale/qzeros/padding 后约 7.4×，fused kernel 还避免了 dequant 中间矩阵的读写。

可能追问：

> W4A16 里的 W4 和 A16 分别是什么意思？为什么 activation 不也量化成 INT4？

### Q48：INT4-g128 里的 g128 是什么意思？group size 为什么会影响精度和性能？

我的原回答：

> 按 128 个元素为一个 group 进行量化得到一个 group scale。得到最大值和量化维度的除，例如 INT4 就是，scale = val/maxval^4，然后乘算得到。group size 越小，颗粒度越小精度损失越少，颗粒度越大，损失越大。颗粒度越小，需要进行的 scale 得到次数越多，访存次数越多，性能越差。

评估：

- g128 的含义回答正确：每 128 个 weight 共享一组量化参数。
- 精度/性能 trade-off 回答正确。
- 量化公式需要纠正。对称 INT4 常见范围是 [-8, 7] 或 [-7, 7]，常用 `scale = max_abs / 7`，量化为 `q = round(w / scale)`，反量化为 `w_hat = q * scale`。
- 如果是 AWQ/非对称格式，还可能有 zero-point/qzeros，反量化通常是 `(q - zero) * scale`，具体取决于格式。

推荐回答：

> INT4-g128 表示 group size 是 128，也就是每 128 个权重共享一组量化参数，最重要的是 scale，有些格式还会有 zero-point 或 qzeros。
>
> 以对称 INT4 为例，一个 group 里先找最大绝对值 `max_abs`，INT4 有效整数范围可以近似看成 [-7, 7]，所以 scale 通常可以设为：
>
> ```text
> scale = max_abs / 7
> q = round(w / scale)
> q = clamp(q, -8, 7) 或 clamp(q, -7, 7)
> w_hat = q * scale
> ```
>
> 如果是非对称量化或 AWQ 格式，反量化会带 zero-point，形式类似：
>
> ```text
> w_hat = (q - zero) * scale
> ```
>
> group size 会影响精度。group 越小，每组 scale 越能贴合这一小段权重的动态范围，量化误差更小；group 越大，很多分布差异比较大的权重共享同一个 scale，某些小权重会被大 outlier 拉低分辨率，所以精度损失更明显。
>
> group size 也会影响性能和压缩率。group 越小，scale/qzeros 数量越多，metadata 访存越多，kernel 里 scale 加载和索引计算也更多，压缩率下降；group 越大，metadata 开销更小，访存更友好，但精度可能变差。g128 是工程上常见的折中：scale 开销不算太大，精度也通常可接受。

一句话记忆：

> g128 就是 128 个 weight 共用一组 scale；group 小精度好但 metadata 多，group 大性能和压缩率好但量化误差更大。

可能追问：

> INT4 两个 4-bit 权重是怎么 packed 到一个 uint8 里的？kernel 里怎么解包？

### Q49：INT4 两个 4-bit 权重是怎么 packed 到一个 uint8 里的？kernel 里怎么解包？

我的原回答：

> 偶数在低四位，奇数在高四位存储，解包的时候，解包第四位后位移动四位，再继续解包。

评估：

- 核心正确：一个 `uint8` 存两个 4-bit weight，偶数列/第一个 weight 放低 4 位，奇数列/第二个 weight 放高 4 位。
- 需要补充 signed/unsigned 的处理。很多格式里 packed int4 是 unsigned nibble，需要配合 zero-point；如果按 signed int4 解释，需要把 0~15 还原到 -8~7。
- 面试中要能写出位运算。

推荐回答：

> INT4 packing 通常是一个 `uint8` 存两个 4-bit 权重。假设第 `j` 个权重对应 packed 数组里的：
>
> ```text
> byte = packed[j / 2]
> ```
>
> 如果 `j` 是偶数，就取低 4 位：
>
> ```cpp
> q = byte & 0x0F;
> ```
>
> 如果 `j` 是奇数，就取高 4 位：
>
> ```cpp
> q = (byte >> 4) & 0x0F;
> ```
>
> 然后根据量化格式做还原。如果是 unsigned int4 + zero-point，反量化大概是：
>
> ```cpp
> w = (q - zero) * scale;
> ```
>
> 如果是 signed int4，需要把 0~15 还原到 -8~7，例如：
>
> ```cpp
> int q_signed = (q >= 8) ? (q - 16) : q;
> w = q_signed * scale;
> ```
>
> 在 fused W4A16 GEMV kernel 里，线程会按列读取 packed byte，解出两个 nibble，在寄存器里乘对应 group 的 scale，得到近似 FP16/FP32 权重值，然后立即和 activation 做乘加累加，不把反量化后的完整权重矩阵写回 global memory。

一句话记忆：

> `byte & 0xF` 取低四位，`byte >> 4` 取高四位；再按 signed 或 zero-point 格式反量化，最后直接参与 GEMV。

可能追问：

> Fused W4A16 GEMV kernel 里，一个线程/warp 通常负责输出矩阵的哪一部分？为什么 decode 阶段更适合 GEMV 而不是 GEMM？

### Q50：Fused W4A16 GEMV kernel 里，一个线程或 warp 通常负责输出矩阵的哪一部分？为什么 decode 阶段更适合 GEMV 而不是 GEMM？

我的原回答：

> 不知道

评估：

- 这个问题要结合 `src/ops/linear/nvidia/linear_nvidia.cu` 里的实现回答。
- 当前项目的 W4A16 GEMV 是按输出行并行：每一行对应输出向量的一个元素 `y[row]`，由 `WARPS_PER_ROW` 个 warp 共同计算这一行和输入向量 `x` 的点积。
- decode 阶段 batch size 通常是 1，每层 linear 的输入是 `[1, K]`，矩阵乘退化成向量乘矩阵，所以 GEMV 比 GEMM 路径更直接。

推荐回答：

> 我的 fused W4A16 GEMV kernel 是按输出行来组织线程的。权重矩阵可以看成 `W[N, K]`，输入是一个 FP16 向量 `x[K]`，输出是 `y[N]`。每个输出元素 `y[row]` 对应权重矩阵的一行和输入向量做点积：
>
> ```text
> y[row] = sum_k dequant(W_int4[row, k]) * x[k]
> ```
>
> 代码里每个 block 固定 256 个线程，每个输出行分配 `WARPS_PER_ROW` 个 warp，也就是 `32 * WARPS_PER_ROW` 个线程一起算这一行。`WARPS_PER_ROW` 会根据 K 的大小选择：K 小时 1 个 warp 一行，K 中等时 2 个 warp 一行，K 大时 4 个 warp 一行。这样 K 维越长，就用更多 warp 来分摊这一行的点积。
>
> 每个线程负责这一行 K 维上的一部分列。kernel 里用 `uint32` 一次读取 4 个 byte，也就是 8 个 INT4 weight；同时用 `half2` 读取对应的 FP16 activation。线程在寄存器里解包 INT4、减 zero/offset、乘 group scale 得到近似权重值，然后和 activation 做乘加，得到局部 `sum`。
>
> 局部 sum 先通过 warp shuffle 做 warp 内归约。如果一行用了多个 warp，再用 shared memory 做跨 warp 归约。最后由一个线程写出 `y[row]`，并且可以顺便融合 bias 和 residual add。
>
> decode 阶段更适合 GEMV，是因为 autoregressive decode 每次只生成一个 token。每层 linear 的输入 shape 是 `[1, hidden]`，也就是一个向量；矩阵乘 `[1, K] × [K, N]` 本质上就是向量乘矩阵，输出 `[1, N]`。这时用 GEMM 的矩阵规模太瘦，计算复用差，cuBLAS GEMM 的开销和通用性不一定划算；专门的 GEMV kernel 可以直接针对 memory-bound 的单 token decode 优化权重读取、解包和归约。
>
> 但 prefill 阶段不同。prefill 输入是 prompt 的多个 token，shape 是 `[S, K]`，这时是标准 GEMM，矩阵规模更大、复用更好，所以项目里 M>1 时会 fallback 到 dequantize + cuBLAS GEMM，而 fused W4A16 GEMV 主要用于 decode 的 M=1 场景。

一句话记忆：

> W4A16 GEMV 是“若干 warp 算一行输出”：线程分摊 K 维点积，寄存器内解包反量化，warp/shared memory 归约；decode 只有单 token，所以 `[1,K]×[K,N]` 退化成 GEMV。

可能追问：

> 为什么这个 GEMV kernel 是 memory-bound？怎么从访存量和计算量解释？

### Q51：为什么这个 W4A16 GEMV kernel 是 memory-bound？怎么从访存量和计算量解释？

我的原回答：

> 从 roofline 模型上来看，计算得到了一个值，超过了那个阈值，说明是 memory-bound。

评估：

- 思路对：可以用 Roofline 模型解释。
- 结论方向需要纠正：Roofline 里通常计算算术强度 `Arithmetic Intensity = FLOPs / Byte`，再和机器平衡点 `Peak FLOPS / Peak Bandwidth` 比较。算术强度低于机器平衡点时是 memory-bound；高于机器平衡点时才更可能是 compute-bound。
- GEMV 的特点是权重通常读一次用一次，数据复用很低，所以算术强度低，天然容易 memory-bound。

推荐回答：

> 可以用 Roofline 模型解释。先看算术强度，也就是每读 1 byte 数据能做多少 FLOPs：
>
> ```text
> Arithmetic Intensity = FLOPs / Bytes
> ```
>
> 再和硬件的机器平衡点比较：
>
> ```text
> Ridge Point = Peak FLOPS / Peak Memory Bandwidth
> ```
>
> 如果算术强度低于这个 ridge point，说明 GPU 的计算单元还没吃满，性能主要受显存带宽限制，也就是 memory-bound；如果高于 ridge point，才更可能是 compute-bound。
>
> GEMV 在 decode 阶段很容易 memory-bound，因为 batch size 是 1，输入是一个向量，权重矩阵每个元素基本只读一次、用一次，几乎没有 GEMM 那种 batch/token 维度上的复用。每个 weight 参与一次乘加，大概 2 FLOPs，但要从显存读权重，还要读 activation 和 scale。即使 INT4 把权重压到 0.5 byte/elem，算术强度仍然不高。
>
> 对 W4A16 GEMV 来说，INT4 weight 减少了权重带宽，但 kernel 仍然主要在搬权重：每个输出行都要扫一遍 K 维 packed weight，并读取对应 activation 和 scale。activation 向量可以被不同输出行复用，但权重矩阵太大，是主要流量。因此性能上限更接近 `显存带宽 / 每 token 需要读取的权重量`，而不是 GPU 的峰值 TFLOPS。
>
> 这也是为什么 INT4 weight-only 对 decode 有明显帮助：它减少的是 memory-bound 路径里最大的权重读取流量。但它不会让 GEMV 变成高复用 GEMM，所以最终吞吐仍然受显存带宽、cache 命中、解包开销和 launch overhead 影响。

一句话记忆：

> GEMV 每个权重基本读一次用一次，FLOPs/Byte 低于 Roofline ridge point，所以是 memory-bound；INT4 优化的是这个 memory-bound 路径里的权重带宽。

可能追问：

> 为什么 prefill 阶段的 GEMM 比 decode 阶段的 GEMV 更容易利用 GPU 算力？

### Q52：为什么 prefill 阶段的 GEMM 比 decode 阶段的 GEMV 更容易利用 GPU 算力？

我的原回答：

> 因为 GEMM 是 compute-bound，同时用的是 cuBLAS 中的 GEMM，使用 Tensor Core，效率很高。

评估：

- 方向正确，但要避免绝对化。GEMM 是否 compute-bound 取决于矩阵形状和算术强度；prefill 的 GEMM 通常比 decode GEMV 算术强度高得多，因此更容易接近 compute-bound，并能更好利用 Tensor Core。
- 核心原因是 prefill 有多个 token，权重矩阵可以被多个 token 复用；decode 单 token GEMV 中权重基本读一次用一次。

推荐回答：

> prefill 阶段输入是 prompt 的多个 token，linear 的输入 shape 通常是 `[S, K]`，权重是 `[K, N]`，所以计算是 GEMM：`[S, K] × [K, N] -> [S, N]`。这里同一块权重会被 S 个 token 复用，矩阵分块后也更适合使用 shared memory、L2 cache 和 Tensor Core，所以算术强度明显高于 decode 的 GEMV。
>
> decode 阶段每次只有一个新 token，输入是 `[1, K]`，矩阵乘退化成 GEMV。每个权重基本读一次只用于一次乘加，复用很低，所以更容易 memory-bound。
>
> 因此 prefill GEMM 更容易让 cuBLAS 使用高效 GEMM kernel 和 Tensor Core，把 GPU 算力吃起来；decode GEMV 则更关注权重带宽、量化压缩、融合和减少 launch overhead。

一句话记忆：

> Prefill 有多 token，权重可复用，GEMM 算术强度高；decode 单 token，权重读一次用一次，GEMV 更 memory-bound。

### Q53：为什么 INT4 权重量化能带来权重压缩，但吞吐提升通常达不到同等比例？比如权重压缩 74%，为什么吞吐只提升 1.8×？

我的原回答：

> 权重访存减少不等于端到端所有开销减少。INT4 kernel 还有解包、scale 读取、反量化计算。decode 里除了 linear/GEMV，还有 attention、norm、RoPE、sampling、kernel launch。GPU 带宽利用率、cache、访存合并、occupancy 都会影响实际吞吐。

评估：

- 回答很好，已经覆盖核心点。
- 可以再补充 Amdahl 定律：只优化了端到端中的一部分，整体加速会被未优化部分限制。
- 还可以强调 weight compression 是存储/权重带宽口径，throughput 是全链路口径，二者不能直接等同。

推荐回答：

> 权重压缩和吞吐提升不是一个口径。INT4-g128 能让权重存储明显变小，也能减少 linear/GEMV 的权重读取带宽，但端到端 decode 不只有权重读取。
>
> 首先，INT4 kernel 自身有额外开销。它要从 packed byte 里解包两个 4-bit 值，要读取 group scale 或 qzeros，还要做反量化，再参与 FP16/FP32 累加。这些操作会消耗寄存器、指令和一定的访存，实际收益会低于理论权重压缩比例。
>
> 其次，decode 全链路里除了 linear/GEMV，还有 PagedAttention、RoPE、RMSNorm、SwiGLU、sampling、KV cache 读写、kernel launch 和调度开销。INT4 主要优化 linear 的权重访存，对其他部分没有同等比例的加速。
>
> 这可以用 Amdahl 定律理解：如果 linear 权重访存占总时间的一部分，那么即使这部分加速很多，整体吞吐提升也会被 attention、norm、sampling、launch overhead 等未优化部分限制。
>
> 最后，实际吞吐还取决于 kernel 的带宽利用率、访存合并、cache 命中、occupancy、寄存器压力和解包效率。比如我的 kernel 尝试过更宽的向量化加载，但因为寄存器压力和 occupancy 下降，反而不一定更快。因此权重压缩 74%，端到端吞吐提升 1.8× 是合理的。

一句话记忆：

> INT4 压缩的是权重和 linear 带宽，但吞吐是全链路结果；解包/scale/attention/norm/launch 等都会吃掉一部分理论收益。

可能追问：

> INT4 量化为什么会影响精度？AWQ 是怎么降低量化误差的？

### Q54：INT4 量化为什么会影响精度？AWQ 是怎么降低量化误差的？

我的原回答：

> 一个是范围变小，一个是精度变小。AWQ 是通过激活值注意力机制，提前对精度影响大的 1% 权重值提前进行量化的反处理，保证量化后该值没有损失。

评估：

- “范围变小、精度变小”方向正确，但要说得更精确：INT4 可表示离散值少，量化步长变大，会产生 rounding/clipping error。
- AWQ 不是注意力机制，也不是简单保护“1% 权重值”。它利用校准数据的 activation 分布，识别对输出更重要/更敏感的通道，并通过 per-channel scaling 保护这些 salient weights/channels。
- AWQ 不能保证完全没有损失，而是降低量化误差对模型输出的影响。

推荐回答：

> INT4 量化会影响精度，主要因为 4-bit 能表示的离散值很少。FP16/FP32 权重是连续近似的浮点数，而 INT4 只能表示有限个整数值，比如 signed INT4 大概是 -8 到 7。量化时需要把一组浮点权重映射到这些离散值上，所以会产生 rounding error；如果某些权重超出当前 scale 能覆盖的范围，还可能产生 clipping error。
>
> group-wise quantization 用 scale 缓解这个问题，但一个 group 内很多权重共享同一个 scale。如果 group 里有 outlier，大权重会决定 scale，小权重的分辨率就会变差，量化误差变大。group size 越大，这个问题通常越明显。
>
> AWQ 的核心思想是 activation-aware weight quantization。它认为并不是所有权重量化误差对最终输出的影响都一样，和 activation 相乘后影响大的通道更重要。因此 AWQ 会用一小批校准数据统计 activation 分布，找出对输出更敏感的通道或 salient weights，然后通过 per-channel scaling 保护这些重要通道，让它们在量化后保留更好的有效精度。
>
> 可以把 AWQ 理解成：不是盲目让每个 weight 的误差都最小，而是让真实推理激活分布下的输出误差更小。它通常会对重要通道放大权重、相应缩放激活或在等价变换中保持数学输出尽量不变，再做 INT4 量化。这样重要权重在 INT4 离散格点里能获得更好的表示，整体精度下降更小。
>
> 但 AWQ 不是完全无损，也不是注意力机制。它是一种基于 activation 统计的 PTQ 方法，目标是在不重新训练或少量校准的情况下，让 weight-only INT4 量化尽量保持模型精度。

一句话记忆：

> INT4 误差来自离散值少、scale 共享和 outlier；AWQ 用 activation 统计找重要通道，通过缩放保护 salient weights，让量化后的输出误差更小。

可能追问：

> AWQ 格式里的 scale/qzeros 在你的 kernel 里是怎么使用的？你的实现和标准 AWQ 兼容到什么程度？

### Q55：AWQ 格式里的 scale/qzeros 在你的 kernel 里是怎么使用的？你的实现和标准 AWQ 兼容到什么程度？

我的原回答：

> 除了量化 scale 的反计算，还有 zero point 和输入输出维度的考虑。

评估：

- 方向是对的：AWQ 兼容主要涉及 scale、qzeros/zero-point，以及权重布局维度转换。
- 需要明确当前项目有两条 INT4 路径：原生 AWQ `I32 qweight + qzeros + scales` 路径，以及项目自定义/转换后的 `U8 packed + scale` fused W4A16 GEMV 路径。
- 当前 fused W4A16 kernel 实际使用的是 `(int4_val - 8) * scale` 的对称/offset 格式，没有在 fused kernel 参数里直接使用 qzeros。原生 AWQ qzeros 路径会先 dequantize 到 FP32 buffer，再走 linear，不是 fused GEMV。

推荐回答：

> AWQ 格式里通常有三类关键数据：packed qweight、scale 和 qzeros。qweight 存 INT4 权重，scale 是 group-wise 的反量化比例，qzeros 存 zero-point，用于非对称量化。反量化一般可以写成：
>
> ```text
> w = (q - zero) * scale
> ```
>
> 或者在我的 fused U8 路径里使用简化的对称 offset 形式：
>
> ```text
> w = (int4_val - 8) * scale
> ```
>
> 当前项目里要分两种情况。第一种是原生 AWQ 权重路径，权重 dtype 是 `I32`，同时有 `scale` 和 `qzeros`。这时项目会调用 `dequantize_awq_int4(dq_buf, w, qz, sc, group_size)`，先按照 AWQ 的 qweight/qzeros/scales 做反量化，得到 FP32 权重 buffer，再调用普通 `linear`。这条路径兼容 AWQ 的 qzeros 和布局，但不是 fused W4A16 GEMV。
>
> 第二种是我自己 fused W4A16 GEMV 使用的 `U8 packed` 路径，权重布局是 `[out_features, in_features / 2]`，scale 是 `[out_features, num_groups]`。kernel 里每次从 packed byte 取低四位/高四位，做 `(val - 8) * scale[row][k / group_size]`，然后立刻和 FP16 activation 相乘累加。这条路径没有直接使用 qzeros，所以更准确地说是兼容 AWQ-like 的 packed/scale 加载格式，或者需要在离线转换时把 AWQ 的 zero-point 信息折算/转换到当前 kernel 支持的格式。
>
> 维度上也要注意 AWQ 常见 qweight 布局可能是 `[in_features, out_features / pack_factor]`，scales 可能是 `[num_groups, out_features]`；而我的 fused kernel 更希望权重按输出行存成 `[out_features, in_features / 2]`，scales 是 `[out_features, num_groups]`。所以加载 AWQ 时需要处理 transpose、pack 方向、group 维度和 TP 切分，否则 row/col 对不上，结果就会错。
>
> 因此面试中我会说：项目对 AWQ 的兼容是分层的。原生 AWQ 路径支持 qzeros，但性能上走 dequantize + linear；高性能 fused W4A16 GEMV 路径主要支持转换后的 U8 packed + per-group scale 格式，对 qzeros 的支持还不完整，这是后续可以改进的地方。

一句话记忆：

> AWQ 反量化核心是 `(q - zero) * scale`；我的原生 AWQ 路径能用 qzeros 但不 fused，fused W4A16 路径目前主要用 `(val - 8) * scale`，更像 AWQ-like 转换格式。

可能追问：

> 你如果要让 fused W4A16 kernel 原生支持 AWQ qzeros，需要怎么改？

### Q56：如果要让 fused W4A16 kernel 原生支持 AWQ qzeros，你会怎么改？

我的原回答：

> 不知道

评估：

- 这是后续优化方向，不是当前已实现功能，面试中不需要讲太深。
- 核心修改是让 fused kernel 直接读取 qzeros，并把反量化公式从固定 offset 的 `(val - 8) * scale` 改成 AWQ 的 `(q - zero) * scale`。
- 还需要处理 AWQ qweight/qzeros 的 pack 布局、group 维度和可能的 transpose。

推荐回答：

> 如果要让 fused W4A16 kernel 原生支持 AWQ qzeros，我会把 qzeros 作为额外参数传进 kernel，并按照和 scale 相同的 group 维度索引 zero-point。当前 fused kernel 是：
>
> ```text
> val = unpack_int4(byte)
> w = (val - 8) * scale[row][group]
> ```
>
> 支持 AWQ 后应该改成：
>
> ```text
> q = unpack_int4(byte)
> zero = unpack_qzero(qzeros, row/group/col 对应位置)
> w = (q - zero) * scale[group][row]
> ```
>
> 或者根据转换后的布局写成：
>
> ```text
> w = (q - zero[row][group]) * scale[row][group]
> ```
>
> 这里最麻烦的不是公式，而是布局。AWQ 的 qweight 通常是 int32 packing，可能按 `[in_features, out_features / pack_factor]` 存，qzeros 也 packed；而我的 fused kernel 当前希望 `[out_features, in_features / 2]` 的 row-major U8 packed。要么离线把 AWQ qweight/qzeros 转成 kernel 友好的布局，要么在 kernel 里按 AWQ 原生布局计算索引，但后者会让访存更复杂。
>
> 为了性能，我更倾向于离线转换：加载 AWQ 后，把 qweight 转成 `[out, in/2]`，qzeros 转成 `[out, num_groups]` 或者 packed zero 的 kernel-friendly layout，scale 转成 `[out, num_groups]`。这样 fused kernel 只是在每个 group 多读一个 zero-point，然后反量化时使用 `(q - zero) * scale`，不会破坏主体 GEMV 结构。
>
> 这仍然会带来额外 qzeros 访存和解包开销，所以需要 benchmark，看原生 qzeros 支持和离线折算/对称转换哪个更划算。

一句话记忆：

> 原生支持 AWQ qzeros，就是 kernel 多读 group zero-point，把 `(val - 8) * scale` 改成 `(q - zero) * scale`；难点主要在 qzeros/qweight 的 packed 布局和访存效率。

可能追问：

> 为什么你当前 M>1 的 INT4 prefill 不走 fused W4A16 GEMV，而是 fallback 到 dequantize + cuBLAS GEMM？

### Q57：为什么你当前 M>1 的 INT4 prefill 不走 fused W4A16 GEMV，而是 fallback 到 dequantize + cuBLAS GEMM？

我的原回答：

> 因为 prefill 是 GEMM 而不是 GEMV。

评估：

- 回答正确，但需要补充为什么 GEMV kernel 不适合 M>1。
- 当前 fused W4A16 kernel 是为 decode 的 M=1 单向量输入设计的；prefill 输入是多个 token `[M, K]`，更适合 GEMM。
- fallback 到 dequantize + cuBLAS GEMM 是 naive 但合理的工程折中：实现简单，并利用 cuBLAS/Tensor Core 处理大矩阵。

推荐回答：

> 因为 prefill 阶段不是单 token，而是 prompt 中多个 token 一起前向，linear 输入 shape 是 `[M, K]`，M 通常大于 1，所以计算是 `[M, K] × [K, N] -> [M, N]` 的 GEMM。
>
> 我的 fused W4A16 GEMV kernel 是专门为 decode 的 M=1 写的：输入是一个向量 `x[K]`，每个输出行由若干 warp 做一次点积。如果 prefill 阶段直接把 M 个 token 拆成 M 次 GEMV，相当于重复扫描权重 M 次，权重复用很差，也无法充分利用 Tensor Core，整体效率会很低。
>
> prefill 的 GEMM 有更高的算术强度，同一块权重可以被多个 token 复用，cuBLAS GEMM 能用成熟的 tiling 和 Tensor Core，所以当前项目在 M>1 时选择 fallback：先把 INT4 权重 dequantize 成 FP32 buffer，再调用普通 linear/cuBLAS GEMM。这不是最优的 INT4 prefill 实现，但工程上简单稳定，而且 prefill 相比 decode 不是每 token 都反复执行同样次数，所以可以接受。
>
> 如果后续要优化，可以写真正的 W4A16 GEMM kernel，或者接入 Marlin、CUTLASS、TensorRT-LLM 这类支持 weight-only INT4 GEMM 的高性能实现，在 GEMM tile 内完成 INT4 解包、scale 反量化和 Tensor Core 友好的计算。

一句话记忆：

> Decode 是 M=1 GEMV，prefill 是 M>1 GEMM；把 prefill 拆成多次 GEMV 会重复读权重，当前 fallback 到 dequantize + cuBLAS GEMM 是简单但不最优的折中。

可能追问：

> 如果要实现高性能 INT4 GEMM，和你现在的 INT4 GEMV kernel 相比，难点在哪里？

### Q58：什么是 Tensor Parallel？为什么大模型推理里要做 Tensor Parallel，而不是只做 Data Parallel？

我的原回答：

> TP 是张量并行，是指在推理环节中对权重张量进行切分，主要是 qkv 映射和 mlp 的映射环节对权重的切分，并行计算能提高推理速度，减少单卡的显存压力，DP 不了解。

评估：

- TP 的核心回答正确：把模型权重张量按维度切到多张卡上，每张卡算一部分 linear/attention/MLP。
- 需要补充 TP 和 DP 的本质区别：DP 是复制完整模型、切 batch；TP 是切同一个模型内部的权重和中间激活。
- 推理场景下，如果单卡放不下模型，DP 不能解决，因为 DP 每张卡都需要完整模型；TP 可以降低单卡权重和 KV cache 压力。
- TP 会引入通信，比如 Row Parallel 后的 AllReduce、Column Parallel 后的后续拼接/对应切分。

推荐回答：

> Tensor Parallel 是把同一个模型内部的大矩阵权重按张量维度切到多张 GPU 上。比如 QKV projection、attention output projection、MLP 的 gate/up/down projection 这些大 linear 层，可以把权重按输出维或输入维切分，让每张卡只保存和计算其中一部分。这样可以降低单卡显存压力，也能让多张卡并行完成同一层的计算。
>
> Data Parallel 不一样。DP 是每张卡复制一份完整模型，然后把不同请求或不同 batch 切到不同卡上独立推理。DP 可以提高整体服务吞吐，但每张卡仍然要放完整模型，所以如果模型本身单卡放不下，DP 解决不了显存问题。
>
> TP 更适合解决“单个模型太大或单请求 latency 要降低”的问题。它把一个请求的一层计算拆到多卡并行执行，代价是每层之间需要通信同步，比如 AllReduce、AllGather 或 ReduceScatter。DP 更适合模型单卡能放下、但请求量很大时横向扩吞吐。
>
> 在我的项目里，TP 主要参考 Megatron-LM 的算子级并行方式，对 QKV/MLP 等 linear 权重做 Column Parallel 和 Row Parallel 切分，并配合通信抽象层实现 AllReduce 等集合通信。

一句话记忆：

> DP 是“每张卡一份完整模型，切请求/切 batch”；TP 是“一个模型切到多张卡，切权重和中间计算”。DP 扩吞吐，TP 降单卡显存并并行同一次推理。

可能追问：

> Megatron-LM 里的 Column Parallel 和 Row Parallel 分别怎么切 linear？为什么有的地方需要 AllReduce，有的地方不需要？

### Q59：Megatron-LM 里的 Column Parallel 和 Row Parallel 分别怎么切 linear？为什么有的地方需要 AllReduce，有的地方不需要？

我的原回答：

> Column Parallel 在 qkv 的映射部分，最后在 attention 的映射部分进行 Row Parallel。列并行得到的是部分矩阵的完整结果不需要进行 allreduce，只有行切分得到的是完整矩阵部分结果，需要 allreduce，MLP 同样是先列后行。

评估：

- 回答方向正确：QKV 和 MLP gate/up 通常用 Column Parallel，attention o_proj 和 MLP down_proj 通常用 Row Parallel。
- 需要把维度讲严谨：Column Parallel 是按输出维/权重列切分，Row Parallel 是按输入维/权重行切分，具体取决于权重矩阵记号。
- 更准确地说：Column Parallel 每张卡得到输出 hidden 的一个 slice，不需要立刻 AllReduce；Row Parallel 每张卡计算完整输出的一部分加和项，需要 AllReduce sum 得到完整输出。

推荐回答：

> 对一个 linear，可以写成：
>
> ```text
> Y = X W
> X: [B, in]
> W: [in, out]
> Y: [B, out]
> ```
>
> Column Parallel 是按输出维切 `W`，也就是：
>
> ```text
> W = [W1, W2, ..., Wp]
> Yi = X Wi
> ```
>
> 每张卡拿到 `Y` 在输出维上的一个 slice，比如 `[B, out/p]`。这个结果本身就是自己负责的那部分输出，不需要做 sum，所以不需要 AllReduce。如果后续算子也能按这个输出维分片继续算，就可以不 AllGather，直接保持 shard 状态。
>
> Row Parallel 是按输入维切 `W`，同时输入 `X` 也按输入维切：
>
> ```text
> X = [X1, X2, ..., Xp]
> W = [W1; W2; ...; Wp]
> Yi_partial = Xi Wi
> Y = sum_i Yi_partial
> ```
>
> 每张卡算出来的是完整输出 `Y` 的一个 partial sum，所以必须对各卡结果做 AllReduce sum，才能得到正确的完整输出。
>
> 在 Transformer 里通常是 Column -> Row 成对出现。Attention 里 Q/K/V projection 用 Column Parallel，把不同 head 或 head slice 分到不同卡上，各卡可以独立做本地 attention；attention output projection `o_proj` 用 Row Parallel，把各卡的 attention 输出分片投回 hidden，并通过 AllReduce 得到完整 hidden。MLP 里 gate_proj/up_proj 用 Column Parallel，得到中间维度的分片并本地做 SwiGLU；down_proj 用 Row Parallel，把分片中间激活投回 hidden，再 AllReduce。
>
> 所以一句话：Column Parallel 切输出，结果是输出 shard，通常不需要 AllReduce；Row Parallel 切输入，结果是 partial sum，需要 AllReduce。

一句话记忆：

> QKV/gate/up 用 Column 切输出，不立刻通信；o_proj/down 用 Row 切输入，算完 partial hidden 后 AllReduce sum。

可能追问：

> QKV Column Parallel 后，attention 的 heads 是怎么分到不同 GPU 上的？GQA 下 KV heads 又怎么切？

### Q60：QKV Column Parallel 后，attention heads 是怎么分到不同 GPU 上的？GQA 下 KV heads 又怎么切？

我的原回答：

> 不知道

评估：

- 这是 TP + GQA 的关键点。QKV Column Parallel 通常按 head 维切输出，每个 rank 负责一部分 Q heads 和对应的 KV heads。
- GQA 里 `num_q_heads > num_kv_heads`，多个 Q head 共享一个 KV head，所以 TP 切分时要同时考虑 `num_heads` 和 `num_kv_heads`。
- 最简单实现要求 `num_heads % tp_size == 0` 且 `num_kv_heads % tp_size == 0`，每张卡拿本地 `local_nh = num_heads / tp_size` 和 `local_nkvh = num_kv_heads / tp_size`。

推荐回答：

> QKV projection 做 Column Parallel，本质上是把 Q/K/V linear 的输出维切开。对 attention 来说，输出维可以理解成 head 维：
>
> ```text
> Q: [B, S, num_heads * head_dim]
> K/V: [B, S, num_kv_heads * head_dim]
> ```
>
> TP 后每个 rank 只保留一部分 heads：
>
> ```text
> local_num_heads = num_heads / tp_size
> local_num_kv_heads = num_kv_heads / tp_size
> ```
>
> 当前 rank 只计算自己的 Q heads、K heads、V heads，并把 KV cache 也按本地 KV heads 存在本 rank 上。attention 计算时，本 rank 的 Q heads 只访问本 rank 的 K/V heads，不需要先 AllGather 全部 heads。
>
> GQA 的特点是 Q heads 比 KV heads 多，多个 Q heads 共享一个 KV head。比如 Qwen2-1.5B 常见是 `num_heads=12`，`num_kv_heads=2`，也就是每个 KV head 服务 6 个 Q heads。TP 切分时要保持这种映射关系。最简单的实现要求 `num_heads` 和 `num_kv_heads` 都能被 `tp_size` 整除，这样每张卡拿连续的一组 Q heads 和 KV heads，本地仍能按 `q_head / group_size` 找到对应的 KV head。
>
> 如果 `num_kv_heads` 不能被 `tp_size` 整除，就会比较麻烦。工业实现可能复制 KV heads、限制 TP size、或者做不均匀切分/特殊映射。当前项目更适合防守成 naive 实现：要求 head 数能整除 TP size，每个 rank 只保存本地 Q/K/V head 和本地 KV cache。
>
> 后面 attention o_proj 用 Row Parallel，把各 rank 的本地 attention 输出 `[B, S, local_num_heads * head_dim]` 投回 hidden 的 partial sum，再 AllReduce 得到完整 hidden。

一句话记忆：

> QKV Column Parallel 是按 head 切；GQA 下每 rank 拿一部分 Q heads 和对应 KV heads，KV cache 也随 KV heads 切；naive 实现要求 Q heads/KV heads 都能被 TP size 整除。

可能追问：

> KV-Cache 切分在 TP 下有什么好处？为什么 attention 阶段通常不需要 AllGather 所有 KV heads？

### Q61：KV-Cache 切分在 TP 下有什么好处？为什么 attention 阶段通常不需要 AllGather 所有 KV heads？

我的原回答：

> KV cache 本身需要很大的显存，切分 TP 下能缓解显存压力，因为 QKV 的 slice 是完整的结果，可以分卡独立进行 attention，得到的结果也是完整的结果。

评估：

- “KV cache 很占显存，TP 切分能缓解显存压力”正确。
- “各卡能独立 attention”正确，但需要说清楚是本地 Q heads 对本地 KV heads 做 attention。
- “得到完整结果”不够严谨。每张卡得到的是本地 attention heads 的输出 slice，不是完整 hidden；后续 o_proj Row Parallel 后通过 AllReduce 得到完整 hidden。

推荐回答：

> KV cache 在长上下文和多并发时显存占用很大，形状大致是：
>
> ```text
> [num_layers, num_tokens, num_kv_heads, head_dim]
> ```
>
> 如果做 TP，并且 QKV 按 head 维切分，那么每个 rank 只负责 `local_nkvh = num_kv_heads / tp_size` 个 KV heads。这样 KV cache 也只需要存本 rank 的 KV heads，单卡 KV cache 显存大约下降到 `1 / tp_size`。
>
> attention 阶段通常不需要 AllGather 所有 KV heads，是因为 Q heads 也被切到了各个 rank 上。GQA/MHA 里每个 Q head 只会访问它对应的 KV head，不需要访问其他 rank 的 Q/K/V heads。只要 TP 切分保持 Q heads 和对应 KV heads 在同一个 rank，本地就能独立完成 attention。
>
> 但每个 rank attention 后得到的是本地 head slice，比如 `[B, S, local_num_heads, head_dim]`，flatten 后是 hidden 的一部分，不是完整 hidden。后面的 attention output projection `o_proj` 使用 Row Parallel，每张卡基于自己的本地 attention 输出计算 partial hidden，然后通过 AllReduce sum 得到完整 hidden states。
>
> 所以通信不是发生在 attention 内部 gather KV，而是发生在 o_proj 之后的 AllReduce。

一句话记忆：

> TP 下 KV cache 按 KV head 切，显存降到 1/tp；本地 Q heads 只看本地 KV heads，所以 attention 不 AllGather，o_proj 后再 AllReduce 回完整 hidden。

可能追问：

> 你项目里 TP 对哪些权重做 Column 切分，哪些权重做 Row 切分？加载权重时怎么切？

### Q63：Row Parallel 后为什么用 AllReduce，而不是 AllGather？AllReduce 的数学含义是什么？

我的原回答：

> 因为矩阵是完整的，但是结果确实部分的，不需要拼接，而且求和得到完整的结果而不是完整的矩阵。

评估：

- 核心正确：Row Parallel 需要的是把 partial sum 相加，不是把不同输出 slice 拼起来。
- 需要把“矩阵完整”改得更精确：Row Parallel 是按输入维切分权重和输入，每个 rank 计算完整输出维度上的一个 partial contribution。
- AllGather 用于拼接不同分片；AllReduce 用于对相同 shape 的 tensor 做逐元素求和并让所有 rank 都拿到结果。

推荐回答：

> Row Parallel 里，linear 可以写成：
>
> ```text
> Y = X W
> X: [B, in]
> W: [in, out]
> Y: [B, out]
> ```
>
> 按输入维切分后：
>
> ```text
> X = [X1, X2, ..., Xp]
> W = [W1; W2; ...; Wp]
> ```
>
> 每个 rank 计算：
>
> ```text
> Y_i_partial = X_i W_i
> ```
>
> 这里每个 `Y_i_partial` 的 shape 都是 `[B, out]`，也就是说每张卡都算出了完整输出维度上的一部分加和项，而不是输出维的不同 slice。真正的结果是：
>
> ```text
> Y = sum_i Y_i_partial
> ```
>
> 所以需要 AllReduce(sum)，对所有 rank 上同 shape 的 partial output 做逐元素求和，并把求和后的完整 `Y` 分发回每个 rank。
>
> AllGather 不适合这里，因为 AllGather 是把不同 slice 拼接起来。如果对 Row Parallel 的 partial outputs 做 AllGather，只会得到多个 `[B, out]` partial tensor 拼在一起，数学上不是正确的 `Y`。

一句话记忆：

> Row Parallel 每张卡算的是同一个输出 tensor 的 partial sum，所以要 AllReduce 求和；Column Parallel 每张卡算的是不同输出 slice，才可能需要 AllGather 拼接。

可能追问：

> AllReduce、AllGather、ReduceScatter 三个集合通信分别适合什么场景？

### Q64：AllReduce、AllGather、ReduceScatter 三个集合通信分别适合什么场景？

我的原回答：

> 我的项目中只用到了 AllReduce，其他两个不知道。

评估：

- 诚实边界正确：当前模型 TP 主链路主要用 Row Parallel 后的 AllReduce。
- 简历如果写“实现 AllReduce、AllGather、ReduceScatter 全套集合操作”，需要确认代码里是否真的有接口和后端实现；模型 forward 是否用到是另一回事。
- 面试中可以说：项目模型路径主要用 AllReduce，AllGather/ReduceScatter 是通信抽象层能力或后续 TP 扩展需要。

推荐回答：

> AllReduce 是对所有 rank 上同 shape 的 tensor 做规约，比如 sum，然后把结果发回每个 rank。Row Parallel linear 后每张卡得到完整 hidden 的 partial sum，所以需要 AllReduce(sum) 得到完整 hidden states。我的 Qwen2 TP 主链路主要用的就是这个。
>
> AllGather 是把每个 rank 持有的不同 slice 收集并拼接，让每个 rank 都拿到完整 tensor。它适合 Column Parallel 后如果后续算子不能继续吃 shard，需要把输出维拼回完整 hidden 的场景。例如每张卡有 `[B, out/tp]`，AllGather 后每张卡得到 `[B, out]`。Megatron 风格里很多地方会避免立刻 AllGather，让后续 Row Parallel 直接消费 shard。
>
> ReduceScatter 可以理解成 AllReduce 的分片版本：先对所有 rank 的 tensor 做 reduce，比如 sum，然后把 reduce 后的结果按维度 scatter 给各个 rank，每个 rank 只保留一片。它适合 sequence parallel 或者想降低激活显存的场景，比如某些层后不需要每张卡都保留完整 hidden，而是只保留 hidden/sequence 的 shard。
>
> 对我的项目来说，模型 forward 里最关键、实际使用的是 Row Parallel 后的 AllReduce。AllGather 和 ReduceScatter 更偏通信抽象层的完整性和后续扩展能力，不能夸大成当前 Qwen2 主路径大量使用。

一句话记忆：

> AllReduce 是“加完每卡都有完整结果”；AllGather 是“拼完每卡都有完整结果”；ReduceScatter 是“加完再切，每卡只拿一片”。

可能追问：

> 为什么 FP16 直通通信能让带宽翻倍？你项目里多精度通信是怎么做的？

### Q65：为什么 FP16 直通通信能让通信带宽压力下降？你项目里多精度通信是怎么做的？

我的原回答：

> 之前推理用的是 FP32，所以 FP16 带宽翻倍了，多精度通信不知道。

评估：

- “FP32 到 FP16 通信数据量减半，所以等效带宽压力下降/吞吐提升”正确。
- 更准确地说，不是物理网络带宽翻倍，而是同样带宽下传输元素数量翻倍，通信时间近似减半。
- 多精度通信在项目中通过 `CommDataType` 抽象实现，AllReduce 根据激活 dtype 选择 F32/F16/BF16。

推荐回答：

> FP16 直通通信的核心收益是通信数据量减半。以前 TP AllReduce 如果按 FP32 传，每个元素 4 bytes；现在激活本身是 FP16，就可以直接按 FP16 传，每个元素 2 bytes。同样的 interconnect 带宽下，传同样数量元素需要的字节数减半，所以通信时间理论上接近减半，也可以说等效元素带宽翻倍。
>
> 项目里不是固定把通信当成 `float*`，而是增加了 `CommDataType`，包括 F32、F16、BF16。模型侧 `allReduceIfTP` 会根据当前 `act_dtype` 选择通信类型：FP16 激活用 `CommDataType::F16`，BF16 激活用 `BF16`，FP32 激活用 `F32`。NCCL 后端再把它映射到 `ncclFloat16`、`ncclBfloat16` 或 `ncclFloat32`。
>
> 这样 Row Parallel 后的 hidden states 如果本来就是 FP16，就不需要先转 FP32 再通信，避免额外带宽和转换开销。但数值上要注意，FP16 AllReduce 的累加精度低于 FP32，推理场景通常可以接受；如果对精度敏感，也可以通过环境变量或配置强制 FP32 激活/通信。

一句话记忆：

> FP16 通信不是物理带宽变大，而是每个元素从 4B 变 2B；项目用 `CommDataType` 把 AllReduce 分发到 NCCL/MPI 的 F16/BF16/F32 路径。

可能追问：

> FP16 AllReduce 会不会带来精度问题？什么时候要用 FP32 通信？

### Q66：FP16 AllReduce 会不会带来精度问题？什么时候需要用 FP32 通信？

我的原回答：

> FP16 通信数据量减半，但累加精度和动态范围比 FP32 差。推理阶段通常对这种误差更容忍，尤其 hidden states 本身就是 FP16。多卡数较多、累加项多、模型对数值敏感时，可以切回 FP32。我的项目里可以根据 act_dtype 选择 F16/BF16/F32 通信。

评估：

- 回答很好，核心完整。
- 可以补充 BF16：2 bytes，但指数位和 FP32 一样，动态范围更大，适合作为 FP16 和 FP32 之间的折中。
- 注意区分“通信精度”和“kernel 内部累加精度”：NCCL 按数据类型做 reduce，具体数值误差与 dtype 和 rank 数有关。

推荐回答：

> FP16 AllReduce 确实可能带来精度问题。它的优点是每个元素只有 2 bytes，相比 FP32 的 4 bytes，通信数据量减半；但它的尾数位和动态范围都比 FP32 小，多 rank 求和时会有更大的 rounding error 和溢出/下溢风险。
>
> 推理阶段通常比训练更能容忍这类误差，而且如果模型的 hidden states 本身就是 FP16，那么直接用 FP16 通信可以避免额外 cast 和带宽开销，所以通常是合理的性能选择。
>
> 但如果 TP size 很大、AllReduce 累加项很多，或者模型对数值比较敏感，比如 logits/归约结果出现明显偏差，就可以切回 FP32 通信。另一种折中是 BF16：它也是 2 bytes，通信量和 FP16 一样，但指数位和 FP32 一样，动态范围更好，只是尾数精度更低。
>
> 我的项目里通过 `CommDataType` 支持 F32/F16/BF16。模型侧 `allReduceIfTP` 根据 `act_dtype` 选择通信 dtype：FP16 激活走 F16，BF16 激活走 BF16，FP32 激活走 F32；NCCL 后端再映射到 `ncclFloat16`、`ncclBfloat16`、`ncclFloat32`。

一句话记忆：

> FP16 AllReduce 用精度换带宽；推理默认可接受，数值敏感或 rank 多时切 FP32，BF16 是 2B 通信量和大动态范围的折中。

可能追问：

> NCCL 和 MPI 在你的通信抽象层里分别扮演什么角色？为什么需要多后端通信抽象？

### Q67：NCCL 和 MPI 在你的通信抽象层里分别扮演什么角色？为什么需要多后端通信抽象？

我的原回答：

> NCCL 辅助 GPU 之间的通信，用于 TP 的 AllReduce，MPI 负责 CPU 与 GPU 之间的通信，主要是 rank0 的分发工作。

评估：

- NCCL 的定位基本正确：用于 GPU 间高性能集合通信，尤其是 TP 的 AllReduce。
- MPI 的描述需要纠正：MPI 不是 CPU 和 GPU 之间通信，而是多进程/多节点之间的通信与协调机制。它可以用于 rank 初始化、广播配置、CPU buffer 的集合通信，也可以作为没有 NCCL 时的 fallback。
- 多后端通信抽象的价值是让模型层只依赖统一 `Comm` 接口，不关心底层是 NCCL、MPI 还是 Mock。

推荐回答：

> NCCL 主要负责 GPU 间的高性能集合通信。TP 推理里 Row Parallel 后需要对 GPU 上的 hidden states 做 AllReduce，NCCL 能直接在 GPU device buffer 上做 `ncclAllReduce`，支持 FP16/BF16/FP32，并利用 NVLink/PCIe/多机网络做优化，所以是 GPU TP 的主力后端。
>
> MPI 更偏通用的多进程通信和控制面。它可以负责 rank/world size 管理、rank0 广播配置或权重元信息、进程间同步，也可以对 CPU buffer 做 AllReduce/AllGather/ReduceScatter。在某些环境没有 NCCL，或者要做 CPU 侧数据分发时，MPI 可以作为 fallback 或辅助后端。严格说，MPI 不是“CPU 和 GPU 之间通信”，CPU/GPU 之间的数据搬运还是 cudaMemcpy/runtime；MPI 是 rank 之间通信。
>
> 项目里抽象出统一的 `Comm` 接口，包括 `allReduceSum`、`allGather`、`reduceScatter`、`broadcast`、`send/recv` 等，并用 `CommDataType` 表示 F32/F16/BF16。这样模型层只调用 `comm->allReduceSum(t->data(), count, dtype)`，不用关心底层是 NCCL、MPI 还是单卡 Mock。以后换通信后端、做测试或支持不同部署环境时，模型代码不用改。
>
> 简单说，NCCL 是 GPU 数据面的高性能通信后端，MPI 是更通用的进程间通信/控制面和 fallback，多后端抽象让 TP 模型层和底层通信实现解耦。

一句话记忆：

> NCCL 管 GPU device buffer 的高性能集合通信；MPI 管 rank 间通用通信和控制面；统一 Comm 抽象让模型 forward 不依赖具体通信库。

可能追问：

> rank、world size、tp_rank、tp_size 分别是什么意思？模型加载和推理时怎么用？

### Q68：rank、world size、tp_rank、tp_size 分别是什么意思？模型加载和推理时怎么用？

我的原回答：

> 不知道

评估：

- 这是分布式/TP 的基础概念。
- `rank/world_size` 是通信层概念，表示当前进程编号和总进程数。
- `tp_rank/tp_size` 是模型并行层概念，表示当前进程在 tensor parallel group 内的编号和组大小。
- 当前项目里可以简单理解为 `rank == tp_rank`，`world_size == tp_size`，但更复杂系统里可能还有 DP/PP，二者不一定相等。

推荐回答：

> `world_size` 是通信世界里总共有多少个进程或 GPU，`rank` 是当前进程在这个通信世界里的编号，比如 0 到 `world_size-1`。这是 NCCL/MPI 这类通信库使用的概念。
>
> `tp_size` 是 tensor parallel group 的大小，也就是一个模型被切到多少张卡上；`tp_rank` 是当前进程在这个 TP group 里的编号。模型加载权重和 forward 时会用 `tp_rank/tp_size` 决定当前 rank 应该加载哪一片权重、负责哪些 heads、保存哪些 KV cache。
>
> 在我的项目里，主要做的是单一 TP group，所以通常可以认为：
>
> ```text
> world_size == tp_size
> rank == tp_rank
> ```
>
> 但在更复杂的系统里，如果同时有 Data Parallel、Pipeline Parallel，`world_size` 可能是所有进程总数，而 `tp_size` 只是其中一个 TP 组的大小。比如总共有 8 张卡，可能是 `tp_size=2`、`dp_size=4`，这时 world size 是 8，但每个 TP group 只有 2 个 rank。
>
> 模型加载时，Column Parallel 权重根据 `tp_rank` 取输出维的第 `tp_rank` 片，Row Parallel 权重根据 `tp_rank` 取输入维的第 `tp_rank` 片。推理时，本 rank 只计算本地 shard；Row Parallel 后通过通信层在 TP group 内做 AllReduce。

一句话记忆：

> rank/world_size 是通信进程编号；tp_rank/tp_size 是模型切分编号。简单 TP 场景二者相同，复杂 DP/PP/TP 混合时不一定相同。

可能追问：

> 如果某个权重维度不能被 tp_size 整除，你的项目怎么处理？工业实现又会怎么处理？

### Q69：项目里的通信抽象层是怎么实现的？从模型里的 `allReduceIfTP` 调用，到 NCCL/MPI 后端真正执行通信，中间经过哪些层？

我的原回答：

> 我不太记得了，只记得 rank 0 通过 json 文件进行分发，其他 rank 通信忘记了。

评估：

- rank 0 分发这个记忆方向接近 NCCL 初始化，但需要纠正：项目里 NCCL 初始化时 rank 0 生成 `ncclUniqueId`，通过共享文件或 TCP 分发给其他 rank，不是每次通信数据都通过 json 文件。
- 真正 AllReduce 数据通信发生在 NCCL/MPI 后端。NCCL 路径直接对 GPU device buffer 调 `ncclAllReduce`；MPI 路径可能把 GPU 数据拷到 CPU，再用 `MPI_Allreduce`。
- 模型层只依赖统一 `Comm` 抽象，不直接依赖 NCCL/MPI API。

推荐回答：

> 项目里通信是分层实现的。模型层在 Row Parallel 后调用 `allReduceIfTP(tensor, count)`。这个函数会先判断 `tp_size > 1` 且 `comm` 存在，然后根据当前激活 dtype 选择通信类型：FP16 走 `CommDataType::F16`，BF16 走 `BF16`，FP32 走 `F32`，最后调用统一接口：
>
> ```text
> comm->allReduceSum(t->data(), count, comm_dtype)
> ```
>
> `Comm` 是通信抽象基类，底下可以有 Mock、NCCL、MPI 等后端。这样 Qwen2 forward 不关心底层通信库，只知道要对一段 tensor 做 AllReduce sum。
>
> 如果后端是 NCCL，初始化时每个 rank 会创建 NCCL communicator。rank 0 先调用 `ncclGetUniqueId` 生成 `ncclUniqueId`，然后通过共享文件或 TCP 分发给其他 rank；所有 rank 拿到同一个 unique id 后调用 `ncclCommInitRank` 加入同一个通信组。真正通信时，`allReduceSum` 会把 `CommDataType` 映射成 `ncclFloat16/ncclBfloat16/ncclFloat32`，然后直接对 GPU buffer 调 `ncclAllReduce`。
>
> 如果后端是 MPI，它使用 `MPI_Init` 初始化进程通信，rank/world size 从 `MPI_COMM_WORLD` 获得。FP32 可以直接 `MPI_Allreduce`；FP16/BF16 在标准 MPI 路径里通常要转成 FP32 做归约再转回。项目注释里也说明，如果传入的是 GPU 指针，MPI 后端会走 GPU 到 CPU、MPI 通信、CPU 到 GPU 的中转路径，所以它更多是兼容/fallback，不是高性能 GPU TP 首选。
>
> 单卡或测试时可以用 Mock 后端，AllReduce 基本是空操作。
>
> 所以完整链路是：
>
> ```text
> qwen2.cpp Row Parallel 后
> -> allReduceIfTP
> -> 根据 act_dtype 选择 CommDataType
> -> Comm 抽象接口 allReduceSum
> -> NCCL/MPI/Mock 后端
> -> ncclAllReduce 或 MPI_Allreduce
> ```

一句话记忆：

> rank0 分发的是 NCCL communicator 初始化需要的 unique id；真正 tensor 通信走 Comm 抽象后的 NCCL/MPI AllReduce，模型层不直接碰底层通信 API。

可能追问：

> NCCL 初始化时为什么需要 unique id？rank0 分发 unique id 和真正 AllReduce 数据通信有什么区别？

### Q70：NCCL 初始化时为什么需要 `ncclUniqueId`？它和 `rank/world_size` 分别起什么作用？

我的原回答：

> ncclUniqueId 用来让多个进程加入同一个 NCCL communicator。rank0 生成 ID，其他 rank 拿到同一个 ID。每个 rank 调 `ncclCommInitRank(comm, world_size, unique_id, rank)`。world_size 告诉 NCCL 组里有几个 rank，rank 告诉 NCCL 当前进程是第几个。初始化完成后，真正数据通信走 communicator，不再走 unique id 分发通道。

评估：

- 回答正确，已经抓住 NCCL 初始化的核心。
- 可以补充：`ncclUniqueId` 是初始化令牌，用来让独立进程 rendezvous 到同一个通信组；不是每次通信的数据通道。
- `rank` 必须在 `[0, world_size)` 内唯一，否则 communicator 初始化和 collective 匹配都会出错。

推荐回答：

> NCCL 是多进程通信库，每个 rank 是独立进程。为了让这些独立进程知道自己属于同一个通信组，需要一个共同的初始化令牌，这就是 `ncclUniqueId`。
>
> 在项目里，rank0 调 `ncclGetUniqueId` 生成这个 ID，然后通过 TCP 或单机共享文件分发给其他 rank。所有 rank 拿到同一个 ID 后，分别调用：
>
> ```cpp
> ncclCommInitRank(&_nccl_comm, world_size, nccl_id, rank);
> ```
>
> 其中 `world_size` 告诉 NCCL 这个 communicator 里总共有多少个 rank，`rank` 告诉 NCCL 当前进程是第几个 rank。每个 rank 的编号必须唯一，并且范围是 `[0, world_size)`。
>
> 初始化完成后，每个进程里都有一个 `_nccl_comm`。后续 AllReduce、Broadcast、AllGather 等真正的数据通信都走这个 communicator，而不再通过 unique id 的分发通道。也就是说，unique id 只用于初始化阶段让大家 rendezvous 到同一个组，真正的数据传输由 NCCL communicator 内部管理。

一句话记忆：

> `ncclUniqueId` 是把多个独立 rank 拉进同一个 NCCL 通信组的初始化令牌；`world_size` 是组大小，`rank` 是组内编号，真正通信发生在 communicator 建好之后。

可能追问：

> 如果某个 rank 没有进入同一次 `ncclAllReduce`，会发生什么？为什么 collective 调用顺序必须一致？
