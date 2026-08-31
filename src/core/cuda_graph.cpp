#include "cuda_graph.hpp"

#ifdef ENABLE_NVIDIA_API
// 这里手动声明少量 CUDA Runtime API，避免本文件直接依赖 cuda_runtime.h。
// 这些函数只在 ENABLE_NVIDIA_API 打开时使用；非 NVIDIA 构建会直接走 eager 路径。
extern "C" {
int cudaStreamBeginCapture(void* stream, int mode);
int cudaStreamEndCapture(void* stream, void** pGraph);
int cudaGraphInstantiate(void** pExec, void* graph, void** errNode, char* errLog, size_t bufSize);
int cudaGraphLaunch(void* exec, void* stream);
int cudaGraphGetNodes(void* graph, void** nodes, size_t* numNodes);
int cudaGraphDestroy(void* graph);
int cudaGraphExecDestroy(void* exec);
const char* cudaGetErrorString(int error);
int cudaDeviceSynchronize();
}

// cudaStreamPerThread：每个线程自己的默认 stream。
// CUDA Graph 不能安全捕获 legacy NULL stream 上的所有行为，所以这里固定使用
// per-thread default stream 来 capture/replay。
static void* const kPerThreadStream = (void*)2;
#endif

#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace llaisys::core {

#ifdef ENABLE_NVIDIA_API
static bool env_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] == '1';
}

static bool env_flag_disabled(const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] == '0';
}
#endif

CUDAGraphRunner::~CUDAGraphRunner() {
#ifdef ENABLE_NVIDIA_API
    if (env_flag_enabled("LLAISYS_CUDAGRAPH_STATS")) {
        print_stats();
    }
#endif
#ifdef ENABLE_NVIDIA_API
    // CUDAGraphRunner 持有两个 CUDA 资源：
    //   _graph      = cudaGraph_t，捕获到的图描述
    //   _graph_exec = cudaGraphExec_t，实例化后可直接 launch 的执行体
    // 析构时需要分别释放。
    if (_graph_exec) cudaGraphExecDestroy(_graph_exec);
    if (_graph) cudaGraphDestroy(_graph);
#endif
    _graph = nullptr;
    _graph_exec = nullptr;
}

bool CUDAGraphRunner::launch(std::function<void()> fn) {
#ifdef ENABLE_NVIDIA_API
    // 调试开关：强制不使用 CUDA Graph，直接执行传入的函数。
    // 这样可以快速判断问题来自 graph capture/replay，还是来自算子本身。
    if (env_flag_enabled("LLAISYS_ENFORCE_EAGER")) {
        fn();
        ++_eager_count;
        if (env_flag_enabled("LLAISYS_CUDAGRAPH_VERBOSE")) {
            fprintf(stderr, "[CUDAGraph] eager forced by LLAISYS_ENFORCE_EAGER\n");
        }
        return false;
    }

    // 首次调用先 eager 跑一遍，用来初始化 cuBLAS handle、workspace、算子内部
    // lazy buffer 等。注意这里必须直接返回，不能同一次 launch 继续 capture，
    // 否则同一个 decode token 会被执行两遍并重复写 KV cache。
    if (!_warmup_done && !_captured &&
        !env_flag_disabled("LLAISYS_CUDAGRAPH_WARMUP")) {
        fn();
        ++_eager_count;
        ++_warmup_count;
        _warmup_done = true;
        if (env_flag_enabled("LLAISYS_CUDAGRAPH_VERBOSE")) {
            fprintf(stderr, "[CUDAGraph] eager warmup done, capture starts on next launch\n");
        }
        return false;
    }

    // 如果已经捕获并实例化过 graph，后续 decode 直接 replay。
    // 注意：这里不会再次调用 fn()；graph 中保存的是第一次 fn() 执行时
    // 提交到 stream 上的 CUDA kernel 序列。
    if (_captured && _graph_exec) {
        int err = cudaGraphLaunch(_graph_exec, kPerThreadStream);
        if (err != 0) {
            ++_replay_fail_count;
            fprintf(stderr, "[CUDAGraph] replay failed: %s\n",
                    cudaGetErrorString(err));
            // replay 失败时丢弃旧 graph，递归调用 launch(fn) 重新捕获一次。
            invalidate();
            return launch(fn);
        }
        ++_replay_count;
        return true;
    }

    // 走到这里说明还没有可用 graph，或 graph 已被 invalidate。
    // 为避免残留资源，先清理旧的 graph/exec。
    if (_graph_exec) {
        cudaGraphExecDestroy(_graph_exec);
        _graph_exec = nullptr;
    }
    if (_graph) {
        cudaGraphDestroy(_graph);
        _graph = nullptr;
    }

    // 第一次执行：开始捕获当前 stream。
    // capture 开始后，fn() 内部提交的 CUDA kernel/memcpy 等操作会被记录进 graph。
    void *graph = nullptr;
    int err = cudaStreamBeginCapture(kPerThreadStream, 0 /*cudaStreamCaptureModeGlobal*/);
    if (err != 0) {
        ++_begin_capture_fail_count;
        fprintf(stderr, "[CUDAGraph] beginCapture failed: %s, falling back to eager\n",
                cudaGetErrorString(err));
        fn();
        ++_eager_count;
        return false;
    }

    // 执行用户传入的计算逻辑。这里不会立即保存 C++ 函数本身，
    // 保存的是函数执行过程中入队到 CUDA stream 的操作序列。
    try {
        fn();
    } catch (...) {
        // 如果 fn() 在 capture 中抛异常，必须尽力结束 capture 并清理残留 graph。
        // 否则当前 stream 可能保持在 capture 状态，后续 CUDA 调用会继续失败。
        void *aborted_graph = nullptr;
        int end_err = cudaStreamEndCapture(kPerThreadStream, &aborted_graph);
        if (aborted_graph) {
            cudaGraphDestroy(aborted_graph);
        }
        ++_end_capture_fail_count;
        fprintf(stderr, "[CUDAGraph] fn threw during capture; endCapture cleanup %s: %s\n",
                end_err == 0 ? "succeeded" : "failed",
                cudaGetErrorString(end_err));
        invalidate();
        throw;
    }

    // 结束 capture，得到 cudaGraph_t。
    err = cudaStreamEndCapture(kPerThreadStream, &graph);
    if (err != 0) {
        ++_end_capture_fail_count;
        fprintf(stderr, "[CUDAGraph] endCapture failed: %s\n",
                cudaGetErrorString(err));
        // capture 失败时回退到 eager 执行，保证功能仍然可用。
        fn();
        ++_eager_count;
        return false;
    }

    // 将捕获到的 graph 实例化成 cudaGraphExec_t。
    // graph 是“图描述”，exec 是之后可以反复 cudaGraphLaunch 的可执行对象。
    void *exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != 0) {
        ++_instantiate_fail_count;
        fprintf(stderr, "[CUDAGraph] instantiate failed: %s\n",
                cudaGetErrorString(err));
        cudaGraphDestroy(graph);
        fn();
        ++_eager_count;
        return false;
    }

    // 保存 graph 和 exec，后续 launch() 会直接 replay _graph_exec。
    _graph = graph;
    _graph_exec = exec;
    _captured = true;
    size_t node_count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &node_count) == 0) {
        _node_count = node_count;
    }
    ++_capture_count;

    // 第一次 capture 结束后，也用 graph 执行一次，而不是依赖 capture 期间的 fn()
    // 已经产生最终结果。这样首轮和后续 replay 的执行路径保持一致。
    err = cudaGraphLaunch(exec, kPerThreadStream);
    if (err != 0) {
        ++_initial_launch_fail_count;
        fprintf(stderr, "[CUDAGraph] initial launch failed: %s\n",
                cudaGetErrorString(err));
        // 首次 graph launch 失败则清掉 graph，回退到 eager。
        invalidate();
        fn();
        ++_eager_count;
        return false;
    }

    if (env_flag_enabled("LLAISYS_CUDAGRAPH_VERBOSE")) {
        fprintf(stderr, "[CUDAGraph] captured graph #%zu\n", _capture_count);
    }
    return false;

#else
    // 非 NVIDIA 构建没有 CUDA Graph，直接执行原函数。
    fn();
    return false;
#endif
}

void CUDAGraphRunner::invalidate() {
#ifdef ENABLE_NVIDIA_API
    // 当 graph 拓扑或关键指针不再稳定时调用 invalidate。
    // 下一次 launch(fn) 会重新 capture。
    if (_graph_exec) {
        cudaGraphExecDestroy(_graph_exec);
        _graph_exec = nullptr;
    }
    if (_graph) {
        cudaGraphDestroy(_graph);
        _graph = nullptr;
    }
#endif
    _captured = false;
}

void CUDAGraphRunner::print_stats() const {
    fprintf(stderr,
            "[CUDAGraph] stats: captured=%zu replayed=%zu eager=%zu warmup=%zu "
            "begin_fail=%zu end_fail=%zu instantiate_fail=%zu replay_fail=%zu "
            "initial_launch_fail=%zu nodes=%zu captured_now=%d\n",
            _capture_count, _replay_count, _eager_count, _warmup_count,
            _begin_capture_fail_count, _end_capture_fail_count,
            _instantiate_fail_count, _replay_fail_count,
            _initial_launch_fail_count, _node_count, _captured ? 1 : 0);
}

} // namespace llaisys::core
