#include "cuda_graph.hpp"

#ifdef ENABLE_NVIDIA_API
extern "C" {
int cudaStreamBeginCapture(void* stream, int mode);
int cudaStreamEndCapture(void* stream, void** pGraph);
int cudaGraphInstantiate(void** pExec, void* graph, void** errNode, char* errLog, size_t bufSize);
int cudaGraphLaunch(void* exec, void* stream);
int cudaGraphDestroy(void* graph);
int cudaGraphExecDestroy(void* exec);
const char* cudaGetErrorString(int error);
}
#endif

#include <stdexcept>
#include <cstdio>

namespace llaisys::core {

CUDAGraphRunner::~CUDAGraphRunner() {
#ifdef ENABLE_NVIDIA_API
    if (_graph_exec) cudaGraphExecDestroy(_graph_exec);
    if (_graph) cudaGraphDestroy(_graph);
#endif
    _graph = nullptr;
    _graph_exec = nullptr;
}

bool CUDAGraphRunner::launch(std::function<void()> fn) {
#ifdef ENABLE_NVIDIA_API
    if (_captured && _graph_exec) {
        int err = cudaGraphLaunch(_graph_exec, nullptr);
        if (err != 0) {
            fprintf(stderr, "[CUDAGraph] replay failed: %s\n",
                    cudaGetErrorString(err));
            invalidate();
            return launch(fn);
        }
        ++_replay_count;
        return true;
    }

    if (_graph_exec) {
        cudaGraphExecDestroy(_graph_exec);
        _graph_exec = nullptr;
    }
    if (_graph) {
        cudaGraphDestroy(_graph);
        _graph = nullptr;
    }

    void *graph = nullptr;
    int err = cudaStreamBeginCapture(nullptr, 0 /*cudaStreamCaptureModeGlobal*/);
    if (err != 0) {
        fprintf(stderr, "[CUDAGraph] beginCapture failed: %s, falling back to eager\n",
                cudaGetErrorString(err));
        fn();
        return false;
    }

    fn();

    err = cudaStreamEndCapture(nullptr, &graph);
    if (err != 0) {
        fprintf(stderr, "[CUDAGraph] endCapture failed: %s\n",
                cudaGetErrorString(err));
        fn();
        return false;
    }

    void *exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != 0) {
        fprintf(stderr, "[CUDAGraph] instantiate failed: %s\n",
                cudaGetErrorString(err));
        cudaGraphDestroy(graph);
        fn();
        return false;
    }

    _graph = graph;
    _graph_exec = exec;
    _captured = true;
    ++_capture_count;

    err = cudaGraphLaunch(exec, nullptr);
    if (err != 0) {
        fprintf(stderr, "[CUDAGraph] initial launch failed: %s\n",
                cudaGetErrorString(err));
        invalidate();
        fn();
        return false;
    }

    return false;

#else
    fn();
    return false;
#endif
}

void CUDAGraphRunner::invalidate() {
#ifdef ENABLE_NVIDIA_API
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

} // namespace llaisys::core
