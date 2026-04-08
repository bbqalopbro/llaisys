#pragma once

#include "llaisys.h"
#include <cstddef>
#include <functional>
#include <vector>
#include <memory>

namespace llaisys::core {

/**
 * CUDAGraphRunner — captures a CUDA graph from a recorded lambda and replays it.
 *
 * Workflow:
 *   1. Create runner: CUDAGraphRunner runner;
 *   2. First call: runner.launch(fn) → captures the graph by executing fn
 *   3. Subsequent calls: runner.launch(fn) → replays the captured graph
 *      (fn is NOT re-invoked; the graph instance is replayed)
 *
 * Invalidation:
 *   - Call runner.invalidate() when graph topology changes (e.g., batch size change)
 *   - Next launch() will re-capture
 *
 * For pointer-stable arguments (e.g., block pool pointers that don't change),
 * the graph can be replayed as-is. For dynamic pointers, either:
 *   a) Use cudaGraphExecKernelNodeSetParams to update kernel arguments, or
 *   b) Invalidate and re-capture (simple approach, ~1ms overhead)
 *
 * This implementation uses approach (b) for simplicity.
 */
class CUDAGraphRunner {
public:
    CUDAGraphRunner() = default;
    ~CUDAGraphRunner();

    CUDAGraphRunner(const CUDAGraphRunner &) = delete;
    CUDAGraphRunner &operator=(const CUDAGraphRunner &) = delete;

    // Launch: captures on first call, replays thereafter.
    // fn: a callable that enqueues CUDA kernels on the current stream.
    // Returns true if the graph was replayed (false if captured fresh).
    bool launch(std::function<void()> fn);

    // Mark graph as invalid; next launch() will re-capture.
    void invalidate();

    // Whether a valid graph is currently captured.
    bool is_captured() const { return _captured; }

    // Get capture/replay counts for profiling.
    size_t capture_count() const { return _capture_count; }
    size_t replay_count() const { return _replay_count; }

private:
    void *_graph = nullptr;        // cudaGraph_t
    void *_graph_exec = nullptr;   // cudaGraphExec_t
    bool _captured = false;
    size_t _capture_count = 0;
    size_t _replay_count = 0;
};

/**
 * CUDAGraphDecodeSession — manages CUDA graph for the decode loop.
 *
 * The decode loop has stable topology when:
 *   - Batch size is unchanged
 *   - Number of active layers is the same
 *   - Block pool base pointers haven't changed
 *
 * When any of these change, call invalidate() to re-capture.
 */
class CUDAGraphDecodeSession {
public:
    CUDAGraphDecodeSession() = default;

    void set_batch_size(size_t bs) {
        if (bs != _batch_size) {
            _batch_size = bs;
            _runner.invalidate();
        }
    }

    CUDAGraphRunner &runner() { return _runner; }

    void invalidate() { _runner.invalidate(); }

    size_t batch_size() const { return _batch_size; }

private:
    CUDAGraphRunner _runner;
    size_t _batch_size = 0;
};

} // namespace llaisys::core
