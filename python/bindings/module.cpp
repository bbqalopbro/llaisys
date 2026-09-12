#include "llaisys/models/qwen2.h"
#include "src/engine/schedule_plan.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace llaisys::engine;

void bindCache(py::module_ &module);
#ifdef LLAISYS_ENABLE_DLPACK_CACHE
void bindPagedStorage(py::module_ &module);
#endif

namespace {

class Qwen2BatchRuntime {
public:
    Qwen2BatchRuntime(uintptr_t model_handle, size_t max_batch_size,
                      size_t max_seq_per_slot)
        : max_batch_size_(max_batch_size) {
        if (model_handle == 0) throw std::invalid_argument("model handle is null");
        ctx_ = llaisysQwen2BatchContextCreate(
            reinterpret_cast<LlaisysQwen2Model *>(model_handle),
            max_batch_size, max_seq_per_slot);
        if (!ctx_) throw std::runtime_error("failed to create Qwen2 batch runtime");
    }

    ~Qwen2BatchRuntime() {
        if (ctx_) llaisysQwen2BatchContextDestroy(ctx_);
    }

    Qwen2BatchRuntime(const Qwen2BatchRuntime &) = delete;
    Qwen2BatchRuntime &operator=(const Qwen2BatchRuntime &) = delete;

    void reset(size_t slot_id) {
        validateSlot(slot_id);
        llaisysQwen2BatchSlotReset(ctx_, slot_id);
    }

    int64_t prefill(size_t slot_id, const std::vector<int64_t> &tokens,
                    const SamplingParams &sampling) {
        validateSlot(slot_id);
        if (tokens.empty()) throw std::invalid_argument("prefill tokens are empty");
        const int64_t result = llaisysQwen2BatchPrefill(
            ctx_, slot_id, const_cast<int64_t *>(tokens.data()), tokens.size(),
            sampling.temperature, sampling.top_k, sampling.top_p);
        if (result < 0) throw std::runtime_error("prefill failed");
        return result;
    }

    int64_t prefillChunk(size_t slot_id, const std::vector<int64_t> &tokens,
                         int64_t start_pos, bool is_last_chunk,
                         const SamplingParams &sampling) {
        validateSlot(slot_id);
        if (tokens.empty()) throw std::invalid_argument("prefill chunk is empty");
        const int64_t result = llaisysQwen2BatchPrefillChunk(
            ctx_, slot_id, tokens.data(), tokens.size(), start_pos,
            is_last_chunk ? 1 : 0, sampling.temperature, sampling.top_k,
            sampling.top_p);
        if (is_last_chunk && result < 0)
            throw std::runtime_error("incremental prefill failed");
        return result;
    }

    size_t prefixLookup(size_t slot_id, const std::vector<int64_t> &tokens) {
        validateSlot(slot_id);
        if (tokens.empty()) return 0;
        return llaisysQwen2BatchPrefixLookup(ctx_, slot_id, tokens.data(),
                                             tokens.size());
    }

    bool prefixPublish(size_t slot_id, const std::vector<int64_t> &tokens) {
        validateSlot(slot_id);
        if (tokens.empty()) return false;
        return llaisysQwen2BatchPrefixPublish(ctx_, slot_id, tokens.data(),
                                              tokens.size()) != 0;
    }

    std::vector<int64_t> decode(
        const std::vector<size_t> &slots,
        const std::vector<int64_t> &tokens,
        const std::vector<float> &temperatures,
        const std::vector<int> &top_ks,
        const std::vector<float> &top_ps) {
        const size_t count = slots.size();
        if (count == 0 || tokens.size() != count || temperatures.size() != count ||
            top_ks.size() != count || top_ps.size() != count)
            throw std::invalid_argument("decode arrays must have the same non-zero length");
        for (size_t slot_id : slots) validateSlot(slot_id);
        std::vector<int64_t> outputs(count, -1);
        llaisysQwen2BatchDecodePerRequest(
            ctx_, const_cast<size_t *>(slots.data()), count,
            const_cast<int64_t *>(tokens.data()),
            const_cast<float *>(temperatures.data()),
            const_cast<int *>(top_ks.data()), const_cast<float *>(top_ps.data()),
            outputs.data());
        return outputs;
    }

    uintptr_t saveSlot(size_t slot_id) const {
        validateSlot(slot_id);
        return reinterpret_cast<uintptr_t>(llaisysQwen2BatchSlotSave(ctx_, slot_id));
    }

    void restoreSlot(size_t slot_id, uintptr_t snapshot_handle) {
        validateSlot(slot_id);
        if (snapshot_handle == 0) throw std::invalid_argument("snapshot handle is null");
        llaisysQwen2BatchSlotRestore(
            ctx_, slot_id,
            reinterpret_cast<LlaisysQwen2CacheSnapshot *>(snapshot_handle));
    }

    StepResult execute(const SchedulePlan &plan) {
        StepResult result;
        result.step_id = plan.step_id;
        for (size_t slot_id : plan.reset_slots) reset(slot_id);
        for (const auto &item : plan.prefills) {
            const int64_t token = prefillChunk(
                item.slot_id, item.token_ids, item.start_pos,
                item.is_last_chunk, item.sampling);
            if (item.is_last_chunk)
                result.outputs.push_back({item.request_id, item.slot_id, token});
        }

        if (!plan.decodes.empty()) {
            const size_t count = plan.decodes.size();
            std::vector<size_t> slots(count);
            std::vector<int64_t> tokens(count), outputs(count, -1);
            std::vector<float> temperatures(count), top_ps(count);
            std::vector<int> top_ks(count);
            for (size_t i = 0; i < count; ++i) {
                const auto &item = plan.decodes[i];
                validateSlot(item.slot_id);
                slots[i] = item.slot_id;
                tokens[i] = item.token_id;
                temperatures[i] = item.sampling.temperature;
                top_ks[i] = item.sampling.top_k;
                top_ps[i] = item.sampling.top_p;
            }
            llaisysQwen2BatchDecodePerRequest(
                ctx_, slots.data(), count, tokens.data(), temperatures.data(),
                top_ks.data(), top_ps.data(), outputs.data());
            for (size_t i = 0; i < count; ++i) {
                result.outputs.push_back(
                    {plan.decodes[i].request_id, slots[i], outputs[i]});
            }
        }
        return result;
    }

    int64_t slotPosition(size_t slot_id) const {
        validateSlot(slot_id);
        return llaisysQwen2BatchSlotGetPos(ctx_, slot_id);
    }
    size_t freeBlocks() const { return llaisysQwen2BatchGetFreeBlocks(ctx_); }
    size_t totalBlocks() const { return llaisysQwen2BatchGetTotalBlocks(ctx_); }
    int blockSize() const { return llaisysQwen2BatchGetBlockSize(ctx_); }

private:
    void validateSlot(size_t slot_id) const {
        if (slot_id >= max_batch_size_) throw std::out_of_range("slot id out of range");
    }

    LlaisysQwen2BatchContext *ctx_ = nullptr;
    size_t max_batch_size_ = 0;
};

} // namespace

PYBIND11_MODULE(_C, module) {
    module.doc() = "Safe Python bindings for the llaisys scheduling/runtime boundary";
    bindCache(module);
#ifdef LLAISYS_ENABLE_DLPACK_CACHE
    bindPagedStorage(module);
    module.attr("v4_native_storage_available") = true;
#else
    module.attr("v4_native_storage_available") = false;
#endif

    py::class_<SamplingParams>(module, "SamplingParams")
        .def(py::init<>())
        .def_readwrite("temperature", &SamplingParams::temperature)
        .def_readwrite("top_k", &SamplingParams::top_k)
        .def_readwrite("top_p", &SamplingParams::top_p);
    py::class_<PrefillItem>(module, "PrefillItem")
        .def(py::init<>())
        .def_readwrite("request_id", &PrefillItem::request_id)
        .def_readwrite("slot_id", &PrefillItem::slot_id)
        .def_readwrite("token_ids", &PrefillItem::token_ids)
        .def_readwrite("start_pos", &PrefillItem::start_pos)
        .def_readwrite("is_last_chunk", &PrefillItem::is_last_chunk)
        .def_readwrite("sampling", &PrefillItem::sampling);
    py::class_<DecodeItem>(module, "DecodeItem")
        .def(py::init<>())
        .def_readwrite("request_id", &DecodeItem::request_id)
        .def_readwrite("slot_id", &DecodeItem::slot_id)
        .def_readwrite("token_id", &DecodeItem::token_id)
        .def_readwrite("sampling", &DecodeItem::sampling);
    py::class_<SchedulePlan>(module, "SchedulePlan")
        .def(py::init<>())
        .def_readwrite("step_id", &SchedulePlan::step_id)
        .def_readwrite("reset_slots", &SchedulePlan::reset_slots)
        .def_readwrite("prefills", &SchedulePlan::prefills)
        .def_readwrite("decodes", &SchedulePlan::decodes);
    py::class_<SequenceOutput>(module, "SequenceOutput")
        .def_readonly("request_id", &SequenceOutput::request_id)
        .def_readonly("slot_id", &SequenceOutput::slot_id)
        .def_readonly("token_id", &SequenceOutput::token_id);
    py::class_<StepResult>(module, "StepResult")
        .def_readonly("step_id", &StepResult::step_id)
        .def_readonly("outputs", &StepResult::outputs);
    py::class_<Qwen2BatchRuntime>(module, "Qwen2BatchRuntime")
        .def(py::init<uintptr_t, size_t, size_t>())
        .def("reset", &Qwen2BatchRuntime::reset, py::call_guard<py::gil_scoped_release>())
        .def("prefill", &Qwen2BatchRuntime::prefill,
             py::call_guard<py::gil_scoped_release>())
        .def("prefill_chunk", &Qwen2BatchRuntime::prefillChunk,
             py::call_guard<py::gil_scoped_release>())
        .def("prefix_lookup", &Qwen2BatchRuntime::prefixLookup,
             py::call_guard<py::gil_scoped_release>())
        .def("prefix_publish", &Qwen2BatchRuntime::prefixPublish,
             py::call_guard<py::gil_scoped_release>())
        .def("decode_per_request", &Qwen2BatchRuntime::decode,
             py::call_guard<py::gil_scoped_release>())
        .def("execute", &Qwen2BatchRuntime::execute,
             py::call_guard<py::gil_scoped_release>())
        .def("slot_save", &Qwen2BatchRuntime::saveSlot,
             py::call_guard<py::gil_scoped_release>())
        .def("slot_restore", &Qwen2BatchRuntime::restoreSlot,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("free_blocks", &Qwen2BatchRuntime::freeBlocks)
        .def_property_readonly("total_blocks", &Qwen2BatchRuntime::totalBlocks)
        .def_property_readonly("block_size", &Qwen2BatchRuntime::blockSize)
        .def("slot_position", &Qwen2BatchRuntime::slotPosition);
}
