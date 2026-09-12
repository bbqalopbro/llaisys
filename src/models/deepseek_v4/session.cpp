#include "session.hpp"
#include "src/core/context/context.hpp"
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <cmath>

namespace llaisys::models::deepseek_v4 {
namespace {
using Tensor = backends::native::Tensor;
void check(bool ok, const char *msg) { if (!ok) throw std::invalid_argument(msg); }
struct Worker {
    core::Runtime &runtime;
    ModelConfig config;
    Checkpoint checkpoint;
    ModelBackends backends;
    Model model;
    std::vector<std::unique_ptr<ModelState>> slots;
    bool experimental_chunking;
    uint64_t steps = 0;
    Worker(core::Runtime &r, const std::string &source, const std::string &path, int64_t cap,
           size_t count, const Session::BackendFactory &factory, bool chunks)
        : runtime(r), config(ModelConfig::fromDirectory(source, cap)), checkpoint(path), backends(factory(r, config)),
          model(r, config, checkpoint, backends), slots(count), experimental_chunking(chunks) {}
    void slot(size_t id) const { check(id < slots.size(), "native session slot out of range"); }
    void sampling(const engine::SamplingParams &p) const {
        check(std::isfinite(p.temperature) && p.temperature >= 0 && p.top_k >= 0 && std::isfinite(p.top_p)
              && p.top_p > 0 && p.top_p <= 1 && (p.temperature == 0 || p.top_k == 1), "native base session currently requires explicit greedy sampling");
    }
    void token(int64_t id) const { check(id >= 0 && id < config.vocabulary, "native token outside vocabulary"); }
    SessionResult execute(const engine::SchedulePlan &plan, bool capture) {
        std::set<size_t> resets, touched;
        for (auto id : plan.reset_slots) { slot(id); check(resets.insert(id).second, "duplicate reset slot"); }
        // Validate the entire supplied plan before changing any request state.
        for (const auto &item : plan.prefills) {
            slot(item.slot_id); sampling(item.sampling);
            check(touched.insert(item.slot_id).second && !item.token_ids.empty(), "duplicate slot or empty prefill");
            for (auto id : item.token_ids) token(id);
            auto &state = slots[item.slot_id]; int64_t position = resets.count(item.slot_id) || !state ? 0 : state->position();
            check(item.start_pos == position && position < config.capacity && item.token_ids.size() <= static_cast<uint64_t>(config.capacity - position), "prefill position/capacity mismatch");
            check(experimental_chunking || (position == 0 && item.is_last_chunk), "original full/chunk numerical gate requires explicit experimental chunking opt-in");
        }
        for (const auto &item : plan.decodes) {
            slot(item.slot_id); sampling(item.sampling); token(item.token_id);
            check(touched.insert(item.slot_id).second && !resets.count(item.slot_id) && slots[item.slot_id]
                  && slots[item.slot_id]->position() > 0 && slots[item.slot_id]->position() < config.capacity, "invalid/duplicate decode slot or capacity");
        }
        for (auto id : plan.reset_slots) slots[id].reset();
        SessionResult result{plan.step_id, {}};
        auto forward = [&](uint64_t request, size_t id, const std::vector<int64_t> &tokens, bool emit) {
            auto &state = slots[id];
            if (!state) { state = std::make_unique<ModelState>(runtime, config); model.reset(*state); }
            Tensor ids(runtime, {1, static_cast<int64_t>(tokens.size())}, {kDLInt, 64, 1}); ids.upload(tokens.data(), ids.bytes());
            ModelWorkspace workspace(runtime, config, state->position(), tokens.size(), emit);
            auto output = model.forward(ids, *state, workspace);
            if (emit) {
                SessionOutput row{{request, id, -1}, {}};
                output.next_ids->download(&row.sequence.token_id, sizeof(int64_t));
                if (capture) {
                    row.logits.resize(config.vocabulary); output.logits->download(row.logits.data(), output.logits->bytes());
                    for (auto value : row.logits) check(std::isfinite(value), "non-finite native logits");
                }
                result.outputs.push_back(std::move(row));
            }
            runtime.synchronize(); // Workspace destruction stays after queued work.
        };
        try {
            for (const auto &item : plan.prefills) forward(item.request_id, item.slot_id, item.token_ids, item.is_last_chunk);
            for (const auto &item : plan.decodes) forward(item.request_id, item.slot_id, {item.token_id}, true);
            ++steps; return result;
        } catch (...) {
            // The caller gets no partial result. Drop every touched request;
            // never let an earlier successful slot silently advance on retry.
            for (auto id : touched) slots[id].reset();
            throw;
        }
    }
};
}

struct Session::Impl {
    std::mutex mutex, join_mutex;
    std::condition_variable ready;
    std::deque<std::function<void(Worker &)>> commands;
    bool stopping = false;
    std::thread thread;
    Impl(std::string source, std::string path, int64_t cap, size_t count, BackendFactory factory, bool chunks) {
        check(count > 0 && count <= 1024 && static_cast<bool>(factory), "invalid native session slots/backend factory");
        std::promise<void> started; auto startup = started.get_future();
        thread = std::thread([this, source = std::move(source), path = std::move(path), cap, count, chunks,
                              factory = std::move(factory), started = std::move(started)]() mutable {
            std::unique_ptr<Worker> worker;
            try {
                core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
                worker = std::make_unique<Worker>(core::context().runtime(), source, path, cap, count, factory, chunks);
                started.set_value();
            } catch (...) { started.set_exception(std::current_exception()); return; }
            for (;;) {
                std::function<void(Worker &)> call;
                { std::unique_lock<std::mutex> lock(mutex); ready.wait(lock, [&] { return stopping || !commands.empty(); });
                  if (commands.empty()) break;
                  call = std::move(commands.front()); commands.pop_front(); }
                call(*worker); // packaged_task stores exceptions in the future.
            }
            // All model/state/kernels die on their owning thread, before its
            // thread-local Context. No Python/GIL is needed during teardown.
            worker.reset();
        });
        try { startup.get(); } catch (...) { close(); throw; }
    }
    template<class F> auto run(F f) -> decltype(f(std::declval<Worker &>())) {
        using R = decltype(f(std::declval<Worker &>()));
        auto task = std::make_shared<std::packaged_task<R(Worker &)>>(std::move(f)); auto result = task->get_future();
        { std::lock_guard<std::mutex> lock(mutex); check(!stopping, "native session is closed");
          commands.push_back([task](Worker &worker) { (*task)(worker); }); }
        ready.notify_one(); return result.get();
    }
    void close() {
        std::lock_guard<std::mutex> join_lock(join_mutex);
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_one(); if (thread.joinable()) thread.join();
    }
    ~Impl() { close(); }
};
Session::Session(std::string source, std::string checkpoint, int64_t capacity, size_t slots, BackendFactory factory, bool chunks)
    : impl_(std::make_unique<Impl>(std::move(source), std::move(checkpoint), capacity, slots, std::move(factory), chunks)) {}
Session::~Session() = default;
void Session::close() { impl_->close(); }
SessionResult Session::execute(engine::SchedulePlan plan, bool capture) {
    return impl_->run([plan = std::move(plan), capture](Worker &worker) { return worker.execute(plan, capture); });
}
SessionInfo Session::info() {
    return impl_->run([](Worker &w) {
        size_t active = 0; for (const auto &s : w.slots) if (s) ++active;
        SessionInfo info{w.config.layers, w.config.vocabulary, w.config.capacity, w.checkpoint.loadedCount(),
            w.checkpoint.auxiliaryCount(), active, w.checkpoint.loadedBytes(), w.steps, w.slots.size(),
            w.experimental_chunking, w.model.calls(), w.model.failures(), {}};
        // One Kernel may be bound to many layers/aliases. Count actual objects
        // once, then aggregate compatible identities, not binding references.
        std::set<const backends::native::Kernel *> seen;
        std::map<std::tuple<std::string, std::string, std::string, uint32_t>, size_t> groups;
        auto collect = [&](const KernelBindings &bindings) {
            for (const auto &[name, kernel] : bindings) {
                (void)name;
                if (!seen.insert(kernel.get()).second) continue;
                const auto &id = kernel->identity();
                auto [it, inserted] = groups.emplace(std::make_tuple(id.backend, id.version, id.operation, id.contract_revision), info.operators.size());
                if (inserted) info.operators.push_back({id, 0, 0});
                auto &row = info.operators[it->second]; row.calls += kernel->calls(); row.failures += kernel->failures();
            }
        };
        collect(w.backends.global); for (const auto &layer : w.backends.layers) collect(layer);
        return info;
    });
}
} // namespace llaisys::models::deepseek_v4
