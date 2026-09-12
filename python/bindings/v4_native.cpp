#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "src/models/deepseek_v4/session.hpp"
#include "src/backends/v4_reference.hpp"
#include <set>

namespace py = pybind11;
using namespace llaisys::models::deepseek_v4;
namespace {
template<class T> T integer(py::handle value) {
    if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) throw py::value_error("expected integer, not bool/float");
    return py::cast<T>(value);
}
void keys(const py::dict &d, std::set<std::string> allowed) {
    for (auto item : d) if (!allowed.count(py::cast<std::string>(item.first))) throw py::value_error("unknown native plan field");
}
llaisys::engine::SamplingParams sampling(const py::dict &d) {
    keys(d, {"temperature", "top_k", "top_p"});
    return {d.contains("temperature") ? py::cast<float>(d["temperature"]) : 0.0F,
        d.contains("top_k") ? integer<int>(d["top_k"]) : 1, d.contains("top_p") ? py::cast<float>(d["top_p"]) : 1.0F};
}
llaisys::engine::SchedulePlan plan(const py::dict &d) {
    keys(d, {"step_id", "reset_slots", "prefills", "decodes"}); llaisys::engine::SchedulePlan result;
    if (d.contains("step_id")) result.step_id = integer<uint64_t>(d["step_id"]);
    if (d.contains("reset_slots")) for (auto id : py::cast<py::list>(d["reset_slots"])) result.reset_slots.push_back(integer<size_t>(id));
    if (d.contains("prefills")) for (auto value : py::cast<py::list>(d["prefills"])) {
        auto item = py::cast<py::dict>(value); keys(item, {"request_id", "slot_id", "token_ids", "start_pos", "is_last_chunk", "sampling"});
        llaisys::engine::PrefillItem p; p.slot_id = integer<size_t>(item["slot_id"]);
        p.request_id = item.contains("request_id") ? integer<uint64_t>(item["request_id"]) : p.slot_id;
        p.start_pos = item.contains("start_pos") ? integer<int64_t>(item["start_pos"]) : 0;
        p.is_last_chunk = true;
        if (item.contains("is_last_chunk")) {
            if (!py::isinstance<py::bool_>(item["is_last_chunk"])) throw py::value_error("is_last_chunk must be bool");
            p.is_last_chunk = py::cast<bool>(item["is_last_chunk"]);
        }
        for (auto id : py::cast<py::list>(item["token_ids"])) p.token_ids.push_back(integer<int64_t>(id));
        p.sampling = sampling(item.contains("sampling") ? py::cast<py::dict>(item["sampling"]) : py::dict());
        result.prefills.push_back(std::move(p));
    }
    if (d.contains("decodes")) for (auto value : py::cast<py::list>(d["decodes"])) {
        auto item = py::cast<py::dict>(value); keys(item, {"request_id", "slot_id", "token_id", "sampling"});
        llaisys::engine::DecodeItem p; p.slot_id = integer<size_t>(item["slot_id"]); p.token_id = integer<int64_t>(item["token_id"]);
        p.request_id = item.contains("request_id") ? integer<uint64_t>(item["request_id"]) : p.slot_id;
        p.sampling = sampling(item.contains("sampling") ? py::cast<py::dict>(item["sampling"]) : py::dict()); result.decodes.push_back(p);
    }
    return result;
}
}
PYBIND11_MODULE(_v4_native, module) {
    module.doc() = "Explicit native V4 base-model batch execution; TileLang/ATen reference, not fused batch or paged storage.";
    py::class_<Session>(module, "Session")
        .def(py::init([](std::string source, std::string checkpoint, int64_t capacity, size_t slots,
                        const std::map<std::string, std::pair<std::string, std::string>> &entries,
                        std::string tilelang, std::string hadamard, bool chunks) {
            std::map<std::string, llaisys::backends::v4_reference::Bundle> bundles;
            for (const auto &[key, value] : entries) bundles.emplace(key, llaisys::backends::v4_reference::Bundle{value.first, value.second});
            py::gil_scoped_release release;
            return std::make_unique<Session>(source, checkpoint, capacity, slots,
                [bundles = std::move(bundles), tilelang = std::move(tilelang), hadamard = std::move(hadamard)](auto &runtime, const auto &config) {
                    return llaisys::backends::v4_reference::make(runtime, config, bundles, tilelang, hadamard);
                }, chunks);
        }), py::arg("source_model"), py::arg("checkpoint"), py::arg("capacity"), py::arg("slots"),
            py::arg("bundles"), py::arg("tilelang_version"), py::arg("hadamard_version"), py::arg("experimental_chunking") = false)
        .def("execute", [](Session &self, const py::dict &input, bool capture) {
            auto schedule = plan(input); SessionResult result;
            { py::gil_scoped_release release; result = self.execute(std::move(schedule), capture); }
            py::list rows;
            for (const auto &out : result.outputs) {
                py::dict row; row["request_id"] = out.sequence.request_id; row["slot_id"] = out.sequence.slot_id; row["token_id"] = out.sequence.token_id;
                row["logits"] = capture ? py::cast(out.logits) : py::none(); rows.append(row);
            }
            py::dict response; response["step_id"] = result.step_id; response["outputs"] = rows; return response;
        }, py::arg("plan"), py::arg("capture_logits") = false)
        .def("info", [](Session &self) {
            SessionInfo info; { py::gil_scoped_release release; info = self.info(); }
            py::dict result; result["layers"] = info.layers; result["vocabulary"] = info.vocabulary; result["capacity"] = info.capacity;
            result["loaded_weights"] = info.loaded_weights; result["auxiliary_weights"] = info.auxiliary_weights; result["loaded_bytes"] = info.loaded_bytes;
            result["active_slots"] = info.active_slots; result["steps"] = info.steps;
            result["slots"] = info.slots; result["experimental_chunking"] = info.experimental_chunking;
            result["model_calls"] = info.model_calls; result["model_failures"] = info.model_failures;
            py::list operators;
            for (const auto &op : info.operators) {
                py::dict row; row["backend"] = op.identity.backend; row["version"] = op.identity.version;
                row["operation"] = op.identity.operation; row["contract_revision"] = op.identity.contract_revision;
                row["calls"] = op.calls; row["failures"] = op.failures; row["fallback"] = false;
                operators.append(row);
            }
            result["operators"] = operators;
            result["execution"] = "native-cpp-serial-slots"; result["cache"] = "continuous-ring-reference";
            result["fallback"] = false; result["python_operator_callbacks"] = false; return result;
        })
        .def("close", &Session::close, py::call_guard<py::gil_scoped_release>())
        .def("__enter__", [](Session &self) -> Session & { return self; }, py::return_value_policy::reference_internal)
        .def("__exit__", [](Session &self, py::object, py::object, py::object) { py::gil_scoped_release release; self.close(); return false; });
}
