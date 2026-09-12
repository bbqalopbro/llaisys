// Independent native process: no Python interpreter, Torch, pybind or callbacks.
#include "src/backends/tilelang/native_kernel.hpp"
#include "src/core/context/context.hpp"
#include "src/models/deepseek_v4/linear.hpp"
#include "src/backends/native/paged_storage.hpp"
#include "src/models/deepseek_v4/cache_layout.hpp"
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using llaisys::backends::tilelang::NativeKernel;
using llaisys::backends::tilelang::NativeTensor;

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static void requireNativeProcess() {
    require(dlsym(RTLD_DEFAULT, "Py_IsInitialized") == nullptr, "unexpected Python interpreter dependency");
    std::ifstream maps("/proc/self/maps");
    require(maps.good(), "cannot inspect native process dependencies");
    std::string mapping;
    while (std::getline(maps, mapping)) {
        require(mapping.find("libpython") == std::string::npos && mapping.find("/torch/lib/") == std::string::npos,
                "unexpected Python/Torch runtime mapping");
    }
}

static std::vector<unsigned char> readBytes(const std::string &path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file.good() && file.tellg() == static_cast<std::streamoff>(count), "invalid raw tensor file size");
    file.seekg(0);
    std::vector<unsigned char> bytes(count);
    file.read(reinterpret_cast<char *>(bytes.data()), count);
    require(file.good(), "cannot read raw tensor");
    return bytes;
}

static std::pair<std::shared_ptr<NativeTensor>, std::vector<unsigned char>>
readTensor(std::istream &manifest, llaisys::core::Runtime &runtime, int paged_ratio = -1) {
    unsigned code, bits, lanes;
    size_t rank, bytes;
    manifest >> code >> bits >> lanes >> rank;
    require(manifest.good() && rank > 0 && rank <= 16 && code <= 255 && bits <= 255 && lanes <= 65535,
            "invalid tensor header");
    std::vector<int64_t> shape(rank);
    for (auto &dim : shape) manifest >> dim;
    std::string input, output;
    manifest >> bytes >> std::quoted(input) >> std::quoted(output);
    require(manifest.good() && bytes < (1ull << 31), "invalid test tensor metadata");
    DLDataType dtype{static_cast<uint8_t>(code), static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)};
    std::shared_ptr<NativeTensor> value;
    if (paged_ratio >= 0) {
        using llaisys::backends::native::PagedStorage;
        auto layout = std::make_shared<llaisys::models::deepseek_v4::InterleavedLayerLayout>(128, 512, 128, paged_ratio);
        require(shape.size() == 3 && shape[0] == 1 && shape[2] == 512 && shape[1] > 0
                && static_cast<size_t>(shape[1]) % layout->slots(0) == 0 && code == kDLBfloat && bits == 16 && lanes == 1,
                "invalid paged latent fixture");
        auto pool = PagedStorage::create(runtime, shape[1] / layout->slots(0), layout);
        value = pool->componentTensor(0, shape, dtype);
        require(pool->activeExports() == 1, "latent fixture has no retaining export");
        // No pool owner survives this helper except the actual kernel argument.
        // Setup uploads directly to PagedCacheStorage, not via a Torch tensor.
    } else {
        value = std::make_shared<NativeTensor>(runtime, shape, dtype);
    }
    require(value->bytes() == bytes, "packed dtype storage byte mismatch");
    auto host = readBytes(input, bytes);
    value->upload(host.data(), host.size());
    return {std::move(value), readBytes(output, bytes)};
}

static void checkExact(NativeTensor &tensor, const std::vector<unsigned char> &expected, const std::string &name) {
    std::vector<unsigned char> actual(tensor.bytes());
    tensor.download(actual.data(), actual.size());
    require(actual == expected, ("non-exact native output/input mutation in " + name).c_str());
}

// An injected user backend tests the real C++ dispatch boundary: its error must
// propagate without replacing it by the default TileLang implementation.
class FailingBackend final : public llaisys::backends::native::Kernel {
public:
    explicit FailingBackend(llaisys::core::Runtime &runtime, uint32_t revision = 1)
        : Kernel(runtime, {"test-user-library", "failure-injection-v1", "act_quant", revision}) {}
    void call(const std::vector<NativeTensor *> &) override { ++count_; throw std::runtime_error("injected backend failure"); }
    void close() override {}
    uint64_t calls() const override { return count_; }
    uint64_t failures() const override { return count_; }
private:
    uint64_t count_ = 0;
};

int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: tilelang-native-standalone manifest.txt");
        require(std::getenv("SLURM_JOB_ID") != nullptr, "GPU standalone test requires Slurm");
        requireNativeProcess();
        llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        auto &runtime = llaisys::core::context().runtime();
        require(runtime.stream() != nullptr, "test requires the native non-default CUDA stream");
        std::ifstream manifest(argv[1]);
        std::string magic, tilelang_version;
        size_t cases;
        manifest >> magic >> cases >> tilelang_version;
        require(magic == "LLAISYS_TILELANG_NATIVE_TEST_V2" && cases > 0 && cases < 256, "invalid test manifest");
        size_t checked = 0, negative = 0, paged_cases = 0;
        for (size_t index = 0; index < cases; ++index) {
            std::string name, library;
            size_t count;
            manifest >> name >> std::quoted(library) >> count;
            require(manifest.good() && count > 0 && count < 32, "invalid kernel test entry");
            int paged_ratio = -1;
            for (int ratio : {0, 4, 128}) {
                if (name.rfind("sparse_attention_paged_r" + std::to_string(ratio) + "_", 0) == 0) paged_ratio = ratio;
            }
            require(llaisys::backends::native::PagedStorage::liveBytes() == 0, "previous paged fixture leaked storage");
            NativeKernel kernel(runtime, library, {"tilelang", tilelang_version, name, 1});
            std::vector<std::shared_ptr<NativeTensor>> owned;
            std::vector<NativeTensor *> arguments;
            std::vector<std::vector<unsigned char>> expected;
            for (size_t tensor = 0; tensor < count; ++tensor) {
                auto [value, reference] = readTensor(manifest, runtime, tensor == 1 ? paged_ratio : -1);
                expected.push_back(std::move(reference));
                arguments.push_back(value.get());
                owned.push_back(std::move(value));
            }
            const auto original_stream = TVMFFIEnvGetStream(kDLCUDA, runtime.deviceId());
            kernel.call(arguments);
            if (paged_ratio >= 0) {
                require(llaisys::backends::native::PagedStorage::liveBytes() >= owned[1]->bytes(),
                        "sparse attention did not retain the authoritative paged pool");
                ++paged_cases;
            }
            require(TVMFFIEnvGetStream(kDLCUDA, runtime.deviceId()) == original_stream, "FFI stream was not restored");
            for (size_t tensor = 0; tensor < count; ++tensor) {
                checkExact(*owned[tensor], expected[tensor], name);
                ++checked;
            }
            // Rejections happen on host before an invalid kernel can launch.
            bool rejected = false;
            try { kernel.call({}); } catch (const std::exception &) { rejected = true; }
            require(rejected && kernel.failures() == 1, "wrong argument count was not reported");
            ++negative;
            require(TVMFFIEnvGetStream(kDLCUDA, runtime.deviceId()) == original_stream, "error path changed FFI stream");
            rejected = false;
            std::thread foreign([&] {
                try { kernel.call(arguments); } catch (const std::runtime_error &) { rejected = true; }
            });
            foreign.join();
            require(rejected, "foreign thread invocation accepted");
            ++negative;
            kernel.close();
            rejected = false;
            try { kernel.call(arguments); } catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "closed kernel invocation accepted");
            ++negative;
            std::cout << "case " << index << " " << name << " exact\n";
        }
        size_t linear_cases;
        manifest >> magic >> linear_cases;
        require(manifest.good() && magic == "LINEARS" && linear_cases > 0 && linear_cases < 64, "invalid Linear test section");
        for (size_t index = 0; index < linear_cases; ++index) {
            std::string name, quant_library, gemm_library;
            manifest >> name >> std::quoted(quant_library) >> std::quoted(gemm_library);
            std::vector<std::shared_ptr<NativeTensor>> owned;
            std::vector<std::vector<unsigned char>> expected;
            for (size_t i = 0; i < 6; ++i) {
                auto [tensor, reference] = readTensor(manifest, runtime);
                owned.push_back(std::move(tensor));
                expected.push_back(std::move(reference));
            }
            const bool fp4 = owned[1]->dtype().code == kDLFloat4_e2m1fn;
            // Execute with a real nonzero-offset contiguous view. Guard bytes
            // before it must remain unchanged and are not part of the GEMM.
            const auto input_shape = owned[0]->shape();
            auto input_owner = std::make_shared<NativeTensor>(runtime,
                std::vector<int64_t>{input_shape[0] + 1, input_shape[1]}, DLDataType{kDLBfloat, 16, 1});
            std::vector<unsigned char> guarded_input(input_owner->bytes(), 0x5a);
            std::copy(expected[0].begin(), expected[0].end(), guarded_input.begin() + input_shape[1] * 2);
            input_owner->upload(guarded_input.data(), guarded_input.size());
            owned[0] = input_owner->asStrided(input_shape, {input_shape[1], 1}, input_shape[1]);
            auto quant = std::make_shared<NativeKernel>(runtime, quant_library,
                llaisys::backends::native::KernelIdentity{"tilelang", tilelang_version, "act_quant", 1});
            auto gemm = std::make_shared<NativeKernel>(runtime, gemm_library,
                llaisys::backends::native::KernelIdentity{"tilelang", tilelang_version, fp4 ? "fp4_gemm" : "fp8_gemm", 1});
            using llaisys::models::deepseek_v4::QuantizedLinear;
            using llaisys::models::deepseek_v4::LinearWorkspace;
            auto layer = std::make_unique<QuantizedLinear>(owned[1], owned[2], quant, gemm);
            std::weak_ptr<NativeTensor> weight_lease = owned[1], scale_lease = owned[2];
            owned[1].reset();
            owned[2].reset();
            require(!weight_lease.expired() && !scale_lease.expired(), "Linear failed to retain model weights");
            auto &input = *owned[0];
            auto &output = *owned[3];
            LinearWorkspace workspace(runtime, input.shape()[0], layer->inputDim());
            const auto pointer = workspace.activation.view().data;
            const auto original_stream = TVMFFIEnvGetStream(kDLCUDA, runtime.deviceId());
            for (int repeat = 0; repeat < 2; ++repeat) {
                layer->forward(input, output, workspace);
                require(workspace.activation.view().data == pointer, "Linear replaced its reusable workspace");
                require(TVMFFIEnvGetStream(kDLCUDA, runtime.deviceId()) == original_stream, "Linear changed ambient FFI stream");
                checkExact(output, expected[3], name);
                checkExact(workspace.activation, expected[4], name);
                checkExact(workspace.scales, expected[5], name);
                checked += 3;
            }
            checkExact(input, expected[0], name);
            checkExact(*input_owner, guarded_input, name + " input guard");
            ++checked;
            checkExact(*weight_lease.lock(), expected[1], name);
            checkExact(*scale_lease.lock(), expected[2], name);
            checked += 3;
            require(quant->calls() == 2 && gemm->calls() == 2 && !quant->failures() && !gemm->failures(),
                    "native Linear did not call the selected pair exactly twice");
            {
                NativeTensor empty_input(runtime, {0, layer->inputDim()}, {kDLBfloat, 16, 1});
                NativeTensor empty_output(runtime, {0, layer->outputDim()}, {kDLBfloat, 16, 1});
                LinearWorkspace empty_workspace(runtime, 0, layer->inputDim());
                layer->forward(empty_input, empty_output, empty_workspace);
                require(quant->calls() == 2 && gemm->calls() == 2, "empty Linear launched a GPU kernel");
            }
            bool rejected = false;
            if (layer->outputDim() <= layer->inputDim()) {
                auto alias = input.asStrided({input.shape()[0], layer->outputDim()}, {layer->outputDim(), 1});
                try { layer->forward(input, *alias, workspace); } catch (const std::invalid_argument &) { rejected = true; }
                require(rejected && quant->calls() == 2 && gemm->calls() == 2, "aliased output view was not rejected");
                ++negative;
                rejected = false;
            }
            NativeTensor bad_output(runtime, {input.shape()[0], layer->outputDim() + 1}, {kDLBfloat, 16, 1});
            try { layer->forward(input, bad_output, workspace); } catch (const std::invalid_argument &) { rejected = true; }
            require(rejected && quant->calls() == 2 && gemm->calls() == 2, "bad output was not rejected before quantization");
            ++negative;
            LinearWorkspace wrong_workspace(runtime, input.shape()[0], layer->inputDim() + 128);
            rejected = false;
            try { layer->forward(input, output, wrong_workspace); } catch (const std::invalid_argument &) { rejected = true; }
            require(rejected && quant->calls() == 2 && gemm->calls() == 2, "bad workspace was not rejected before quantization");
            ++negative;
            rejected = false;
            try { QuantizedLinear invalid(weight_lease.lock(), scale_lease.lock(), std::make_shared<FailingBackend>(runtime, 2), gemm); }
            catch (const std::invalid_argument &) { rejected = true; }
            require(rejected, "incompatible native operator contract revision accepted");
            ++negative;
            {
                auto failing = std::make_shared<FailingBackend>(runtime);
                QuantizedLinear custom(weight_lease.lock(), scale_lease.lock(), failing, gemm);
                rejected = false;
                try { custom.forward(input, output, workspace); } catch (const std::runtime_error &) { rejected = true; }
                require(rejected && failing->failures() == 1 && gemm->calls() == 2 && quant->calls() == 2,
                        "injected custom backend error triggered fallback");
                require(custom.quantBackend().backend == "test-user-library", "custom backend identity lost");
                checkExact(output, expected[3], name);
                ++checked;
                ++negative;
            }
            layer.reset();
            require(weight_lease.expired() && scale_lease.expired(), "Linear leaked model weight owners");
            quant->close();
            gemm->close();
            std::cout << "linear " << index << " " << name << " exact twice; weights released; no fallback\n";
        }
        manifest >> std::ws;
        require(manifest.eof(), "unparsed native test manifest content");
        bool rejected = false;
        try { NativeTensor::checkedBytes({INT64_MAX, 2}, {kDLBfloat, 16, 1}); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "overflowing descriptor accepted");
        ++negative;
        rejected = false;
        try { NativeTensor::checkedBytes({2, 8}, {kDLFloat4_e2m1fn, 8, 1}); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "invalid FP4 descriptor accepted");
        ++negative;
        require(NativeKernel::destructionErrors() == 0, "native destruction reported an error");
        require(paged_cases == 6 && llaisys::backends::native::PagedStorage::liveBytes() == 0
                && llaisys::backends::native::PagedStorage::destructionErrors() == 0,
                "paged sparse coverage or storage teardown failed");
        requireNativeProcess();
        std::cout << "{\"all_passed\":true,\"cases\":" << cases << ",\"tensor_byte_comparisons\":" << checked
                  << ",\"linear_cases\":" << linear_cases << ",\"linear_repetitions\":2"
                  << ",\"linear_input_offset_rows\":1,\"empty_linear_checks\":" << linear_cases
                  << ",\"negative_checks\":" << negative
                  << ",\"native_paged_sparse_cases\":" << paged_cases
                  << ",\"native_paged_live_bytes\":0,\"native_paged_destruction_errors\":0"
                  << ",\"python_interpreter\":false,\"torch_runtime\":false,\"fallback\":false,\"destruction_errors\":0}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "native TileLang test failed: " << error.what() << '\n';
        return 2;
    }
}
