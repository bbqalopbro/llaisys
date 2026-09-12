// Native ATen correctness process. libtorch C++ is intentional; Python is not.
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>

#include "src/backends/aten/structural_kernel.hpp"
#include "src/backends/hadamard/native_kernel.hpp"
#include "src/core/context/context.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

using llaisys::backends::native::Tensor;
using llaisys::backends::aten::StructuralKernel;
using llaisys::backends::aten::StructuralOptions;

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static void noPython() {
    check(dlsym(RTLD_DEFAULT, "Py_IsInitialized") == nullptr, "Python interpreter unexpectedly loaded");
    std::ifstream maps("/proc/self/maps"); check(maps.good(), "cannot inspect process mappings");
    std::string line;
    while (std::getline(maps, line))
        check(line.find("libpython") == std::string::npos && line.find("libtorch_python") == std::string::npos,
              "Python runtime mapping unexpectedly loaded");
}
static std::vector<unsigned char> readBytes(const std::string &path, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(file.good() && file.tellg() == static_cast<std::streamoff>(bytes), "invalid reference tensor file size");
    file.seekg(0);
    std::vector<unsigned char> value(bytes);
    if (bytes) file.read(reinterpret_cast<char *>(value.data()), bytes);
    check(file.good(), "cannot read reference bytes");
    return value;
}
static std::vector<int64_t> dimensions(std::istream &stream, size_t rank) {
    check(rank && rank <= 16, "invalid tensor rank");
    std::vector<int64_t> dims(rank);
    for (auto &dim : dims) stream >> dim;
    check(stream.good(), "truncated tensor dimensions");
    return dims;
}

int main(int argc, char **argv) {
    try {
        check(argc == 2 && std::getenv("SLURM_JOB_ID"), "native ATen test requires a Slurm job and manifest");
        noPython();
        llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, 0);
        auto &runtime = llaisys::core::context().runtime();
        std::ifstream manifest(argv[1]);
        std::string magic, version, hadamard_version;
        size_t cases;
        manifest >> magic >> cases >> version >> hadamard_version;
        check(magic == "LLAISYS_ATEN_NATIVE_TEST_V1" && cases > 0 && cases < 256
              && version == StructuralKernel::compiledVersion(), "invalid manifest or ATen version mismatch");
        size_t compared = 0, negative = 0, rejection_cases = 0;
        for (size_t index = 0; index < cases; ++index) {
            std::string name, operation;
            StructuralOptions options;
            float hadamard_scale;
            size_t count; bool expect_failure;
            manifest >> name >> operation >> options.epsilon >> options.activation_limit >> options.route_scale
                     >> options.reduction_axis >> options.inverse_rotary >> options.stable_indexer_ties
                     >> options.hash_routing >> options.score_function >> hadamard_scale >> count >> expect_failure;
            check(manifest.good() && count > 0 && count <= 8, "invalid structural case");
            std::vector<std::shared_ptr<Tensor>> bases, views;
            std::vector<Tensor *> args;
            std::vector<std::vector<unsigned char>> expected;
            for (size_t i = 0; i < count; ++i) {
                unsigned code, bits, lanes; size_t rank, bytes;
                std::string input, output;
                manifest >> code >> bits >> lanes >> rank;
                check(code <= 255 && bits <= 255 && lanes <= 65535, "invalid dtype descriptor");
                auto shape = dimensions(manifest, rank);
                manifest >> bytes >> std::quoted(input) >> std::quoted(output);
                check(bytes < (1ull << 31), "oversized test tensor");
                auto base = std::make_shared<Tensor>(runtime, shape,
                    DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
                check(base->bytes() == bytes, "native tensor byte count mismatch");
                auto initial = readBytes(input, bytes); base->upload(initial.data(), initial.size());
                manifest >> rank;
                auto view_shape = dimensions(manifest, rank), strides = dimensions(manifest, rank);
                int64_t offset; manifest >> offset;
                auto view = base->asStrided(std::move(view_shape), std::move(strides), offset);
                args.push_back(view.get()); views.push_back(std::move(view)); bases.push_back(std::move(base));
                expected.push_back(readBytes(output, bytes));
            }
            std::unique_ptr<llaisys::backends::native::Kernel> implementation;
            if (operation == "hadamard")
                implementation = std::make_unique<llaisys::backends::hadamard::NativeKernel>(runtime,
                    llaisys::backends::native::KernelIdentity{"cuda-hadamard", hadamard_version, operation, 1}, hadamard_scale);
            else implementation = std::make_unique<StructuralKernel>(runtime,
                llaisys::backends::native::KernelIdentity{"aten-reference", version, operation, 1}, options);
            auto &kernel = *implementation;
            const auto original_stream = c10::cuda::getCurrentCUDAStream(0);
            bool failed = false;
            try { kernel.call(args); } catch (const std::exception &error) {
                if (!expect_failure) throw;
                failed = true;
                std::cout << "rejection " << name << ": " << error.what() << '\n';
            }
            check(failed == expect_failure, "expected backend rejection did not occur");
            rejection_cases += expect_failure;
            check(c10::cuda::getCurrentCUDAStream(0) == original_stream, "ATen stream not restored after call");
            for (size_t i = 0; i < bases.size(); ++i) {
                std::vector<unsigned char> actual(bases[i]->bytes());
                bases[i]->download(actual.data(), actual.size());
                if (actual != expected[i]) {
                    size_t mismatches = 0;
                    for (size_t j = 0; j < actual.size(); ++j) mismatches += actual[j] != expected[i][j];
                    throw std::runtime_error(name + " arg " + std::to_string(i) + " differs in " + std::to_string(mismatches) + " bytes");
                }
                ++compared;
            }
            failed = false;
            try { kernel.call({}); } catch (const std::exception &) { failed = true; }
            check(failed && kernel.failures() == 1 + static_cast<uint64_t>(expect_failure), "bad arity not reported");
            check(c10::cuda::getCurrentCUDAStream(0) == original_stream, "ATen error changed ambient stream");
            ++negative;
            failed = false;
            std::thread foreign([&] { try { kernel.call(args); } catch (const std::runtime_error &) { failed = true; } });
            foreign.join(); check(failed, "foreign-thread call accepted"); ++negative;
            kernel.close(); failed = false;
            try { kernel.call(args); } catch (const std::runtime_error &) { failed = true; }
            check(failed, "closed kernel call accepted"); ++negative;
            std::cout << "case " << index << " " << name << " exact\n";
        }
        manifest >> std::ws; check(manifest.eof(), "unparsed manifest data");
        {
            auto base = std::make_shared<Tensor>(runtime, std::vector<int64_t>{2, 4}, DLDataType{kDLBfloat, 16, 1});
            std::vector<uint16_t> payload{1, 2, 3, 4, 5, 6, 7, 8}; base->upload(payload.data(), 16);
            auto view = base->asStrided({2, 2}, {4, 1}, 1);
            check(view->overlaps(*base) && !view->isContiguous(), "strided view alias metadata wrong");
            bool failed = false;
            try { view->download(payload.data(), view->bytes()); } catch (const std::invalid_argument &) { failed = true; }
            check(failed, "strided raw transfer accepted"); ++negative;
            failed = false;
            try { base->asStrided({2, 8}, {8, 1}); } catch (const std::invalid_argument &) { failed = true; }
            check(failed, "out-of-bounds view accepted"); ++negative;
            failed = false;
            try { base->asStrided({2, 2}, {-4, 1}); } catch (const std::invalid_argument &) { failed = true; }
            check(failed, "negative stride accepted"); ++negative;
            base.reset();
            auto row = view->asStrided({2}, {1});
            std::vector<uint16_t> actual(2); row->download(actual.data(), 4);
            check(actual == std::vector<uint16_t>{2, 3}, "view did not retain native storage after original owner release");
            Tensor empty(runtime, {1, 3, 0}, {kDLInt, 64, 1});
            check(empty.bytes() == 0 && empty.view().data == nullptr, "empty descriptor allocated nonempty storage");
            empty.upload(nullptr, 0); empty.download(nullptr, 0);
            Tensor packed(runtime, {2, 4}, {kDLFloat4_e2m1fn, 4, 1});
            failed = false;
            try { packed.asStrided({2}, {1}, 1); } catch (const std::invalid_argument &) { failed = true; }
            check(failed, "half-byte view offset accepted"); ++negative;
        }
        noPython();
        std::cout << "{\"all_passed\":true,\"cases\":" << cases << ",\"rejection_cases\":" << rejection_cases
                  << ",\"tensor_byte_comparisons\":" << compared << ",\"negative_checks\":" << negative
                  << ",\"python_interpreter\":false,\"aten_cpp\":true,\"fallback\":false,\"allow_tf32_cublas\":"
                  << (at::globalContext().allowTF32CuBLAS() ? "true" : "false") << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "native ATen verification failed: " << error.what() << '\n';
        return 2;
    }
}
