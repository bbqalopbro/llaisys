#pragma once

#include "tensor.hpp"
#include <stdexcept>
#include <string>

namespace llaisys::backends::native {

struct KernelIdentity {
    std::string backend;
    std::string version;
    std::string operation;
    uint32_t contract_revision = 1;
};

// Host-known launch metadata, not a device tensor or Python callback. Each
// versioned operator defines its exact positional integer/real scalar schema.
// Existing operators reject nonempty metadata unless explicitly implemented.
struct CallScalars {
    std::vector<int64_t> integers;
    std::vector<double> reals;
};

// C++ operator-library seam. No TVM, Torch or Python types cross this interface.
// Implementations enqueue on runtime().stream(), honor the selected contract,
// retain their code through completion, and throw rather than switch backends.
// The caller owns inputs/outputs/workspace through queued execution. Graph
// capture, concurrent streams and arbitrary strided views are not yet promised.
class Kernel {
public:
    Kernel(core::Runtime &runtime, KernelIdentity identity)
        : runtime_(runtime), identity_(std::move(identity)) {
        if (identity_.backend.empty() || identity_.version.empty() || identity_.operation.empty()
            || !identity_.contract_revision)
            throw std::invalid_argument("explicit native kernel identity and contract revision required");
    }
    virtual ~Kernel() = default;
    Kernel(const Kernel &) = delete;
    Kernel &operator=(const Kernel &) = delete;
    virtual void call(const std::vector<Tensor *> &arguments) = 0;
    virtual void callWithScalars(const std::vector<Tensor *> &arguments, const CallScalars &scalars) {
        if (!scalars.integers.empty() || !scalars.reals.empty())
            throw std::invalid_argument("this native kernel contract does not accept launch scalars");
        call(arguments);
    }
    virtual void close() = 0;
    virtual uint64_t calls() const = 0;
    virtual uint64_t failures() const = 0;
    core::Runtime &runtime() const { return runtime_; }
    const KernelIdentity &identity() const { return identity_; }

protected:
    core::Runtime &runtime_;
    const KernelIdentity identity_;
};

} // namespace llaisys::backends::native
