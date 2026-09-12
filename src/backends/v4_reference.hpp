#pragma once
#include "src/models/deepseek_v4/model.hpp"
namespace llaisys::backends::v4_reference {
struct Bundle { std::string operation, library; };
// Explicit reference backend assembly, kept OUT of the model/runtime classes.
// Other operator libraries can supply their own ModelBackends/Session factory.
models::deepseek_v4::ModelBackends make(core::Runtime &runtime, const models::deepseek_v4::ModelConfig &config,
    const std::map<std::string, Bundle> &bundles, const std::string &tilelang_version, const std::string &hadamard_version);
}
