#pragma once

#include "llaisys.h"

namespace llaisys::ops {

void deepseek_v4_hyperconnection_split_reference(
    float *pre, float *post, float *combination,
    const float *mixes, const float *scale, const float *base,
    int rows, int hc_mult, int iterations, float eps,
    llaisysDeviceType_t device_type);

} // namespace llaisys::ops
