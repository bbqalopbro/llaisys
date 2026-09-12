#pragma once

namespace llaisys::ops::nvidia {

void deepseek_v4_hyperconnection_split_reference(
    float *pre, float *post, float *combination,
    const float *mixes, const float *scale, const float *base,
    int rows, int hc_mult, int iterations, float eps);

} // namespace llaisys::ops::nvidia
