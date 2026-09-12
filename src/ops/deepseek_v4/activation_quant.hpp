#pragma once

#include "llaisys.h"

namespace llaisys::ops {

void deepseek_v4_activation_quant_reference(
    void *data, int rows, int columns, int block_size,
    int quant_mode, int power_of_two_scale,
    llaisysDataType_t dtype, llaisysDeviceType_t device_type);

} // namespace llaisys::ops
