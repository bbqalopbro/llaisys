#pragma once

namespace llaisys::ops::nvidia {

void deepseek_v4_activation_quant_reference(
    void *data, int rows, int columns, int block_size,
    int quant_mode, int power_of_two_scale);

} // namespace llaisys::ops::nvidia
