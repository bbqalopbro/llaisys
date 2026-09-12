#include "activation_quant.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/activation_quant.cuh"
#endif

#include <stdexcept>

namespace llaisys::ops {

void deepseek_v4_activation_quant_reference(
    void *data, int rows, int columns, int block_size,
    int quant_mode, int power_of_two_scale,
    llaisysDataType_t dtype, llaisysDeviceType_t device_type) {
    if (!data || rows <= 0 || columns <= 0 || block_size <= 0 ||
        columns % block_size != 0 || (quant_mode != 0 && quant_mode != 1)) {
        throw std::invalid_argument("invalid DeepSeek-V4 activation quant arguments");
    }
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA && dtype == LLAISYS_DTYPE_BF16) {
        return nvidia::deepseek_v4_activation_quant_reference(
            data, rows, columns, block_size, quant_mode, power_of_two_scale);
    }
#endif
    throw std::invalid_argument(
        "DeepSeek-V4 activation quant supports NVIDIA BF16 only");
}

} // namespace llaisys::ops
