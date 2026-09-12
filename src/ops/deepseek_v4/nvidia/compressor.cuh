#pragma once

namespace llaisys::ops::nvidia {
// Projected FP32 inputs, before BF16 rounding, RMSNorm, RoPE and QAT.
// State is [B, coefficient * ratio, coefficient * dimension], coefficient=2
// for ratio 4, otherwise 1. start_pos=0 resets both state buffers.
void deepseek_v4_compress_projected_reference(
    float *output, float *kv_state, float *score_state,
    const float *kv, const float *score, const float *ape,
    int batch, int sequence, int dimension, int ratio, int start_pos,
    void *stream);
}
