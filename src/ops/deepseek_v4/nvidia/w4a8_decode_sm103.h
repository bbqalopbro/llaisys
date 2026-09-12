#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Experimental B300 (SM103) small-M W4A8 projection, C[M,N] = A[M,K] B[N,K]^T.
// A: E4M3 FP8 bytes, SA: UE8M0 bytes with K/128 columns.
// B: contiguous E2M1 FP4, low nibble first, N*K/2 bytes.
// SB: contiguous UE8M0 bytes [N,K/32]. C: contiguous BF16 [M,N].
// A and SA row strides are in bytes (their elements occupy one byte).
// Supports 1<=M<=8, 1<=N<=65536, 128<=K<=32768 and K%128==0.
// A/B bases and A row stride must be 16-byte aligned; C must be 2-byte aligned.
// Caller owns all allocations and their lifetime on stream. C must not overlap
// an input. No allocation, synchronization, activation quantization or fallback.
// variant: 0 or 1 = four output warps/CTA; 2 = eight output warps/CTA;
//          3 = one output column/CTA, four warps cooperate along K.
// Returns a cudaError_t value as int. CUDA success means launch acceptance;
// completion errors are reported by the caller's subsequent stream/event wait.
int llaisys_w4a8_decode_sm103(
    const void *a, const uint8_t *sa, const uint8_t *b, const uint8_t *sb,
    void *c, int m, int n, int k, int64_t a_row_stride,
    int64_t sa_row_stride, cudaStream_t stream, int variant);

const char *llaisys_w4a8_decode_sm103_version(void);

#ifdef __cplusplus
}
#endif
