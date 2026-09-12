#ifndef LLAISYS_V4_FP8_PREFILL_SM103_H
#define LLAISYS_V4_FP8_PREFILL_SM103_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/* Independent CUDA 13 + cuBLASLt backend; no Torch/ATen/TileLang dependency.
 * C[M,N] BF16 = A[M,K] E4M3 * transpose(B[N,K] E4M3).
 * as[M,K/128], bs[N/128,K/128] are contiguous UE8M0 bytes.
 * K and N must be multiples of 128. M>=1; non-16 M is internally padded.
 * UE8M0 scale bytes are replicated, never rounded/requantized. A/B FP8
 * payload is unchanged. 0xff scales (NaN) are caller errors, not sanitized.
 *
 * A plan owns a fixed device/stream, descriptors, scale buffers and workspace.
 * Create/select/destroy are initialization operations, outside graph capture.
 * One host caller and one stream per plan. Retain all external device pointers,
 * the plan and its stream until queued work/graph replays finish. No concurrent
 * executions/graph replays sharing one plan. Destroy synchronizes its stream.
 * prepare_b is required once, or after any B-scale change. prepare_a + gemm
 * are available separately for attribution. run includes A-scale preparation.
 * Execute paths allocate nothing and do no device synchronization. Warm run
 * once outside capture before capture/replay; capture/replay must use the
 * plan's stream and keep graph/pointer lifetimes valid.
 * Matrices and dynamic A scales must not overlap, including partial overlaps.
 * Return 0 on success; nonzero errors have a thread-local description.
 * destroy: return 1 means validation rejected and plan remains owned by caller;
 * return 2 means stream synchronization failed but plan was consumed and all
 * releases were attempted. A CUDA context failure may prevent successful frees.
 */
int llaisys_v4_fp8_create(int m, int n, int k, uintptr_t stream,
                         size_t workspace_bytes, void **out_plan);
int llaisys_v4_fp8_prepare_b(void *plan, const void *b_scales_ue8m0);
int llaisys_v4_fp8_prepare_a(void *plan, const void *a_scales_ue8m0);
int llaisys_v4_fp8_gemm(void *plan, const void *a_e4m3, const void *b_e4m3,
                       void *output_bf16);
int llaisys_v4_fp8_run(void *plan, const void *a_e4m3,
                      const void *a_scales_ue8m0, const void *b_e4m3,
                      void *output_bf16);
int llaisys_v4_fp8_algorithm_count(void *plan);
int llaisys_v4_fp8_select_algorithm(void *plan, int index);
int llaisys_v4_fp8_destroy(void *plan);
const char *llaisys_v4_fp8_last_error(void);
#ifdef __cplusplus
}
#endif
#endif
