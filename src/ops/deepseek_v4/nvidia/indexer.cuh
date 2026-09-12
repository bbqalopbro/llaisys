#pragma once
#include <cstddef>

namespace llaisys::ops::nvidia {
void deepseek_v4_indexer_scores_cublas(
    float *scores, void *dots, const void *query, const void *latent,
    const void *weights, int batch, int sequence, int candidates, void *stream);
size_t deepseek_v4_indexer_topk_workspace(int rows, int candidates);
void deepseek_v4_indexer_topk_cub(
    int *indices, const float *scores, void *workspace, size_t workspace_bytes,
    int batch, int sequence, int candidates, int topk, int start_pos,
    int ratio, int index_offset, void *stream);
}
