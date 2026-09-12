// Parse vendor SIMD headers before llaisys.h's legacy __C linkage macro.
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/version.h>

#include "structural_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <set>

namespace llaisys::backends::aten {
namespace {
using at::indexing::Slice;
void check(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}
bool floating(const at::Tensor &x) { return x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat; }
bool integer(const at::Tensor &x) { return x.scalar_type() == at::kInt || x.scalar_type() == at::kLong; }

at::ScalarType scalarType(DLDataType dtype) {
    check(dtype.lanes == 1, "ATen structural backend requires scalar lanes");
    if (dtype.code == kDLBfloat && dtype.bits == 16) return at::kBFloat16;
    if (dtype.code == kDLFloat && dtype.bits == 32) return at::kFloat;
    if (dtype.code == kDLInt && dtype.bits == 32) return at::kInt;
    if (dtype.code == kDLInt && dtype.bits == 64) return at::kLong;
    if (dtype.code == kDLComplex && dtype.bits == 64) return at::kComplexFloat;
    throw std::invalid_argument("unsupported ATen structural tensor dtype");
}

at::Tensor borrow(native::Tensor &tensor) {
    auto view = tensor.view();
    auto options = at::TensorOptions().dtype(scalarType(view.dtype)).device(at::kCUDA, view.device.device_id);
    // ATen from_blob probes the CUDA device of its pointer even for zero
    // elements. An empty native tensor intentionally has no allocation/pointer;
    // represent its metadata with ATen's zero-storage constructor instead.
    if (!tensor.bytes())
        return at::empty_strided({view.shape, static_cast<size_t>(view.ndim)},
                                 {view.strides, static_cast<size_t>(view.ndim)}, options);
    void *data = view.data ? static_cast<void *>(static_cast<char *>(view.data) + view.byte_offset) : nullptr;
    // The view does not acquire/deallocate the pointer. Native owners remain
    // alive through this call and its queued work; ATen copies shape/strides.
    return at::from_blob(data, {view.shape, static_cast<size_t>(view.ndim)},
                         {view.strides, static_cast<size_t>(view.ndim)}, [](void *) {},
                         options);
}

void expect(const at::Tensor &output, const std::vector<int64_t> &shape, at::ScalarType dtype) {
    check(output.sizes().vec() == shape && output.scalar_type() == dtype, "ATen output shape/dtype mismatch");
}

at::Tensor invRms(const at::Tensor &x, double eps) {
    return at::rsqrt(x.square().mean(-1, true) + eps);
}

void validIds(const at::Tensor &ids, int64_t limit) {
    check(limit >= 0 && (limit > 0 || !ids.numel()), "valid ID limit required");
    if (ids.numel()) check(at::logical_and(ids >= 0, ids < limit).all().item<bool>(), "ID outside configured range");
}
}

struct StructuralKernel::Impl {
    std::string operation;
    StructuralOptions options;
    Impl(std::string op, StructuralOptions opts) : operation(std::move(op)), options(std::move(opts)) {}

    void run(const std::vector<native::Tensor *> &native_args, const native::CallScalars &scalars) const {
        if (operation != "tensor_fill" && operation != "compressor_prepare" && operation != "tensor_scale"
            && operation != "indexer_mask" && operation != "indexer_remap" && operation != "attention_prepare" && operation != "rotary_frequencies")
            check(scalars.integers.empty() && scalars.reals.empty(), "unexpected structural launch scalars");
        std::vector<at::Tensor> a;
        a.reserve(native_args.size());
        for (auto *arg : native_args) a.push_back(borrow(*arg));
        auto count = [&](size_t n) { check(a.size() == n, "wrong ATen structural argument count"); };
        auto output = [&](size_t index, const std::vector<int64_t> &shape, at::ScalarType dtype,
                          std::initializer_list<size_t> inputs) {
            expect(a[index], shape, dtype);
            at::assert_no_internal_overlap(a[index]);
            for (size_t input : inputs)
                check(!native_args[index]->overlaps(*native_args[input]), "ATen output aliases a read-only input");
        };
        if (operation == "token_embedding") {
            count(3);
            check(a[0].dim() == 2 && a[0].scalar_type() == at::kBFloat16 && a[0].size(0) > 0 && a[0].size(1) > 0
                  && a[1].dim() == 2 && integer(a[1]) && a[2].dim() == 4 && a[2].size(2) > 0, "invalid token embedding tensors");
            output(2, {a[1].size(0), a[1].size(1), a[2].size(2), a[0].size(1)}, at::kBFloat16, {0, 1});
            validIds(a[1], a[0].size(0));
            a[2].copy_(at::embedding(a[0], a[1]).unsqueeze(2).expand_as(a[2]));
        } else if (operation == "greedy_argmax") {
            count(2);
            check(a[0].dim() == 2 && a[0].scalar_type() == at::kFloat && a[0].size(1) > 0, "greedy argmax requires FP32 logits");
            output(1, {a[0].size(0)}, at::kLong, {0});
            check(at::isfinite(a[0]).all().item<bool>(), "greedy argmax received non-finite logits");
            a[1].copy_(a[0].argmax(-1));
        } else if (operation == "rotary_frequencies") {
            count(1);
            check(a[0].dim() == 2 && a[0].scalar_type() == at::kComplexFloat && a[0].size(0) > 0 && a[0].size(1) > 0
                  && a[0].size(1) <= INT32_MAX / 2 && scalars.integers.size() == 1 && scalars.reals.size() == 4,
                  "RoPE frequencies require complex64 [capacity,R/2], original length and base/factor/beta_fast/beta_slow");
            const int64_t original = scalars.integers[0], dim = 2 * a[0].size(1);
            const double base = scalars.reals[0], factor = scalars.reals[1], fast = scalars.reals[2], slow = scalars.reals[3];
            check(original >= 0 && std::isfinite(base) && base > 1 && std::isfinite(factor) && factor > 0
                  && std::isfinite(fast) && fast > 0 && std::isfinite(slow) && slow > 0, "invalid RoPE frequency scalars");
            at::assert_no_internal_overlap(a[0]); auto opts = a[0].options().dtype(at::kFloat);
            auto frequencies = 1.0 / at::pow(base, at::arange(0, dim, 2, opts) / dim);
            if (original > 0) {
                auto correction = [&](double rotations) { return dim * std::log(original / (rotations * 2 * std::acos(-1.0))) / (2 * std::log(base)); };
                double low = std::max(std::floor(correction(fast)), 0.0), high = std::min(std::ceil(correction(slow)), static_cast<double>(dim - 1));
                if (low == high) high += 0.001;
                auto smooth = 1 - ((at::arange(dim / 2, opts) - low) / (high - low)).clamp(0, 1);
                frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth;
            }
            auto phase = at::outer(at::arange(a[0].size(0), opts.dtype(at::kLong)), frequencies);
            a[0].copy_(at::polar(at::ones_like(phase), phase));
        } else if (operation == "tensor_fill") {
            count(1);
            check(scalars.integers.empty() && scalars.reals.size() == 1 && floating(a[0])
                  && !std::isnan(scalars.reals[0]), "tensor_fill expects one real scalar and BF16/FP32 output");
            at::assert_no_internal_overlap(a[0]);
            a[0].fill_(scalars.reals[0]);
        } else if (operation == "compressor_prepare") {
            count(7);
            check(scalars.integers.size() == 2 && scalars.reals.empty(), "compressor_prepare expects position,ratio");
            const int64_t position = scalars.integers[0], ratio = scalars.integers[1];
            check(position >= 0 && (ratio == 4 || ratio == 128) && a[0].dim() == 3 && a[0].size(0) == 1
                  && a[0].size(1) > 0 && a[0].size(1) <= INT64_MAX - ratio && a[0].sizes() == a[1].sizes()
                  && a[0].scalar_type() == at::kFloat && a[1].scalar_type() == at::kFloat,
                  "invalid compressor projected tensors/metadata");
            const bool overlap = ratio == 4;
            const int64_t coff = overlap ? 2 : 1, width = a[0].size(2), sequence = a[0].size(1);
            check(width > 0 && width % coff == 0 && position <= INT64_MAX - sequence, "invalid compression width/end position");
            const int64_t dim = width / coff, complete = (position % ratio + sequence) / ratio;
            expect(a[2], {ratio, width}, at::kFloat);
            const std::vector<int64_t> state_shape{1, coff * ratio, width};
            output(3, state_shape, at::kFloat, {0, 1, 2, 4, 5, 6});
            output(4, state_shape, at::kFloat, {0, 1, 2, 3, 5, 6});
            const bool decode_emit = position > 0 && sequence == 1 && complete;
            const auto grouped_shape = decode_emit ? std::vector<int64_t>{1, coff * ratio, dim}
                                                  : std::vector<int64_t>{1, complete, coff * ratio, dim};
            output(5, grouped_shape, at::kFloat, {0, 1, 2, 3, 4, 6});
            output(6, grouped_shape, at::kFloat, {0, 1, 2, 3, 4, 5});
            const int64_t offset = overlap ? ratio : 0;
            auto pack = [&](const at::Tensor &kv, const at::Tensor &scores, bool previous_state) {
                if (!overlap) { a[5].copy_(kv); a[6].copy_(scores); return; }
                a[5].zero_(); a[6].fill_(-INFINITY);
                a[5].slice(2, ratio).copy_(kv.slice(3, dim));
                a[6].slice(2, ratio).copy_(scores.slice(3, dim));
                if (previous_state) {
                    a[5].select(1, 0).slice(1, 0, ratio).copy_(a[3].slice(1, 0, ratio).slice(2, 0, dim));
                    a[6].select(1, 0).slice(1, 0, ratio).copy_(a[4].slice(1, 0, ratio).slice(2, 0, dim));
                }
                a[5].slice(1, 1).slice(2, 0, ratio).copy_(kv.slice(1, 0, complete - 1).slice(3, 0, dim));
                a[6].slice(1, 1).slice(2, 0, ratio).copy_(scores.slice(1, 0, complete - 1).slice(3, 0, dim));
            };
            if (position == 0) {
                const int64_t cutoff = sequence - sequence % ratio;
                if (overlap && cutoff >= ratio) {
                    a[3].slice(1, 0, ratio).copy_(a[0].slice(1, cutoff - ratio, cutoff));
                    a[4].slice(1, 0, ratio).copy_(a[1].slice(1, cutoff - ratio, cutoff) + a[2]);
                }
                if (sequence > cutoff) {
                    a[3].slice(1, offset, offset + sequence - cutoff).copy_(a[0].slice(1, cutoff));
                    a[4].slice(1, offset, offset + sequence - cutoff).copy_(a[1].slice(1, cutoff) + a[2].slice(0, 0, sequence - cutoff));
                }
                if (!complete) return;
                auto kv = a[0].slice(1, 0, cutoff).reshape({1, complete, ratio, width});
                auto scores = a[1].slice(1, 0, cutoff).reshape({1, complete, ratio, width}) + a[2];
                pack(kv, scores, false);
            } else if (sequence > 1) {
                const int64_t partial = position % ratio, cutoff = complete * ratio;
                auto ape_ids = (at::arange(sequence, a[0].options().dtype(at::kLong)) + position).remainder(ratio);
                auto scores = a[1] + a[2].index_select(0, ape_ids);
                auto kv = at::cat({a[3].slice(1, offset, offset + partial), a[0]}, 1);
                scores = at::cat({a[4].slice(1, offset, offset + partial), scores}, 1);
                const int64_t remainder = kv.size(1) - cutoff;
                if (remainder) {
                    a[3].slice(1, offset, offset + remainder).copy_(kv.slice(1, cutoff));
                    a[4].slice(1, offset, offset + remainder).copy_(scores.slice(1, cutoff));
                }
                if (!complete) return;
                kv = kv.slice(1, 0, cutoff).reshape({1, complete, ratio, width});
                scores = scores.slice(1, 0, cutoff).reshape({1, complete, ratio, width});
                pack(kv, scores, true);
                if (overlap) {
                    a[3].slice(1, 0, ratio).copy_(kv.select(1, complete - 1));
                    a[4].slice(1, 0, ratio).copy_(scores.select(1, complete - 1));
                }
            } else {
                const int64_t slot = offset + position % ratio;
                a[3].slice(1, slot, slot + 1).copy_(a[0]);
                a[4].slice(1, slot, slot + 1).copy_(a[1] + a[2].select(0, position % ratio));
                if (!complete) return;
                if (overlap) {
                    a[5].copy_(at::cat({a[3].slice(1, 0, ratio).slice(2, 0, dim), a[3].slice(1, ratio).slice(2, dim)}, 1));
                    a[6].copy_(at::cat({a[4].slice(1, 0, ratio).slice(2, 0, dim), a[4].slice(1, ratio).slice(2, dim)}, 1));
                    a[3].slice(1, 0, ratio).copy_(a[3].slice(1, ratio));
                    a[4].slice(1, 0, ratio).copy_(a[4].slice(1, ratio));
                } else { a[5].copy_(a[3]); a[6].copy_(a[4]); }
            }
        } else if (operation == "tensor_row_multiply") {
            count(2);
            check(a[0].dim() >= 2 && floating(a[0]), "row multiply expects BF16/FP32 input");
            auto shape = a[0].sizes().vec(); shape.back() = 1;
            expect(a[1], shape, a[0].scalar_type());
            at::assert_no_internal_overlap(a[0]);
            check(!native_args[0]->overlaps(*native_args[1]), "row multiply aliases its factors");
            a[0].mul_(a[1]);
        } else if (operation == "attention_prepare") {
            // Explicit contiguous reference path: construct candidate metadata,
            // gather a bounded window for chunks and update the circular cache.
            count(5);
            check(scalars.integers.size() == 4 && scalars.reals.empty(), "attention_prepare expects position,window,ratio,topk");
            const int64_t p = scalars.integers[0], win = scalars.integers[1], ratio = scalars.integers[2], topk = scalars.integers[3];
            check(p >= 0 && win > 0 && win <= INT32_MAX / 4 && topk > 0 && (ratio == 0 || ratio == 4 || ratio == 128)
                  && a[0].dim() == 3 && a[0].size(0) == 1 && a[0].size(1) > 0 && a[0].size(2) > 0
                  && a[0].scalar_type() == at::kBFloat16 && p <= INT32_MAX - a[0].size(1), "invalid Attention metadata/input");
            const int64_t s = a[0].size(1), d = a[0].size(2), end = p + s, groups = ratio ? end / ratio : 0;
            const int64_t history = p > 0 && s > 1 ? std::min(p, win - 1) : 0;
            const bool decode = p > 0 && s == 1;
            const int64_t width = p == 0 ? std::min(s, win) : win;
            const int64_t extra = ratio == 4 ? std::min(topk, groups) : groups;
            const int64_t offset = decode ? win : history + s;
            check(a[1].dim() == 3 && a[1].size(0) == 1 && a[1].size(2) == d && a[1].scalar_type() == at::kBFloat16
                  && a[1].size(1) >= win + groups && (ratio || a[1].size(1) == win)
                  && a[1].size(1) <= INT32_MAX && offset <= INT32_MAX - groups && width <= INT32_MAX - extra,
                  "invalid Attention circular/compressed cache capacity");
            output(1, a[1].sizes().vec(), at::kBFloat16, {0, 2, 3, 4});
            expect(a[2], {1, s, ratio == 4 ? extra : 0}, at::kInt);
            output(3, {1, decode ? 0 : offset + groups, d}, at::kBFloat16, {0, 1, 2, 4});
            output(4, {1, s, width + extra}, at::kInt, {0, 1, 2, 3});
            auto positions = at::arange(p, end, a[0].options().dtype(at::kLong));
            auto logical = (positions.unsqueeze(1) - win + 1).clamp_min(0) + at::arange(width, positions.options());
            auto slots = decode ? logical.remainder(win) : logical - (p - history);
            slots = at::where(logical <= positions.unsqueeze(1), slots, -1);
            a[4].slice(2, 0, width).copy_(slots.unsqueeze(0).to(at::kInt));
            if (ratio == 4) {
                if (a[2].numel()) check(at::logical_and(a[2] >= -1, a[2] < offset + groups).all().item<bool>(), "Indexer candidate outside Attention payload");
                a[4].slice(2, width).copy_(a[2]);
            } else if (ratio) {
                auto ids = at::arange(groups, positions.options());
                auto counts = at::floor_divide(positions + 1, ratio).unsqueeze(1);
                a[4].slice(2, width).copy_(at::where(ids.unsqueeze(0) < counts, ids + offset, -1).unsqueeze(0).to(at::kInt));
            }
            if (!decode) {
                if (history) {
                    auto old = at::arange(p - history, p, positions.options()).remainder(win);
                    a[3].slice(1, 0, history).copy_(a[1].index_select(1, old));
                }
                a[3].slice(1, history, offset).copy_(a[0]);
                if (groups) a[3].slice(1, offset).copy_(a[1].slice(1, win, win + groups));
            }
            const int64_t retained = std::min(s, win);
            auto destinations = at::arange(end - retained, end, positions.options()).remainder(win);
            a[1].index_copy_(1, destinations, a[0].slice(1, s - retained));
        } else if (operation == "tensor_scale") {
            count(2);
            check(scalars.integers.empty() && scalars.reals.size() == 1 && std::isfinite(scalars.reals[0])
                  && floating(a[0]), "tensor_scale expects BF16/FP32 input and one finite real scalar");
            output(1, a[0].sizes().vec(), a[0].scalar_type(), {0});
            a[1].copy_(a[0] * scalars.reals[0]);
        } else if (operation == "indexer_mask" || operation == "indexer_remap") {
            const bool mask = operation == "indexer_mask";
            count(mask ? 1 : 2);
            check(scalars.integers.size() == static_cast<size_t>(mask ? 2 : 3) && scalars.reals.empty(),
                  "Indexer metadata expects position,ratio[,offset]");
            const int64_t position = scalars.integers[0], ratio = scalars.integers[1];
            check(position >= 0 && ratio == 4 && a[0].dim() == 3 && a[0].size(0) == 1
                  && a[0].size(1) > 0 && position <= INT32_MAX - a[0].size(1), "invalid Indexer position/rank");
            const int64_t tokens = a[0].size(1), candidates = (position + tokens) / ratio;
            auto counts = at::floor_divide(at::arange(position + 1, position + tokens + 1, a[0].options().dtype(at::kLong)), ratio).unsqueeze(1);
            if (mask) {
                check(a[0].scalar_type() == at::kBFloat16 && a[0].size(2) == candidates, "invalid Indexer score shape/dtype");
                at::assert_no_internal_overlap(a[0]);
                if (position == 0 || tokens > 1) {
                    auto invalid = at::arange(candidates, counts.options()).repeat({tokens, 1}) >= counts;
                    a[0].add_(at::where(invalid, -INFINITY, 0.0));
                }
            } else {
                const int64_t offset = scalars.integers[2];
                check(a[0].scalar_type() == at::kLong && a[0].size(2) <= candidates && offset >= 0
                      && offset <= INT32_MAX - candidates, "invalid Indexer offset/indices");
                output(1, a[0].sizes().vec(), at::kInt, {0});
                // Explicit checked reference backend: ID validation may synchronize.
                validIds(a[0], candidates);
                auto result = position == 0 || tokens > 1 ? at::where(a[0] >= counts, -1, a[0] + offset) : a[0] + offset;
                a[1].copy_(result.to(at::kInt));
            }
        } else if (operation == "tensor_cast") {
            count(2);
            check((floating(a[0]) && floating(a[1])) || (integer(a[0]) && integer(a[1])),
                  "tensor cast requires BF16/FP32 or int32/int64 pairs");
            output(1, a[0].sizes().vec(), a[1].scalar_type(), {0});
            if (a[0].scalar_type() == at::kLong && a[1].scalar_type() == at::kInt && a[0].numel())
                check(at::logical_and(a[0] >= INT32_MIN, a[0] <= INT32_MAX).all().item<bool>(),
                      "int64 to int32 cast would lose values");
            a[1].copy_(a[0].to(a[1].scalar_type()));
        } else if (operation == "row_gather") {
            count(3);
            check(a[0].dim() == 2 && a[1].dim() == 1 && integer(a[1]), "invalid row gather ranks/IDs");
            output(2, {a[1].size(0), a[0].size(1)}, a[0].scalar_type(), {0, 1});
            validIds(a[1], a[0].size(0));
            a[2].copy_(a[0].index_select(0, a[1].to(at::kLong)));
        } else if (operation == "moe_finalize") {
            count(3);
            check(a[0].dim() == 2 && a[0].scalar_type() == at::kFloat && a[1].scalar_type() == at::kBFloat16
                  && a[0].sizes() == a[1].sizes(), "MoE finalize requires FP32 routed plus BF16 shared outputs");
            output(2, a[0].sizes().vec(), at::kBFloat16, {0, 1});
            a[2].copy_((a[0] + a[1]).to(at::kBFloat16));
        } else if (operation == "dense_linear") {
            count(3);
            check(a[0].dim() >= 2 && a[1].dim() == 2 && floating(a[0]) && a[0].scalar_type() == a[1].scalar_type()
                  && a[0].size(-1) == a[1].size(-1), "invalid dense Linear inputs");
            auto shape = a[0].sizes().vec(); shape.back() = a[1].size(0);
            output(2, shape, a[0].scalar_type(), {0, 1});
            a[2].copy_(at::linear(a[0], a[1]));
        } else if (operation == "grouped_linear") {
            count(3);
            check(a[0].dim() == 4 && a[1].dim() == 3 && floating(a[0]) && a[0].scalar_type() == a[1].scalar_type()
                  && a[0].size(2) == a[1].size(0) && a[0].size(3) == a[1].size(2), "invalid grouped Linear inputs");
            output(2, {a[0].size(0), a[0].size(1), a[0].size(2), a[1].size(1)}, a[0].scalar_type(), {0, 1});
            a[2].copy_(at::einsum("bsgd,grd->bsgr", {a[0], a[1]}));
        } else if (operation == "row_inv_rms") {
            count(2);
            check(a[0].dim() >= 2 && a[0].size(-1) > 0 && floating(a[0]), "invalid inverse RMS input");
            auto shape = a[0].sizes().vec(); shape.back() = 1;
            output(1, shape, a[0].scalar_type(), {0});
            a[1].copy_(invRms(a[0], options.epsilon));
        } else if (operation == "rms_norm") {
            count(3);
            check(a[0].dim() >= 2 && a[0].size(-1) > 0 && floating(a[0]) && a[1].dim() == 1
                  && a[1].size(0) == a[0].size(-1) && a[1].scalar_type() == at::kFloat, "invalid RMSNorm inputs");
            output(2, a[0].sizes().vec(), a[0].scalar_type(), {0, 1});
            auto x = a[0].to(at::kFloat);
            a[2].copy_((a[1] * (x * invRms(x, options.epsilon))).to(a[0].scalar_type()));
        } else if (operation == "hc_pre_reduce") {
            count(3);
            check(a[0].dim() == 4 && a[0].scalar_type() == at::kFloat && a[0].size(2) > 0 && a[0].size(3) > 0,
                  "HC pre requires FP32 [B,S,C,D]");
            expect(a[1], {a[0].size(0), a[0].size(1), a[0].size(2)}, at::kFloat);
            output(2, {a[0].size(0), a[0].size(1), a[0].size(3)}, at::kBFloat16, {0, 1});
            // Preserve the published FP32 multiply -> FP32 reduction -> BF16 boundary.
            a[2].copy_((a[1].unsqueeze(-1) * a[0]).sum(2).to(at::kBFloat16));
        } else if (operation == "hc_post_mix") {
            count(5);
            check(a[0].dim() == 3 && a[0].scalar_type() == at::kBFloat16
                  && a[1].dim() == 4 && a[1].scalar_type() == at::kBFloat16 && a[1].size(2) > 0
                  && a[0].size(0) == a[1].size(0) && a[0].size(1) == a[1].size(1)
                  && a[0].size(2) > 0 && a[0].size(2) == a[1].size(3), "invalid HC post hidden/residual");
            const int64_t b = a[0].size(0), s = a[0].size(1), c = a[1].size(2);
            expect(a[2], {b, s, c}, at::kFloat); expect(a[3], {b, s, c, c}, at::kFloat);
            output(4, a[1].sizes().vec(), at::kBFloat16, {0, 1, 2, 3});
            // comb axes are [source copy, destination copy], NOT the transpose.
            a[4].copy_((a[2].unsqueeze(-1) * a[0].unsqueeze(-2)
                       + (a[3].unsqueeze(-1) * a[1].unsqueeze(-2)).sum(2)).to(at::kBFloat16));
        } else if (operation == "hc_head_weights") {
            count(4);
            check(a[0].dim() == 3 && a[0].scalar_type() == at::kFloat && a[0].size(2) > 0,
                  "HC head requires FP32 mixes [B,S,C]");
            expect(a[1], {1}, at::kFloat); expect(a[2], {a[0].size(2)}, at::kFloat);
            output(3, a[0].sizes().vec(), at::kFloat, {0, 1, 2});
            a[3].copy_(at::sigmoid(a[0] * a[1] + a[2]) + options.epsilon);
        } else if (operation == "hc_sum") {
            count(2);
            const int64_t dim = options.reduction_axis < 0 ? options.reduction_axis + a[0].dim() : options.reduction_axis;
            check(a[0].scalar_type() == at::kFloat && dim >= 0 && dim < a[0].dim(), "invalid HC reduction");
            auto shape = a[0].sizes().vec(); shape.erase(shape.begin() + dim);
            output(1, shape, at::kFloat, {0});
            a[1].copy_(a[0].sum(dim));
        } else if (operation == "rotary_inplace") {
            count(2);
            check((a[0].dim() == 3 || a[0].dim() == 4) && floating(a[0]) && a[0].size(-1) % 2 == 0,
                  "invalid RoPE input");
            expect(a[1], {a[0].size(1), a[0].size(-1) / 2}, at::kComplexFloat);
            at::assert_no_internal_overlap(a[0]);
            check(!native_args[0]->overlaps(*native_args[1]), "RoPE input aliases frequencies");
            auto shape = a[0].sizes().vec(); shape.back() /= 2; shape.push_back(2);
            auto x = at::view_as_complex(a[0].to(at::kFloat).reshape(shape));
            auto freqs = options.inverse_rotary ? a[1].conj() : a[1];
            freqs = a[0].dim() == 3 ? freqs.reshape({1, a[0].size(1), -1})
                                    : freqs.reshape({1, a[0].size(1), 1, -1});
            a[0].copy_(at::view_as_real(x * freqs).flatten(-2).to(a[0].scalar_type()));
        } else if (operation == "indexer_scores") {
            count(4);
            check(a[0].dim() == 4 && a[1].dim() == 3 && a[2].dim() == 3
                  && a[0].size(0) == a[1].size(0) && a[0].size(3) == a[1].size(2)
                  && a[2].sizes().vec() == std::vector<int64_t>{a[0].size(0), a[0].size(1), a[0].size(2)}
                  && a[0].scalar_type() == at::kBFloat16 && a[1].scalar_type() == at::kBFloat16
                  && a[2].scalar_type() == at::kBFloat16, "invalid Indexer inputs");
            output(3, {a[0].size(0), a[0].size(1), a[1].size(1)}, at::kBFloat16, {0, 1, 2});
            auto scores = at::einsum("bshd,btd->bsht", {a[0], a[1]});
            a[3].copy_((scores.relu_() * a[2].unsqueeze(-1)).sum(2));
        } else if (operation == "indexer_topk") {
            count(2);
            check(a[0].dim() == 3 && floating(a[0]) && a[1].dim() == 3, "invalid Indexer top-k ranks");
            const int64_t k = a[1].size(2);
            check(k <= a[0].size(2), "Indexer top-k exceeds candidate count");
            output(1, {a[0].size(0), a[0].size(1), k}, at::kLong, {0});
            auto result = options.stable_indexer_ties ? at::argsort(a[0], true, -1, true).slice(-1, 0, k)
                                                     : std::get<1>(a[0].topk(k, -1));
            a[1].copy_(result);
        } else if (operation == "router") {
            count(4);
            check(a[0].dim() == 2 && a[0].scalar_type() == at::kFloat && a[2].dim() == 2, "invalid router logits");
            const int64_t tokens = a[0].size(0), experts = a[0].size(1), k = a[2].size(1);
            check(k > 0 && k <= experts, "invalid router top-k");
            if (options.hash_routing) {
                check(integer(a[1]) && a[1].sizes().vec() == std::vector<int64_t>{tokens, k}, "invalid hash routing IDs");
                validIds(a[1], experts);
                if (k > 1 && tokens) {
                    auto sorted = std::get<0>(a[1].sort(-1));
                    check((sorted.slice(1, 1) != sorted.slice(1, 0, k - 1)).all().item<bool>(),
                          "hash routing requires distinct experts per token");
                }
            } else expect(a[1], {experts}, at::kFloat);
            output(2, {tokens, k}, at::kFloat, {0, 1});
            output(3, {tokens, k}, options.hash_routing ? a[1].scalar_type() : at::kLong, {0, 1, 2});
            auto scores = options.score_function == "softmax" ? a[0].softmax(-1)
                : options.score_function == "sigmoid" ? a[0].sigmoid() : at::softplus(a[0]).sqrt();
            auto ids = options.hash_routing ? a[1] : std::get<1>((scores + a[1]).topk(k, -1));
            auto weights = scores.gather(1, ids.to(at::kLong));
            if (options.score_function != "softmax") weights /= weights.sum(-1, true);
            a[2].copy_(weights * options.route_scale);
            a[3].copy_(ids);
        } else if (operation == "compressor_pool") {
            count(3);
            check((a[0].dim() == 3 || a[0].dim() == 4) && a[0].sizes() == a[1].sizes()
                  && a[0].size(-2) > 0 && a[0].scalar_type() == at::kFloat && a[1].scalar_type() == at::kFloat,
                  "invalid compressor pooling inputs");
            auto shape = a[0].sizes().vec();
            if (a[0].dim() == 3) shape[1] = 1; else shape.erase(shape.end() - 2);
            output(2, shape, at::kFloat, {0, 1});
            a[2].copy_((a[0] * a[1].softmax(-2)).sum(-2, a[0].dim() == 3));
        } else if (operation == "expert_activation") {
            check(a.size() == 3 || a.size() == 4, "expert activation expects optional routing weights");
            check(a[0].dim() == 2 && a[0].sizes() == a[1].sizes() && a[0].scalar_type() == at::kBFloat16
                  && a[1].scalar_type() == at::kBFloat16, "invalid expert gate/up inputs");
            const size_t out = a.size() - 1;
            output(out, a[0].sizes().vec(), at::kBFloat16, {0, 1});
            if (a.size() == 4) {
                expect(a[2], {a[0].size(0), 1}, at::kFloat);
                check(!native_args[out]->overlaps(*native_args[2]), "activation output aliases routing weights");
            }
            auto gate = a[0].to(at::kFloat), up = a[1].to(at::kFloat);
            if (options.activation_limit > 0) {
                gate = gate.clamp_max(options.activation_limit);
                up = up.clamp(-options.activation_limit, options.activation_limit);
            }
            auto value = at::silu(gate) * up;
            if (a.size() == 4) value = a[2] * value;
            a[out].copy_(value.to(at::kBFloat16));
        } else if (operation == "moe_dispatch") {
            count(7);
            check(a[0].dim() == 2 && a[0].scalar_type() == at::kBFloat16 && a[2].dim() == 2 && integer(a[2])
                  && a[1].sizes() == a[2].sizes() && a[1].scalar_type() == at::kFloat
                  && a[0].size(0) == a[2].size(0) && a[6].dim() == 1, "invalid MoE dispatch inputs");
            const int64_t tokens = a[0].size(0), k = a[2].size(1), experts = a[6].size(0) - 1;
            check(experts > 0 && k > 0 && k <= experts && tokens <= INT64_MAX / k, "invalid dispatch expert/token counts");
            output(3, {tokens * k, a[0].size(1)}, at::kBFloat16, {0, 1, 2});
            output(4, {tokens * k, 1}, at::kFloat, {0, 1, 2, 3});
            output(5, {tokens * k}, at::kLong, {0, 1, 2, 3, 4});
            output(6, {experts + 1}, at::kLong, {0, 1, 2, 3, 4, 5});
            validIds(a[2], experts);
            auto flat = a[2].flatten().to(at::kLong);
            auto order = at::argsort(flat, true, 0, false);
            auto rows = at::floor_divide(order, k);
            auto counts = at::bincount(flat, {}, experts);
            check(counts.numel() == experts, "dispatch IDs exceed expert count");
            auto offsets = at::cat({at::zeros({1}, counts.options()), counts.cumsum(0)});
            a[3].copy_(a[0].index_select(0, rows));
            a[4].copy_(a[1].flatten().index_select(0, order).unsqueeze(1));
            a[5].copy_(rows); a[6].copy_(offsets);
        } else if (operation == "moe_combine") {
            count(4);
            check(a[0].dim() == 2 && a[0].scalar_type() == at::kBFloat16 && a[1].dim() == 1
                  && a[1].size(0) == a[0].size(0) && a[1].scalar_type() == at::kLong
                  && a[2].dim() == 1 && a[2].size(0) >= 2 && a[2].scalar_type() == at::kLong
                  && a[3].dim() == 2 && a[3].size(1) == a[0].size(1), "invalid MoE combine inputs");
            output(3, a[3].sizes().vec(), at::kFloat, {0, 1, 2});
            validIds(a[1], a[3].size(0));
            // Same explicit reference D2H synchronization and expert ordering
            // as Python offsets.tolist(). Not a device-only dispatch claim.
            auto host = a[2].cpu();
            const auto *offsets = host.const_data_ptr<int64_t>();
            check(offsets[0] == 0 && offsets[host.numel() - 1] == a[0].size(0), "invalid expert offsets endpoints");
            for (int64_t e = 1; e < host.numel(); ++e)
                check(offsets[e] >= offsets[e - 1] && offsets[e] <= a[0].size(0), "invalid expert offsets order");
            auto result = at::zeros(a[3].sizes(), a[3].options());
            for (int64_t e = 0; e + 1 < host.numel(); ++e) {
                if (offsets[e] == offsets[e + 1]) continue;
                auto rows = a[1].slice(0, offsets[e], offsets[e + 1]);
                result.index_put_({rows}, result.index({rows}) + a[0].slice(0, offsets[e], offsets[e + 1]));
            }
            a[3].copy_(result);
        } else throw std::invalid_argument("unsupported ATen structural operation");
    }
};

const char *StructuralKernel::compiledVersion() { return TORCH_VERSION; }

StructuralKernel::StructuralKernel(core::Runtime &runtime, native::KernelIdentity identity, StructuralOptions options)
    : native::Kernel(runtime, std::move(identity)), thread_(std::this_thread::get_id()) {
    static const std::set<std::string> supported = {"dense_linear", "grouped_linear", "row_inv_rms", "rms_norm", "hc_sum",
        "rotary_inplace", "indexer_scores", "indexer_topk", "router", "compressor_pool", "expert_activation", "moe_dispatch", "moe_combine",
        "tensor_cast", "row_gather", "moe_finalize", "tensor_fill", "compressor_prepare",
        "tensor_scale", "indexer_mask", "indexer_remap", "tensor_row_multiply", "attention_prepare",
        "hc_pre_reduce", "hc_post_mix", "hc_head_weights", "token_embedding", "greedy_argmax", "rotary_frequencies"};
    check(runtime.deviceType() == LLAISYS_DEVICE_NVIDIA && runtime.isActive(), "ATen native backend requires active NVIDIA runtime");
    check(identity_.backend == "aten-reference" && identity_.version == TORCH_VERSION && identity_.contract_revision == 1,
          "ATen backend/version/contract identity mismatch");
    check(supported.count(identity_.operation), "unknown ATen structural operation");
    check(std::isfinite(options.epsilon) && options.epsilon > 0 && std::isfinite(options.activation_limit)
          && std::isfinite(options.route_scale), "invalid structural scalar options");
    check(options.score_function == "softmax" || options.score_function == "sigmoid" || options.score_function == "sqrtsoftplus",
          "unsupported router score function");
    impl_ = std::make_unique<Impl>(identity_.operation, std::move(options));
}

void StructuralKernel::call(const std::vector<native::Tensor *> &arguments) {
    callWithScalars(arguments, {});
}

void StructuralKernel::callWithScalars(const std::vector<native::Tensor *> &arguments, const native::CallScalars &scalars) {
    if (!impl_ || std::this_thread::get_id() != thread_ || !runtime_.isActive())
        throw std::runtime_error("ATen kernel is closed or outside its owning runtime/thread");
    ++calls_;
    try {
        for (auto *tensor : arguments)
            check(tensor && &tensor->runtime() == &runtime_, "foreign or null ATen tensor");
        c10::InferenceMode inference(true);
        c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromExternal(
            reinterpret_cast<cudaStream_t>(runtime_.stream()), runtime_.deviceId()));
        impl_->run(arguments, scalars);
    } catch (...) { ++failures_; throw; }
}

void StructuralKernel::close() {
    if (!impl_) return;
    if (std::this_thread::get_id() != thread_ || !runtime_.isActive())
        throw std::runtime_error("close ATen kernel on its owning runtime/thread");
    runtime_.synchronize();
    impl_.reset();
}

StructuralKernel::~StructuralKernel() {
    try { close(); }
    catch (const std::exception &error) {
        std::fprintf(stderr, "llaisys ATen teardown failed: %s\n", error.what());
        (void)impl_.release();
    }
}

} // namespace llaisys::backends::aten
