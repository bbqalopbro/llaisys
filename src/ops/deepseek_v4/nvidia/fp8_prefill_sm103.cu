// Build: nvcc -std=c++17 -O3 -arch=sm_103 -DLLAISYS_B300_STANDALONE -shared -Xcompiler=-fPIC
//   fp8_prefill_sm103.cu -lcublasLt -o libv4_fp8_prefill_sm103.so
// CUDA 13 documentation, sections 3.1.4 and 3.1.4.3.2:
// https://docs.nvidia.com/cuda/archive/13.0.3/cublas/index.html
// VEC128_32F/BLK128x128_32F support Hopper (CC 9.0), not B300.
// B300 uses native MXFP8 VEC32_UE8M0. Replicating exact UE8M0 metadata
// preserves the DeepSeek per-128 dequantization without changing FP8 payload.
#ifdef LLAISYS_B300_STANDALONE
#include "fp8_prefill_sm103.h"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_message;
void cuda_check(cudaError_t s, const char *op) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s));
}
void blas_check(cublasStatus_t s, const char *op) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(op) + ": cuBLASLt status " + std::to_string(int(s)));
}
#define CUDA_OK(x) cuda_check((x), #x)
#define BLAS_OK(x) blas_check((x), #x)

// Each 128-row x 4-scale tile occupies 512 bytes. The four adjacent K32
// scales equal the original K128 scale. Threads write adjacent uint32 words:
// word = row_mod_32 * 4 + row_div_32, giving contiguous, coalesced stores.
template <bool Weight>
__global__ void expand_scales(const uint8_t *src, uint32_t *dst,
                              int rows, int k128, int outer_tiles) {
    const int64_t total_words = int64_t(outer_tiles) * k128 * 128;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total_words;
         i += int64_t(blockDim.x) * gridDim.x) {
        const int lane = i % 128;
        const int tile = i / 128;
        const int row = (tile / k128) * 128 + (lane % 4) * 32 + lane / 4;
        const int group = tile % k128;
        const uint32_t x = row < rows ? src[(Weight ? row / 128 : row) * k128 + group] : 0;
        dst[i] = x * 0x01010101u;
    }
}

struct Plan {
    int m=0, n=0, k=0, padded_m=0, device=0, selected=0;
    bool a_ready=false, b_ready=false;
    cudaStream_t stream=nullptr;
    cublasLtHandle_t handle=nullptr;
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t left=nullptr, right=nullptr, output=nullptr;
    void *a_scales=nullptr, *b_scales=nullptr, *workspace=nullptr;
    void *padded_a=nullptr, *padded_output=nullptr;
    size_t workspace_bytes=0;
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms;
    ~Plan() {
        // Public destroy first synchronizes and reports errors. Also synchronize
        // here for partial-create rollback before releasing any owned buffers.
        if (handle) cudaStreamSynchronize(stream);
        if (output) cublasLtMatrixLayoutDestroy(output);
        if (right) cublasLtMatrixLayoutDestroy(right);
        if (left) cublasLtMatrixLayoutDestroy(left);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (handle) cublasLtDestroy(handle);
        if (padded_output) cudaFree(padded_output);
        if (padded_a) cudaFree(padded_a);
        if (workspace) cudaFree(workspace);
        if (b_scales) cudaFree(b_scales);
        if (a_scales) cudaFree(a_scales);
    }
};
Plan &checked(void *ptr) {
    if (!ptr) throw std::runtime_error("null plan");
    auto &p = *static_cast<Plan *>(ptr);
    int current; CUDA_OK(cudaGetDevice(&current));
    if (current != p.device) throw std::runtime_error("plan used on a different CUDA device");
    return p;
}
void outside_capture(Plan &p) {
    cudaStreamCaptureStatus status;
    CUDA_OK(cudaStreamIsCapturing(p.stream, &status));
    if (status != cudaStreamCaptureStatusNone)
        throw std::runtime_error("initialization operation during CUDA graph capture");
}
void aligned(const void *p) {
    if (!p || reinterpret_cast<uintptr_t>(p) % 16)
        throw std::runtime_error("FP8/BF16 matrix pointers must be non-null and 16-byte aligned");
}
void disjoint(const void *a,size_t as,const void *b,size_t bs) {
    const uintptr_t x=reinterpret_cast<uintptr_t>(a), y=reinterpret_cast<uintptr_t>(b);
    if (x>UINTPTR_MAX-as || y>UINTPTR_MAX-bs || (x<y+bs && y<x+as))
        throw std::runtime_error("matrix/scale buffers partially overlap or address range overflows");
}
void matrix_ranges(Plan &p,const void *a,const void *b,void *out) {
    aligned(a);aligned(b);aligned(out);
    disjoint(a,size_t(p.m)*p.k,b,size_t(p.n)*p.k);
    disjoint(a,size_t(p.m)*p.k,out,size_t(p.m)*p.n*2);
    disjoint(b,size_t(p.n)*p.k,out,size_t(p.m)*p.n*2);
}
void prepare(Plan &p, const void *scales, bool weight) {
    if (!scales) throw std::runtime_error("null scale pointer");
    const int rows=weight?p.n:p.m, tiles=(rows+127)/128;
    const int blocks=std::min(4096, (tiles*(p.k/128)*128+255)/256);
    if (weight)
        expand_scales<true><<<blocks,256,0,p.stream>>>(static_cast<const uint8_t *>(scales),
            static_cast<uint32_t *>(p.b_scales),rows,p.k/128,tiles);
    else
        expand_scales<false><<<blocks,256,0,p.stream>>>(static_cast<const uint8_t *>(scales),
            static_cast<uint32_t *>(p.a_scales),rows,p.k/128,tiles);
    CUDA_OK(cudaGetLastError());
    if (weight) p.b_ready=true; else p.a_ready=true;
}
void multiply(Plan &p, const void *a, const void *b, void *out) {
    matrix_ranges(p,a,b,out);
    if (!p.a_ready || !p.b_ready) throw std::runtime_error("prepare both scale tensors before GEMM");
    const void *input=a;
    void *result=out;
    if (p.padded_a) {
        CUDA_OK(cudaMemcpyAsync(p.padded_a,a,size_t(p.m)*p.k,cudaMemcpyDeviceToDevice,p.stream));
        CUDA_OK(cudaMemsetAsync(static_cast<uint8_t *>(p.padded_a)+size_t(p.m)*p.k,
                               0,size_t(p.padded_m-p.m)*p.k,p.stream));
        input=p.padded_a; result=p.padded_output;
    }
    const float alpha=1.0f,beta=0.0f;
    // Column-major C^T[N,M] = B[N,K] * A^T[K,M]. Original row-major
    // payload is consumed directly as [K,N]/[K,M] column-major storage.
    BLAS_OK(cublasLtMatmul(p.handle,p.operation,&alpha,b,p.left,input,p.right,
        &beta,result,p.output,result,p.output,&p.algorithms[p.selected].algo,
        p.workspace,p.workspace_bytes,p.stream));
    if (p.padded_output)
        CUDA_OK(cudaMemcpyAsync(out,p.padded_output,size_t(p.m)*p.n*2,cudaMemcpyDeviceToDevice,p.stream));
}
template<class Fn> int boundary(Fn fn) {
    try { fn(); error_message.clear(); return 0; }
    catch(const std::exception &e) { error_message=e.what(); return 1; }
    catch(...) { error_message="unknown C++ exception"; return 1; }
}
}

extern "C" int llaisys_v4_fp8_create(int m,int n,int k,uintptr_t stream,
                                     size_t workspace_bytes,void **out_plan) {
    if (out_plan) *out_plan=nullptr;
    return boundary([&] {
        if (!out_plan || m<=0 || n<=0 || k<=0 || n%128 || k%128 ||
            m>1048576 || n>1048576 || k>1048576 ||
            int64_t((m+127)/128)*(k/128)*128>2147483647 ||
            int64_t(n/128)*(k/128)*128>2147483647)
            throw std::runtime_error("require positive M and N,K multiples of 128, each dimension <= 1048576");
        auto p=std::make_unique<Plan>();
        p->m=m;p->n=n;p->k=k;p->padded_m=(m+15)/16*16;
        p->stream=reinterpret_cast<cudaStream_t>(stream);
        CUDA_OK(cudaGetDevice(&p->device));
        cudaDeviceProp properties; CUDA_OK(cudaGetDeviceProperties(&properties,p->device));
        if (properties.major!=10 || properties.minor!=3)
            throw std::runtime_error("this candidate is validated only for B300 CC 10.3");
        outside_capture(*p);
        BLAS_OK(cublasLtCreate(&p->handle));
        CUDA_OK(cudaMalloc(&p->a_scales,size_t((m+127)/128)*(k/128)*512));
        CUDA_OK(cudaMalloc(&p->b_scales,size_t(n/128)*(k/128)*512));
        p->workspace_bytes=workspace_bytes;
        if (workspace_bytes) CUDA_OK(cudaMalloc(&p->workspace,workspace_bytes));
        if (p->padded_m!=m) {
            CUDA_OK(cudaMalloc(&p->padded_a,size_t(p->padded_m)*k));
            CUDA_OK(cudaMalloc(&p->padded_output,size_t(p->padded_m)*n*2));
        }
        BLAS_OK(cublasLtMatmulDescCreate(&p->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        cublasOperation_t trans=CUBLAS_OP_T;
        BLAS_OK(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
        trans=CUBLAS_OP_N;
        BLAS_OK(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_TRANSB,&trans,sizeof(trans)));
        auto mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        BLAS_OK(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
        BLAS_OK(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
        BLAS_OK(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&p->b_scales,sizeof(void *)));
        BLAS_OK(cublasLtMatmulDescSetAttribute(p->operation,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&p->a_scales,sizeof(void *)));
        BLAS_OK(cublasLtMatrixLayoutCreate(&p->left,CUDA_R_8F_E4M3,k,n,k));
        BLAS_OK(cublasLtMatrixLayoutCreate(&p->right,CUDA_R_8F_E4M3,k,p->padded_m,k));
        BLAS_OK(cublasLtMatrixLayoutCreate(&p->output,CUDA_R_16BF,n,p->padded_m,n));
        cublasLtMatmulPreference_t pref=nullptr;
        BLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
        try {
            BLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &workspace_bytes,sizeof(workspace_bytes)));
            const uint32_t alignment=16;
            for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                              CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
                BLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,attr,&alignment,sizeof(alignment)));
            cublasLtMatmulHeuristicResult_t candidates[32]; int count=0;
            BLAS_OK(cublasLtMatmulAlgoGetHeuristic(p->handle,p->operation,p->left,p->right,p->output,p->output,
                                                  pref,32,candidates,&count));
            for (int i=0;i<count;++i) if(candidates[i].state==CUBLAS_STATUS_SUCCESS)
                p->algorithms.push_back(candidates[i]);
        } catch(...) { cublasLtMatmulPreferenceDestroy(pref);throw; }
        BLAS_OK(cublasLtMatmulPreferenceDestroy(pref));
        if(p->algorithms.empty()) throw std::runtime_error("cuBLASLt returned no MXFP8 algorithm for this shape");
        *out_plan=p.release();
    });
}
extern "C" int llaisys_v4_fp8_prepare_b(void *p,const void *s) {
    return boundary([&]{prepare(checked(p),s,true);});
}
extern "C" int llaisys_v4_fp8_prepare_a(void *p,const void *s) {
    return boundary([&]{prepare(checked(p),s,false);});
}
extern "C" int llaisys_v4_fp8_gemm(void *p,const void *a,const void *b,void *out) {
    return boundary([&]{multiply(checked(p),a,b,out);});
}
extern "C" int llaisys_v4_fp8_run(void *p,const void *a,const void *s,const void *b,void *out) {
    return boundary([&]{auto &plan=checked(p);matrix_ranges(plan,a,b,out);
        if(!s)throw std::runtime_error("null A scale pointer");
        const size_t bytes=size_t(plan.m)*(plan.k/128);
        disjoint(s,bytes,a,size_t(plan.m)*plan.k);
        disjoint(s,bytes,b,size_t(plan.n)*plan.k);
        disjoint(s,bytes,out,size_t(plan.m)*plan.n*2);
        prepare(plan,s,false);multiply(plan,a,b,out);});
}
extern "C" int llaisys_v4_fp8_algorithm_count(void *p) {
    try{return int(checked(p).algorithms.size());}
    catch(const std::exception &e){error_message=e.what();return -1;}
}
extern "C" int llaisys_v4_fp8_select_algorithm(void *p,int index) {
    return boundary([&]{auto &plan=checked(p);outside_capture(plan);
        if(index<0 || index>=int(plan.algorithms.size())) throw std::runtime_error("invalid algorithm index");
        plan.selected=index;});
}
extern "C" int llaisys_v4_fp8_destroy(void *p) {
    if(!p)return 0;
    const int validation=boundary([&]{outside_capture(checked(p));});
    if(validation)return validation;
    auto *plan=static_cast<Plan *>(p);
    const cudaError_t status=cudaStreamSynchronize(plan->stream);
    delete plan;
    if(status!=cudaSuccess) {
        error_message=std::string("destroy consumed plan after stream error: ")+cudaGetErrorString(status);
        return 2;
    }
    error_message.clear();return 0;
}
extern "C" const char *llaisys_v4_fp8_last_error(void){return error_message.c_str();}
#endif // LLAISYS_B300_STANDALONE: default repository CUDA globs compile an empty TU.
