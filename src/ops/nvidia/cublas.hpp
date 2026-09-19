#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <map>
#include <memory>
#include <array>
#include <stdexcept>
namespace llaisys::ops::nvidia {
inline void vendor_cuda(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
inline void vendor_blas(cublasStatus_t e) { if(e!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("framework cuBLAS status "+std::to_string(int(e))); }
inline void vendor_outside_capture() {
    cudaStreamCaptureStatus capture; vendor_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture));
    if(capture!=cudaStreamCaptureStatusNone) throw std::runtime_error("warm up framework cuBLAS shape before Graph capture");
}
struct VendorState {
    cublasHandle_t handle=nullptr; void *workspace=nullptr; int device;
    std::array<std::map<size_t,void *>,8> scratch{};
    explicit VendorState(int d):device(d) {
        vendor_outside_capture();
        vendor_cuda(cudaMalloc(&workspace,4*1024*1024));
        vendor_blas(cublasCreate(&handle));
        vendor_blas(cublasSetStream(handle,cudaStreamPerThread));
        vendor_blas(cublasSetWorkspace(handle,workspace,4*1024*1024));
    }
    ~VendorState() {
        int old; if(cudaGetDevice(&old)!=cudaSuccess) return;
        cudaSetDevice(device); cublasDestroy(handle); cudaFree(workspace);
        for(auto &slot:scratch) for(auto &b:slot) cudaFree(b.second);
        cudaSetDevice(old);
    }
};
inline VendorState &vendor_state() {
    static thread_local std::map<int,std::unique_ptr<VendorState>> states;
    int device; vendor_cuda(cudaGetDevice(&device));
    auto &s=states[device]; if(!s) s=std::make_unique<VendorState>(device); return *s;
}
inline cublasHandle_t get_cublas_handle() { return vendor_state().handle; }
template<class T> T *vendor_buffer(int slot,size_t elements) {
    auto &cache=vendor_state().scratch.at(slot); size_t bytes=elements*sizeof(T);
    auto found=cache.find(bytes);
    if(found==cache.end()) {
        vendor_outside_capture();
        if(cache.size()>=128) throw std::runtime_error("cuBLAS workspace shape cache full");
        void *data=nullptr; vendor_cuda(cudaMalloc(&data,bytes));
        found=cache.emplace(bytes,data).first;
    }
    // Previous Graphs retain their own stable scratch pointers when another shape is prepared.
    return static_cast<T*>(found->second);
}
}
