#include "block_allocator.hpp"

#include <stdexcept>
#include <cstdio>

namespace llaisys::core {

BlockAllocator::BlockAllocator(const BlockAllocatorConfig &config, const LlaisysRuntimeAPI *api)
    : _pool_K(nullptr), _pool_V(nullptr), _api(api), _config(config) {
    if (!_api)
        throw std::runtime_error("BlockAllocator: RuntimeAPI is null");
    if (_config.num_blocks == 0)
        throw std::runtime_error("BlockAllocator: num_blocks must be > 0");
        
    //stride 公式
    _layer_stride = _config.block_size * _config.nkvh * _config.dh * _config.elem_size; //一层里一个 block 的字节数
    _block_stride = _config.nlayer * _layer_stride; //一个完整 block 跨所有层的字节数

    size_t total_bytes = _config.num_blocks * _block_stride; //整个 pool 的总字节数

    _pool_K = _api->malloc_device(total_bytes);
    _pool_V = _api->malloc_device(total_bytes);

    if (!_pool_K || !_pool_V) {
        if (_pool_K) _api->free_device(_pool_K);
        if (_pool_V) _api->free_device(_pool_V);
        _pool_K = _pool_V = nullptr;
        throw std::runtime_error("BlockAllocator: failed to allocate memory pool");
    }

    for (size_t i = 0; i < _config.num_blocks; ++i) {
        _free_list.push_back(static_cast<int>(i));
    }
}

BlockAllocator::~BlockAllocator() {
    if (_pool_K) _api->free_device(_pool_K);
    if (_pool_V) _api->free_device(_pool_V);
    _pool_K = _pool_V = nullptr;
}

int BlockAllocator::alloc() {
    if (_free_list.empty()) return -1;
    int block_id = _free_list.front();
    _free_list.pop_front();
    return block_id;
}

void BlockAllocator::free(int block_id) {
    if (block_id < 0 || block_id >= static_cast<int>(_config.num_blocks)) return;
    _free_list.push_back(block_id);
}

void *BlockAllocator::get_k_ptr(int block_id, int layer) {
    return static_cast<std::byte *>(_pool_K) +
           static_cast<size_t>(block_id) * _block_stride +
           static_cast<size_t>(layer) * _layer_stride;
    /*先从 K pool 起点开始
    跳过 block_id 个完整 block
    再跳到这个 block 内的第 layer 层
    得到该 (block_id, layer) 的 K 区域首地址
    */
}

void *BlockAllocator::get_v_ptr(int block_id, int layer) {
    return static_cast<std::byte *>(_pool_V) +
           static_cast<size_t>(block_id) * _block_stride +
           static_cast<size_t>(layer) * _layer_stride;
}

size_t BlockAllocator::num_free() const {
    return _free_list.size();
}

size_t BlockAllocator::num_total() const {
    return _config.num_blocks;
}

} // namespace llaisys::core
