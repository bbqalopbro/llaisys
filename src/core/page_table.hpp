#pragma once

#include "allocator/block_allocator.hpp"

#include <vector>
#include <cstddef>

namespace llaisys::core {

// Per-sequence page table mapping token positions to block IDs.
// Works with BlockAllocator to provide paged KV-Cache addressing.
class PageTable {
private:
    std::vector<int> _block_ids;
    int _num_tokens;
    int _block_size;

public:
    explicit PageTable(int block_size = 16)
        : _num_tokens(0), _block_size(block_size) {}

    void append_block(int block_id) { //把新申请到的物理块号记进_block_ids
        _block_ids.push_back(block_id);
    }

    void attach_blocks(const std::vector<int> &block_ids, int num_tokens) {
        _block_ids = block_ids;
        _num_tokens = num_tokens;
    }

    int get_block_for_token(int token_pos) const {
        int block_idx = token_pos / _block_size; //这个 token 属于第几个逻辑块
        if (block_idx < 0 || block_idx >= static_cast<int>(_block_ids.size()))
            return -1;
        return _block_ids[block_idx];
    }

    int get_offset_in_block(int token_pos) const {
        return token_pos % _block_size; //它在逻辑块块内第几个位置
    }

    void release_all(BlockAllocator &allocator) {
        for (int bid : _block_ids) {
            allocator.free(bid);
        }
        _block_ids.clear();
        _num_tokens = 0;
    }

    int num_tokens() const { return _num_tokens; }
    void set_num_tokens(int n) { _num_tokens = n; }
    void inc_num_tokens() { ++_num_tokens; }

    int num_blocks() const { return static_cast<int>(_block_ids.size()); }
    const std::vector<int> &block_ids() const { return _block_ids; }
    int block_size() const { return _block_size; }

    bool needs_new_block() const {
        if (_block_ids.empty()) return true;
        return (_num_tokens % _block_size == 0);
    }

    void clear() {
        _block_ids.clear();
        _num_tokens = 0;
    }
};

} // namespace llaisys::core
