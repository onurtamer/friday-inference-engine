#pragma once

#include "types.h"
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace friday {
namespace runtime {

constexpr int KV_BLOCK_SIZE = 16; // 16 tokens per physical page

struct PhysicalKvBlock {
    BlockId block_id = -1;
    int ref_count = 0; // Reference count for Copy-On-Write
    void* d_k_ptr = nullptr; // Device pointer to GPU VRAM for Key block
    void* d_v_ptr = nullptr; // Device pointer to GPU VRAM for Value block
};

class PagedKvCache {
public:
    PagedKvCache(int num_layers, int num_kv_heads, int head_dim, size_t max_blocks, QuantType quant = QuantType::FP16);
    ~PagedKvCache();

    // Context Lifecycle
    ContextId create_context();
    void destroy_context(ContextId ctx_id);

    // Invariant 4: Non-Negotiable Copy-On-Write (COW) Fork Context
    // Shares identical physical pages in O(1) time; ref_count incremented
    ContextId fork_context(ContextId parent_ctx);

    // Append token slot and resolve COW if page is shared upon write
    void prepare_write_slot(ContextId ctx_id, int seq_pos, cudaStream_t stream = 0);

    // Get block table for attention kernel execution
    const std::vector<BlockId>& get_block_table(ContextId ctx_id) const;

    // Get block physical address
    const PhysicalKvBlock& get_physical_block(BlockId block_id) const;

    size_t get_total_blocks() const { return total_blocks_; }
    size_t get_free_blocks() const { return free_blocks_.size(); }
    size_t get_allocated_contexts() const { return page_tables_.size(); }
    size_t get_block_bytes() const { return block_bytes_per_component_ * 2; }

private:
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    size_t total_blocks_;
    QuantType quant_type_;
    size_t block_bytes_per_component_; // Bytes per K block across all layers

    // Global Pre-allocated GPU VRAM Arenas
    void* d_k_pool_ = nullptr;
    void* d_v_pool_ = nullptr;

    std::vector<PhysicalKvBlock> physical_blocks_;
    std::queue<BlockId> free_blocks_;

    // ContextId -> Logical Page Table (BlockId list)
    std::unordered_map<ContextId, std::vector<BlockId>> page_tables_;
    ContextId next_context_id_ = 1;

    BlockId allocate_free_block();
    void release_block(BlockId block_id);
};

} // namespace runtime
} // namespace friday
