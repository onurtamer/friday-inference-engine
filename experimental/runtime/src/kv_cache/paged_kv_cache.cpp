#include "friday/runtime/paged_kv_cache.h"
#include <iostream>
#include <cstring>

namespace friday {
namespace runtime {

PagedKvCache::PagedKvCache(int num_layers, int num_kv_heads, int head_dim, size_t max_blocks, QuantType quant)
    : num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      total_blocks_(max_blocks),
      quant_type_(quant)
{
    // Bytes for 1 component (Key or Value) in a single block across ALL layers:
    // block_tokens * num_layers * num_kv_heads * head_dim * sizeof(half)
    size_t element_size = (quant == QuantType::INT4) ? 1 : sizeof(half);
    block_bytes_per_component_ = (size_t)KV_BLOCK_SIZE * num_layers_ * num_kv_heads_ * head_dim_ * element_size;

    size_t total_k_bytes = total_blocks_ * block_bytes_per_component_;
    size_t total_v_bytes = total_blocks_ * block_bytes_per_component_;

    cudaError_t err_k = cudaMalloc(&d_k_pool_, total_k_bytes);
    cudaError_t err_v = cudaMalloc(&d_v_pool_, total_v_bytes);
    if (err_k != cudaSuccess || err_v != cudaSuccess) {
        throw std::runtime_error("[PagedKvCache] GPU VRAM allocation failed for KV pools!");
    }

    cudaMemset(d_k_pool_, 0, total_k_bytes);
    cudaMemset(d_v_pool_, 0, total_v_bytes);

    physical_blocks_.resize(total_blocks_);
    for (size_t i = 0; i < total_blocks_; ++i) {
        physical_blocks_[i].block_id = (BlockId)i;
        physical_blocks_[i].ref_count = 0;
        physical_blocks_[i].d_k_ptr = (char*)d_k_pool_ + i * block_bytes_per_component_;
        physical_blocks_[i].d_v_ptr = (char*)d_v_pool_ + i * block_bytes_per_component_;
        free_blocks_.push((BlockId)i);
    }

    std::cout << "[PagedKvCache] Baslatildi: " << total_blocks_ << " fiziksel blok ("
              << (double)(total_k_bytes + total_v_bytes) / (1024.0 * 1024.0) << " MB VRAM, "
              << total_blocks_ * KV_BLOCK_SIZE << " token kapasitesi).\n";
}

PagedKvCache::~PagedKvCache() {
    if (d_k_pool_) cudaFree(d_k_pool_);
    if (d_v_pool_) cudaFree(d_v_pool_);
}

ContextId PagedKvCache::create_context() {
    ContextId id = next_context_id_++;
    page_tables_[id] = std::vector<BlockId>();
    return id;
}

void PagedKvCache::destroy_context(ContextId ctx_id) {
    auto it = page_tables_.find(ctx_id);
    if (it == page_tables_.end()) return;

    for (BlockId b : it->second) {
        release_block(b);
    }
    page_tables_.erase(it);
}

// Invariant 4: Non-Negotiable Copy-on-Write (COW) Fork Context
// Zero-copy branching for Dialectical Choir (Architect, Critic, Alternative)
ContextId PagedKvCache::fork_context(ContextId parent_ctx) {
    auto it = page_tables_.find(parent_ctx);
    if (it == page_tables_.end()) {
        throw std::runtime_error("[PagedKvCache] fork_context: Parent context bulunamadi!");
    }

    ContextId child_id = next_context_id_++;
    // 1. Copy logical page table mapping (O(1) pointer copies, NO memory copies)
    page_tables_[child_id] = it->second;

    // 2. Increment reference count for all shared physical blocks
    for (BlockId b : it->second) {
        physical_blocks_[b].ref_count++;
    }

    return child_id;
}

void PagedKvCache::prepare_write_slot(ContextId ctx_id, int seq_pos, cudaStream_t stream) {
    auto it = page_tables_.find(ctx_id);
    if (it == page_tables_.end()) {
        throw std::runtime_error("[PagedKvCache] prepare_write_slot: Context bulunamadi!");
    }

    size_t logical_block_idx = (size_t)seq_pos / KV_BLOCK_SIZE;
    size_t in_block_offset = (size_t)seq_pos % KV_BLOCK_SIZE;

    if (logical_block_idx == it->second.size()) {
        // Yeni bir fiziksel sayfa gereklidir
        BlockId new_b = allocate_free_block();
        physical_blocks_[new_b].ref_count = 1;
        it->second.push_back(new_b);
    } else if (logical_block_idx < it->second.size()) {
        // Mevcut sayfa uzerine yaziliyor; paylasim (COW) kontrolu yap
        BlockId cur_b = it->second[logical_block_idx];
        if (physical_blocks_[cur_b].ref_count > 1) {
            // >>> COPY-ON-WRITE TETIKLENDI <<<
            // Yeni fiziksel blok tahsis et
            BlockId new_b = allocate_free_block();
            physical_blocks_[new_b].ref_count = 1;

            // Onceki gecerli tokenlari (in_block_offset kadar) GPU uzerinde klonla
            if (in_block_offset > 0) {
                size_t element_size = (quant_type_ == QuantType::INT4) ? 1 : sizeof(half);
                size_t copy_bytes = in_block_offset * num_layers_ * num_kv_heads_ * head_dim_ * element_size;
                cudaMemcpyAsync(physical_blocks_[new_b].d_k_ptr, physical_blocks_[cur_b].d_k_ptr, copy_bytes, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(physical_blocks_[new_b].d_v_ptr, physical_blocks_[cur_b].d_v_ptr, copy_bytes, cudaMemcpyDeviceToDevice, stream);
            }

            // Eski blogun referans sayisini dusur
            release_block(cur_b);

            // Child baglamin mantiksal tablosunu yeni bloga guncelle
            it->second[logical_block_idx] = new_b;
        }
    } else {
        throw std::runtime_error("[PagedKvCache] prepare_write_slot: Sirasiz (out-of-order) token pos!");
    }
}

const std::vector<BlockId>& PagedKvCache::get_block_table(ContextId ctx_id) const {
    auto it = page_tables_.find(ctx_id);
    if (it == page_tables_.end()) {
        throw std::runtime_error("[PagedKvCache] get_block_table: Context bulunamadi!");
    }
    return it->second;
}

const PhysicalKvBlock& PagedKvCache::get_physical_block(BlockId block_id) const {
    if (block_id < 0 || (size_t)block_id >= physical_blocks_.size()) {
        throw std::out_of_range("[PagedKvCache] Gecersiz block_id!");
    }
    return physical_blocks_[block_id];
}

BlockId PagedKvCache::allocate_free_block() {
    if (free_blocks_.empty()) {
        throw std::runtime_error("[PagedKvCache] OOM: Serbest KV bloku kalmadi! VRAM butcesi asildi.");
    }
    BlockId b = free_blocks_.front();
    free_blocks_.pop();
    return b;
}

void PagedKvCache::release_block(BlockId block_id) {
    if (block_id < 0 || (size_t)block_id >= physical_blocks_.size()) return;
    
    physical_blocks_[block_id].ref_count--;
    if (physical_blocks_[block_id].ref_count <= 0) {
        physical_blocks_[block_id].ref_count = 0;
        free_blocks_.push(block_id);
    }
}

} // namespace runtime
} // namespace friday
