#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

namespace friday {
namespace runtime {

class VramBudgetExceededException : public std::runtime_error {
public:
    explicit VramBudgetExceededException(const std::string& msg) : std::runtime_error(msg) {}
};

class GpuMemoryArena {
public:
    GpuMemoryArena(size_t total_vram_limit_bytes = 11811160064ULL); // 11.0 GB limit on 12GB GPU
    ~GpuMemoryArena();

    // Deterministic Sub-Arena Reservation
    void* allocate_weights_region(size_t bytes);
    void* allocate_activation_region(size_t bytes);
    void* allocate_kv_page_region(size_t bytes);
    void* allocate_scratch_region(size_t bytes);

    void reset_activations();
    void reset_scratch();

    size_t get_total_allocated() const;
    size_t get_vram_limit() const { return vram_limit_bytes_; }
    size_t get_free_budget() const;

    const VramBudgetProfile& get_profile() const { return profile_; }

    void print_memory_map() const;

private:
    size_t vram_limit_bytes_;
    VramBudgetProfile profile_;

    void* d_weights_ptr_ = nullptr;
    void* d_activations_ptr_ = nullptr;
    void* d_kv_pages_ptr_ = nullptr;
    void* d_scratch_ptr_ = nullptr;

    void check_budget(size_t additional_bytes);
};

} // namespace runtime
} // namespace friday
