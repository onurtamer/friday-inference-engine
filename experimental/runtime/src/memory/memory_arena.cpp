#include "friday/runtime/memory_arena.h"
#include <iomanip>

namespace friday {
namespace runtime {

GpuMemoryArena::GpuMemoryArena(size_t total_vram_limit_bytes)
    : vram_limit_bytes_(total_vram_limit_bytes)
{
    profile_.total_vram_bytes = total_vram_limit_bytes;
    std::cout << "[GpuMemoryArena] Baslatildi: Deterministik VRAM Limiti: "
              << std::fixed << std::setprecision(2)
              << (double)vram_limit_bytes_ / (1024.0 * 1024.0 * 1024.0) << " GB\n";
}

GpuMemoryArena::~GpuMemoryArena() {
    if (d_scratch_ptr_) cudaFree(d_scratch_ptr_);
    if (d_kv_pages_ptr_) cudaFree(d_kv_pages_ptr_);
    if (d_activations_ptr_) cudaFree(d_activations_ptr_);
    if (d_weights_ptr_) cudaFree(d_weights_ptr_);
}

void GpuMemoryArena::check_budget(size_t additional_bytes) {
    size_t cur = get_total_allocated();
    if (cur + additional_bytes > vram_limit_bytes_) {
        std::string err = "[GpuMemoryArena] HATA: VRAM Bütçesi Aşıldı! Mevcut: " +
            std::to_string(cur / (1024 * 1024)) + " MB, Talep: " +
            std::to_string(additional_bytes / (1024 * 1024)) + " MB, Limit: " +
            std::to_string(vram_limit_bytes_ / (1024 * 1024)) + " MB.";
        throw VramBudgetExceededException(err);
    }
}

void* GpuMemoryArena::allocate_weights_region(size_t bytes) {
    check_budget(bytes);
    if (d_weights_ptr_) cudaFree(d_weights_ptr_);
    cudaError_t err = cudaMalloc(&d_weights_ptr_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("[GpuMemoryArena] cudaMalloc failed for weights region!");
    }
    profile_.weights_bytes = bytes;
    return d_weights_ptr_;
}

void* GpuMemoryArena::allocate_activation_region(size_t bytes) {
    check_budget(bytes);
    if (d_activations_ptr_) cudaFree(d_activations_ptr_);
    cudaError_t err = cudaMalloc(&d_activations_ptr_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("[GpuMemoryArena] cudaMalloc failed for activation region!");
    }
    profile_.activation_arena_bytes = bytes;
    return d_activations_ptr_;
}

void* GpuMemoryArena::allocate_kv_page_region(size_t bytes) {
    check_budget(bytes);
    if (d_kv_pages_ptr_) cudaFree(d_kv_pages_ptr_);
    cudaError_t err = cudaMalloc(&d_kv_pages_ptr_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("[GpuMemoryArena] cudaMalloc failed for KV page region!");
    }
    profile_.kv_page_arena_bytes = bytes;
    return d_kv_pages_ptr_;
}

void* GpuMemoryArena::allocate_scratch_region(size_t bytes) {
    check_budget(bytes);
    if (d_scratch_ptr_) cudaFree(d_scratch_ptr_);
    cudaError_t err = cudaMalloc(&d_scratch_ptr_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("[GpuMemoryArena] cudaMalloc failed for scratch region!");
    }
    profile_.scratch_workspace_bytes = bytes;
    return d_scratch_ptr_;
}

void GpuMemoryArena::reset_activations() {
    if (d_activations_ptr_ && profile_.activation_arena_bytes > 0) {
        cudaMemset(d_activations_ptr_, 0, profile_.activation_arena_bytes);
    }
}

void GpuMemoryArena::reset_scratch() {
    if (d_scratch_ptr_ && profile_.scratch_workspace_bytes > 0) {
        cudaMemset(d_scratch_ptr_, 0, profile_.scratch_workspace_bytes);
    }
}

size_t GpuMemoryArena::get_total_allocated() const {
    return profile_.weights_bytes + profile_.activation_arena_bytes + 
           profile_.kv_page_arena_bytes + profile_.scratch_workspace_bytes;
}

size_t GpuMemoryArena::get_free_budget() const {
    size_t used = get_total_allocated();
    return (used < vram_limit_bytes_) ? (vram_limit_bytes_ - used) : 0;
}

void GpuMemoryArena::print_memory_map() const {
    std::cout << "========================================================\n";
    std::cout << "  GPU MEMORY ARENA MAP (DETERMINISTIC VRAM RESIDENCY)\n";
    std::cout << "========================================================\n";
    std::cout << "Weights Arena:        " << std::fixed << std::setprecision(2)
              << profile_.weights_gb() << " GB\n";
    std::cout << "Activation Arena:     " 
              << (double)profile_.activation_arena_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "KV Page Arena:        " 
              << profile_.kv_arena_gb() << " GB\n";
    std::cout << "Scratch Workspace:    " 
              << (double)profile_.scratch_workspace_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Total Allocated:      " 
              << profile_.total_used_gb() << " GB / "
              << (double)vram_limit_bytes_ / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    std::cout << "Free Budget Margin:   " 
              << (double)get_free_budget() / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    std::cout << "========================================================\n";
}

} // namespace runtime
} // namespace friday
