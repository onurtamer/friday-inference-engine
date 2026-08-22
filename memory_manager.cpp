// memory_manager.cpp
#include "memory_manager.h"
#include "friday_types.h" // IO_ALIGNMENT için
#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h> // GetSystemInfo için
#else // Linux
#include <unistd.h> // sysconf(_SC_PAGESIZE) için
#include <errno.h>  // strerror için
#include <cstring>  // strerror
#endif


// Statik yardımcı fonksiyon: Sistem sayfa boyutunu bir kez alır.
static size_t get_system_page_size() {
    static size_t page_size = 0;
    if (page_size == 0) {
#ifdef _WIN32
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        page_size = sysInfo.dwPageSize; // Windows'un gerçek sayfa boyutunu al
#else // Linux
        long ps = sysconf(_SC_PAGESIZE);
        if (ps == -1) {
            fprintf(stderr, "WARNING: Could not get system page size, defaulting to 4096. Error: %s\n", strerror(errno));
            page_size = 4096; // Fallback
        } else {
            page_size = static_cast<size_t>(ps);
        }
#endif
        if (page_size != IO_ALIGNMENT) {
            fprintf(stderr, "WARNING: System page size (%zu) differs from FRIDAY_TYPES_H::IO_ALIGNMENT (%zu). Ensure O_DIRECT compatibility.\n", page_size, IO_ALIGNMENT);
        }
    }
    return page_size;
}

ZeroCopyMemoryManager::ZeroCopyMemoryManager() {
    get_system_page_size();
    fprintf(stdout, "ZeroCopyMemoryManager initialized. System page size: %zu bytes.\n", get_system_page_size());
}

ZeroCopyMemoryManager::~ZeroCopyMemoryManager() {
    std::lock_guard<std::mutex> lock(this->_mutex);
    for (auto const& [ptr, size] : this->_allocated_blocks) {
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFreeHost(ptr));
            fprintf(stdout, "Deallocated pinned memory block at %p, size %zu bytes.\n", ptr, size);
        }
    }
    this->_allocated_blocks.clear();
    fprintf(stdout, "ZeroCopyMemoryManager destroyed, all pinned memory freed.\n");
}

void* ZeroCopyMemoryManager::allocate_pinned(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    size_t page_size = get_system_page_size();
    if (size % page_size != 0) {
        throw std::invalid_argument(
            "ZeroCopyMemoryManager::allocate_pinned: Requested size must be a multiple of system page size " +
            std::to_string(page_size) + " bytes. Requested: " + std::to_string(size)
        );
    }

    void* ptr = nullptr;
    std::lock_guard<std::mutex> lock(this->_mutex);
    CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));

    if (!is_aligned(ptr, page_size)) {
        fprintf(stderr, "WARNING: cudaHostAlloc returned non-page-aligned memory at %p for size %zu. System page size: %zu.\n", ptr, size, page_size);
    }

    this->_allocated_blocks[ptr] = size;
    fprintf(stdout, "Allocated pinned memory at %p, size %zu bytes. Page-aligned: %s\n", ptr, size, is_aligned(ptr, page_size) ? "YES" : "NO");
    return ptr;
}

void ZeroCopyMemoryManager::deallocate_pinned(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(this->_mutex);
    auto it = this->_allocated_blocks.find(ptr);
    if (it == this->_allocated_blocks.end()) {
        fprintf(stderr, "WARNING: Attempted to deallocate unknown pinned memory pointer %p.\n", ptr);
        return;
    }

    CUDA_CHECK(cudaFreeHost(ptr));
    fprintf(stdout, "Deallocated pinned memory block at %p, size %zu bytes.\n", ptr, it->second);
    this->_allocated_blocks.erase(it);
}
