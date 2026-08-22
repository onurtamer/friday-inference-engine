#ifndef FRIDAY_MEMORY_MANAGER_H
#define FRIDAY_MEMORY_MANAGER_H

#include "friday_types.h"
#include <cuda_runtime.h> // For cudaHostAlloc, cudaFreeHost
#include <stdexcept>
#include <vector>
#include <mutex>
#include <map> // Bellek bloklarını takip etmek için

// Hata kontrol makrosu
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

// ZeroCopyMemoryManager sınıfı, pinned (sayfalanmayan) bellek tahsisi ve yönetimini yapar.
class ZeroCopyMemoryManager {
public:
    ZeroCopyMemoryManager();
    ~ZeroCopyMemoryManager();

    // Belirtilen boyutta (IO_ALIGNMENT hizalı) pinned bellek tahsis eder.
    // Bu bellek CPU tarafından erişilebilir ve GPU'ya hızlıca aktarılabilir.
    void* allocate_pinned(size_t size);

    // Tahsis edilmiş pinned belleği serbest bırakır.
    void deallocate_pinned(void* ptr);

    // Belleğin hizalı olup olmadığını kontrol eder.
    static bool is_aligned(void* ptr, size_t alignment = IO_ALIGNMENT) {
        return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
    }

private:
    std::mutex _mutex; // Bellek işlemleri için kilit
    std::map<void*, size_t> _allocated_blocks; // Tahsis edilen blokları takip eder
};

#endif // FRIDAY_MEMORY_MANAGER_H
