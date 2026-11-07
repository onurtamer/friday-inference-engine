#ifndef FRIDAY_TYPES_H
#define FRIDAY_TYPES_H

#include <cstdint> // uint32_t, size_t, etc.
#include <vector>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define INT8 WIN_INT8
#include <windows.h>
#undef INT8
#else
#include <aio.h>    // For aio_read/aio_write
#include <unistd.h> // For off_t, sysconf(_SC_PAGESIZE)
#include <fcntl.h>  // For O_DIRECT
#endif

#include <cuda_runtime.h> // For cudaStream_t

// ----- Model ile İlgili Temel Türler -----

struct ExpertLocation {
    uint32_t expert_id;
    uint32_t cluster_id;
    size_t ssd_start_offset;
    size_t ssd_block_size;
};
using ExpertMap = std::vector<ExpertLocation>;

// ----- I/O ile İlgili Temel Türler -----

enum class AsyncIOStatus {
    PENDING,
    COMPLETED,
    FAILED
};

struct AsyncIORequest {
    void* buffer;
    size_t size;
    off_t offset;
    volatile AsyncIOStatus status;
    #ifdef _WIN32
    OVERLAPPED overlapped; // Windows için
    HANDLE file_handle;    // Windows için (PoC basitleştirmesi için IOManager pointer'ını saklamak için kullanıldı)
    #else
    aiocb cb;             // Linux için
    int fd;               // Linux için
    #endif
};

// ----- Bellek Yönetimi ile İlgili Temel Türler -----

const size_t IO_ALIGNMENT = 4096; // Genellikle 4KB (sayfa boyutu)
const size_t DDR4_CACHE_MAX_SIZE_GB = 6;
const size_t DDR4_CACHE_MAX_SIZE_BYTES = DDR4_CACHE_MAX_SIZE_GB * 1024ULL * 1024ULL * 1024ULL;
const size_t VRAM_MAX_SIZE_GB = 2;
const size_t VRAM_MAX_SIZE_BYTES = VRAM_MAX_SIZE_GB * 1024ULL * 1024ULL * 1024ULL;

#endif // FRIDAY_TYPES_H
