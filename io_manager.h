#ifndef FRIDAY_ASYNC_IO_MANAGER_H
#define FRIDAY_ASYNC_IO_MANAGER_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string> // C++ string için

// CUDA_CHECK makrosu (tekrar tanımlanmaması için burada da tanımlıyoruz veya ayrı bir utility header'dan alıyoruz)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                \
    do {                                                \
        cudaError_t err = call;                         \
        if (err != cudaSuccess) {                       \
            fprintf(stderr, "CUDA error at %s:%d: %s: %s\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE);                         \
        }                                               \
    } while (0)
#endif

// AsyncIOManager sınıfı
class AsyncIOManager {
public:
    // C++'ta Zero-Copy I/O için dosya handle'ı ve eşleşen parametreler tutulabilir.
    // Windows için HANDLE, Linux için int fd.
    // Bu örnekte, temel simülasyon ve hizalama mantığına odaklanacağız.
    // Gerçek implementasyon platforma özgü API'ler kullanır (e.g., CreateFile/ReadFile for Windows, open/pread for Linux).

    AsyncIOManager(const std::string& model_file_path); // Constructor dosya yolunu alsın
    ~AsyncIOManager();

    // Zero-Copy okuma yapabilmek için SSD'nin sektör boyutunu sabit olarak alalım
    static const size_t SECTOR_SIZE = 4096; // 4KB

    // Asenkron okuma isteğini kuyruğa ekler.
    // Okuma boyutu (size_bytes) otomatik olarak SECTOR_SIZE'ın katlarına yuvarlanır.
    void enqueueReadExpertWeights(
        int expert_id,
        size_t file_absolute_offset,      // Dosya içindeki mutlak başlangıç ofseti
        size_t requested_size_bytes,      // İstenen okuma boyutu
        unsigned char* d_destination_ptr, // Paketlenmiş ağırlıkların yazılacağı cihaz belleği işaretçisi
        cudaStream_t stream = nullptr);

    // NVMe diskten pinned host buffer içine doğrudan katman okuma
    bool readLayerToHost(
        size_t file_absolute_offset,
        size_t size_bytes,
        unsigned char* h_destination_ptr);

private:
    std::string model_file_path_; // Açılacak model dosyasının yolu
    FILE* file_handle_ = nullptr; // Kalıcı dosya handle'ı (her katmanda dosya açma-kapama iptal)
};

#endif // FRIDAY_ASYNC_IO_MANAGER_H
