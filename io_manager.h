#ifndef FRIDAY_ASYNC_IO_MANAGER_H
#define FRIDAY_ASYNC_IO_MANAGER_H

#include <cstdio>
#include <cstdlib>
#include <string> // C++ string için


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
