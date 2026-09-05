#include "io_manager.h"
#include <algorithm> // For std::max
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream
#include <vector>    // For std::vector

AsyncIOManager::AsyncIOManager(const std::string& model_file_path)
    : model_file_path_(model_file_path)
{
    file_handle_ = fopen(model_file_path_.c_str(), "rb");
    if (!file_handle_) {
        std::string fallback = "qwen2.5-0.5b-instruct-q4_0.gguf";
        file_handle_ = fopen(fallback.c_str(), "rb");
        if (file_handle_) {
            model_file_path_ = fallback;
        } else {
            fallback = "friday_aligned_int4.bin";
            file_handle_ = fopen(fallback.c_str(), "rb");
            if (file_handle_) {
                model_file_path_ = fallback;
            }
        }
    }
    if (file_handle_) {
        printf("[AsyncIOManager] Model dosyasi kalici acildi: '%s'\n", model_file_path_.c_str());
    } else {
        fprintf(stderr, "[AsyncIOManager] Hata: Model dosyasi acilamadi: '%s'\n", model_file_path_.c_str());
    }
}

AsyncIOManager::~AsyncIOManager() {
    if (file_handle_) {
        fclose(file_handle_);
        file_handle_ = nullptr;
    }
    printf("[AsyncIOManager] Kapatiliyor.\n");
}

void AsyncIOManager::enqueueReadExpertWeights(
    int expert_id,
    size_t file_absolute_offset,
    size_t requested_size_bytes,
    unsigned char* d_destination_ptr,
    cudaStream_t stream)
{
    size_t SECTOR_SIZE = 4096; 

    size_t aligned_offset = file_absolute_offset - (file_absolute_offset % SECTOR_SIZE);
    size_t shift_bytes = file_absolute_offset - aligned_offset;
    size_t aligned_read_size = ((requested_size_bytes + shift_bytes + SECTOR_SIZE - 1) / SECTOR_SIZE) * SECTOR_SIZE;

    printf("[AsyncIOManager] Expert %d icin Sektor Kaydirma hesaplamalari:\n", expert_id);
    printf("  Orijinal Ofset: %zu bayt\n", file_absolute_offset);
    printf("  Istenen Boyut: %zu bayt\n", requested_size_bytes);
    printf("  Hizali Ofset: %zu bayt\n", aligned_offset);
    printf("  Kaydirma (Shift) Miktari: %zu bayt\n", shift_bytes);
    printf("  Hizali Okuma Boyutu: %zu bayt\n", aligned_read_size);

    printf("[AsyncIOManager] Expert %d icin asenkron okuma kuyruga eklendi. \n"
           "Mutlak Ofset: %zu, Okunacak Boyut (Hizali): %zu bayt. Hedef: %p\n",
           expert_id, aligned_offset, aligned_read_size, d_destination_ptr);

    std::ifstream file(model_file_path_, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Hata: Model dosyasi acilamadi: %s\n", model_file_path_.c_str());
        exit(EXIT_FAILURE);
    }

    file.seekg(aligned_offset, std::ios::beg);
    if (file.fail()) {
        fprintf(stderr, "Hata: Dosya ofsetine gitme basarisiz oldu (offset: %zu).\n", aligned_offset);
        file.close();
        exit(EXIT_FAILURE);
    }

    std::vector<unsigned char> host_buffer(aligned_read_size);

    file.read(reinterpret_cast<char*>(host_buffer.data()), aligned_read_size);
    if (file.fail() && !file.eof()) {
        fprintf(stderr, "Hata: Dosyadan okuma basarisiz oldu (boyut: %zu bayt, ofset: %zu).\n", aligned_read_size, aligned_offset);
        file.close();
        exit(EXIT_FAILURE);
    }
    file.close();

    CUDA_CHECK(cudaMemcpy(
        d_destination_ptr,
        host_buffer.data() + shift_bytes,
        requested_size_bytes,             
        cudaMemcpyHostToDevice
    ));

    printf("[AsyncIOManager] Expert %d icin gercek okuma ve GPU'ya kopyalama tamamlandi.\n", expert_id);
}

bool AsyncIOManager::readLayerToHost(
    size_t file_absolute_offset,
    size_t size_bytes,
    unsigned char* h_destination_ptr)
{
    if (!file_handle_) {
        fprintf(stderr, "Hata: Model dosya handle'i gecersiz!\n");
        return false;
    }

#ifdef _WIN32
    if (_fseeki64(file_handle_, (long long)file_absolute_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Hata: Dosya ofsetine gitme basarisiz (_fseeki64, offset: %zu).\n", file_absolute_offset);
        return false;
    }
#else
    if (fseeko(file_handle_, (off_t)file_absolute_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Hata: Dosya ofsetine gitme basarisiz (fseeko, offset: %zu).\n", file_absolute_offset);
        return false;
    }
#endif

    size_t read_bytes = fread(h_destination_ptr, 1, size_bytes, file_handle_);
    if (read_bytes != size_bytes) {
        fprintf(stderr, "Hata: Dosyadan okuma basarisiz oldu (okunan: %zu / istenen: %zu bayt, ofset: %zu).\n",
                read_bytes, size_bytes, file_absolute_offset);
        return false;
    }
    return true;
}
