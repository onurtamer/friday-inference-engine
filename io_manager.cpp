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
