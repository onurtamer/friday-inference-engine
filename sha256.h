#ifndef FRIDAY_SHA256_H
#define FRIDAY_SHA256_H

// Self-contained SHA-256 (FIPS 180-4), used to identify model files.

#include <cstddef>
#include <cstdint>
#include <string>

namespace friday {

class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t len);
    std::string hexDigest();  // finalizes; call once

private:
    void block(const uint8_t* p);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t buf_len_ = 0;
    uint64_t total_ = 0;
};

std::string sha256Hex(const void* data, size_t len);
// Streams the file; progress(bytes_done, bytes_total) is called about every 256 MiB.
std::string sha256File(const std::string& path, void (*progress)(uint64_t, uint64_t) = nullptr);

}  // namespace friday

#endif  // FRIDAY_SHA256_H
