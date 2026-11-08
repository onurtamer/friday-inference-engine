#ifndef FRIDAY_GGUF_FILE_H
#define FRIDAY_GGUF_FILE_H

// Minimal, bounds-checked GGUF (v2/v3) reader: header, metadata key/values and
// the tensor table. Tensor data is not read here; offsets are absolute file
// offsets. Malformed files raise std::runtime_error (never read out of bounds).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace friday {

// GGML tensor types used by supported models (ids from ggml.h).
enum GgmlType : uint32_t {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
    GGML_Q8_0 = 8, GGML_Q8_1 = 9, GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13,
    GGML_Q6_K = 14, GGML_Q8_K = 15, GGML_BF16 = 30,
};

// Type name as written by gguf-py / used in tensor maps ("Q2_K", "F32", ...);
// nullptr for types this engine does not know.
const char* ggmlTypeName(uint32_t type);

// Bytes for a tensor of `type` with dims ne (ne[0] innermost). Throws on
// unknown types or ne[0] not a multiple of the block size.
uint64_t ggmlTensorBytes(uint32_t type, const std::vector<int64_t>& ne);

struct GgufValue {
    enum Kind { UINT, INT, FLOAT, BOOL, STRING, ARRAY } kind = UINT;
    uint32_t gguf_type = 0;      // original GGUF value type id
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    // Arrays: element type and the elements in one of these vectors.
    uint32_t elem_type = 0;
    std::vector<std::string> strs;
    std::vector<int64_t> ints;   // all integer and bool element types
    std::vector<double> floats;
};

struct GgufTensor {
    std::string name;
    std::vector<int64_t> ne;     // dims, ne[0] innermost (GGML order)
    uint32_t type = 0;
    uint64_t offset = 0;         // absolute file offset of the data
    uint64_t size = 0;           // bytes
};

class GgufFile {
public:
    // Parses the header of `path`. Throws std::runtime_error on any error.
    explicit GgufFile(const std::string& path);

    const std::string& path() const { return path_; }
    uint32_t version() const { return version_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t dataOffset() const { return data_offset_; }
    uint64_t fileSize() const { return file_size_; }

    const std::map<std::string, GgufValue>& metadata() const { return kv_; }
    const std::vector<GgufTensor>& tensors() const { return tensors_; }
    const GgufTensor* findTensor(const std::string& name) const;

    bool has(const std::string& key) const { return kv_.count(key) != 0; }
    // Typed getters; throw if the key is missing or has an incompatible type.
    uint64_t getUInt(const std::string& key) const;
    double getFloat(const std::string& key) const;
    std::string getString(const std::string& key) const;
    const std::vector<std::string>& getStringArray(const std::string& key) const;
    const std::vector<int64_t>& getIntArray(const std::string& key) const;
    // Defaults when the key is missing.
    uint64_t getUInt(const std::string& key, uint64_t def) const { return has(key) ? getUInt(key) : def; }
    double getFloat(const std::string& key, double def) const { return has(key) ? getFloat(key) : def; }
    std::string getString(const std::string& key, const std::string& def) const { return has(key) ? getString(key) : def; }

    std::string architecture() const { return getString("general.architecture"); }

private:
    std::string path_;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    uint64_t data_offset_ = 0;
    uint64_t file_size_ = 0;
    std::map<std::string, GgufValue> kv_;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, size_t> tensor_index_;
};

}  // namespace friday

#endif  // FRIDAY_GGUF_FILE_H
