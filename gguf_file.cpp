#include "gguf_file.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace friday {

namespace {

struct TypeTraits { const char* name; uint32_t block; uint32_t bytes; };

const TypeTraits* traits(uint32_t type) {
    static const std::map<uint32_t, TypeTraits> t = {
        {GGML_F32, {"F32", 1, 4}},       {GGML_F16, {"F16", 1, 2}},      {GGML_Q4_0, {"Q4_0", 32, 18}},
        {GGML_Q4_1, {"Q4_1", 32, 20}},   {GGML_Q5_0, {"Q5_0", 32, 22}},  {GGML_Q5_1, {"Q5_1", 32, 24}},
        {GGML_Q8_0, {"Q8_0", 32, 34}},   {GGML_Q8_1, {"Q8_1", 32, 36}},  {GGML_Q2_K, {"Q2_K", 256, 84}},
        {GGML_Q3_K, {"Q3_K", 256, 110}}, {GGML_Q4_K, {"Q4_K", 256, 144}}, {GGML_Q5_K, {"Q5_K", 256, 176}},
        {GGML_Q6_K, {"Q6_K", 256, 210}}, {GGML_Q8_K, {"Q8_K", 256, 292}}, {GGML_BF16, {"BF16", 1, 2}},
    };
    auto it = t.find(type);
    return it == t.end() ? nullptr : &it->second;
}

// GGUF metadata value types
enum : uint32_t {
    T_U8 = 0, T_I8 = 1, T_U16 = 2, T_I16 = 3, T_U32 = 4, T_I32 = 5, T_F32 = 6, T_BOOL = 7,
    T_STRING = 8, T_ARRAY = 9, T_U64 = 10, T_I64 = 11, T_F64 = 12,
};

// Sanity limits: far above any real model, low enough to reject garbage early.
constexpr uint64_t kMaxStringLen = 1ull << 26;     // 64 MiB
constexpr uint64_t kMaxArrayLen = 1ull << 26;      // 67M elements
constexpr uint64_t kMaxTensors = 1ull << 20;
constexpr uint64_t kMaxKv = 1ull << 16;
constexpr uint32_t kMaxDims = 4;

class Reader {
public:
    Reader(FILE* f, uint64_t size) : f_(f), size_(size) {}
    uint64_t pos() const { return pos_; }
    void read(void* dst, uint64_t n) {
        if (n > size_ - pos_) throw std::runtime_error("GGUF: unexpected end of file");
        if (n && std::fread(dst, 1, (size_t)n, f_) != n) throw std::runtime_error("GGUF: read error");
        pos_ += n;
    }
    void skip(uint64_t n) {
        if (n > size_ - pos_) throw std::runtime_error("GGUF: unexpected end of file");
#ifdef _WIN32
        if (_fseeki64(f_, (long long)n, SEEK_CUR) != 0) throw std::runtime_error("GGUF: seek error");
#else
        if (fseeko(f_, (off_t)n, SEEK_CUR) != 0) throw std::runtime_error("GGUF: seek error");
#endif
        pos_ += n;
    }
    template <typename T> T get() { T v; read(&v, sizeof(T)); return v; }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (n > kMaxStringLen) throw std::runtime_error("GGUF: string too long");
        std::string s((size_t)n, '\0');
        if (n) read(&s[0], n);
        return s;
    }
private:
    FILE* f_;
    uint64_t size_;
    uint64_t pos_ = 0;
};

uint64_t scalarSize(uint32_t t) {
    switch (t) {
        case T_U8: case T_I8: case T_BOOL: return 1;
        case T_U16: case T_I16: return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: return 8;
        default: return 0;
    }
}

void readScalar(Reader& r, uint32_t t, GgufValue& v) {
    switch (t) {
        case T_U8: v.kind = GgufValue::UINT; v.u = r.get<uint8_t>(); break;
        case T_U16: v.kind = GgufValue::UINT; v.u = r.get<uint16_t>(); break;
        case T_U32: v.kind = GgufValue::UINT; v.u = r.get<uint32_t>(); break;
        case T_U64: v.kind = GgufValue::UINT; v.u = r.get<uint64_t>(); break;
        case T_I8: v.kind = GgufValue::INT; v.i = r.get<int8_t>(); break;
        case T_I16: v.kind = GgufValue::INT; v.i = r.get<int16_t>(); break;
        case T_I32: v.kind = GgufValue::INT; v.i = r.get<int32_t>(); break;
        case T_I64: v.kind = GgufValue::INT; v.i = r.get<int64_t>(); break;
        case T_F32: v.kind = GgufValue::FLOAT; v.f = r.get<float>(); break;
        case T_F64: v.kind = GgufValue::FLOAT; v.f = r.get<double>(); break;
        case T_BOOL: v.kind = GgufValue::BOOL; v.b = r.get<uint8_t>() != 0; break;
        case T_STRING: v.kind = GgufValue::STRING; v.s = r.str(); break;
        default: throw std::runtime_error("GGUF: unknown metadata value type " + std::to_string(t));
    }
}

int64_t readIntElem(Reader& r, uint32_t t) {
    switch (t) {
        case T_U8: return r.get<uint8_t>();
        case T_I8: return r.get<int8_t>();
        case T_U16: return r.get<uint16_t>();
        case T_I16: return r.get<int16_t>();
        case T_U32: return r.get<uint32_t>();
        case T_I32: return r.get<int32_t>();
        case T_U64: return (int64_t)r.get<uint64_t>();
        case T_I64: return r.get<int64_t>();
        case T_BOOL: return r.get<uint8_t>() != 0;
        default: throw std::runtime_error("GGUF: bad integer array element type");
    }
}

}  // namespace

const char* ggmlTypeName(uint32_t type) {
    const TypeTraits* t = traits(type);
    return t ? t->name : nullptr;
}

uint64_t ggmlTensorBytes(uint32_t type, const std::vector<int64_t>& ne) {
    const TypeTraits* t = traits(type);
    if (!t) throw std::runtime_error("GGUF: unsupported tensor type " + std::to_string(type));
    if (ne.empty()) throw std::runtime_error("GGUF: tensor without dimensions");
    if (ne[0] % t->block != 0) throw std::runtime_error("GGUF: row length not a multiple of the block size");
    uint64_t rows = 1;
    for (size_t d = 1; d < ne.size(); ++d) {
        if (ne[d] <= 0 || rows > (1ull << 40) / (uint64_t)ne[d]) throw std::runtime_error("GGUF: bad tensor dims");
        rows *= (uint64_t)ne[d];
    }
    return (uint64_t)(ne[0] / t->block) * t->bytes * rows;
}

GgufFile::GgufFile(const std::string& path) : path_(path) {
    std::unique_ptr<FILE, int (*)(FILE*)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!f) throw std::runtime_error("GGUF: cannot open " + path);
#ifdef _WIN32
    _fseeki64(f.get(), 0, SEEK_END);
    file_size_ = (uint64_t)_ftelli64(f.get());
    _fseeki64(f.get(), 0, SEEK_SET);
#else
    fseeko(f.get(), 0, SEEK_END);
    file_size_ = (uint64_t)ftello(f.get());
    fseeko(f.get(), 0, SEEK_SET);
#endif
    Reader r(f.get(), file_size_);

    char magic[4];
    r.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) throw std::runtime_error("GGUF: bad magic in " + path);
    version_ = r.get<uint32_t>();
    if (version_ != 2 && version_ != 3) throw std::runtime_error("GGUF: unsupported version " + std::to_string(version_));
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv = r.get<uint64_t>();
    if (n_tensors > kMaxTensors || n_kv > kMaxKv) throw std::runtime_error("GGUF: implausible header counts");

    for (uint64_t k = 0; k < n_kv; ++k) {
        std::string key = r.str();
        const uint32_t t = r.get<uint32_t>();
        GgufValue v;
        v.gguf_type = t;
        if (t == T_ARRAY) {
            v.kind = GgufValue::ARRAY;
            v.elem_type = r.get<uint32_t>();
            const uint64_t n = r.get<uint64_t>();
            if (n > kMaxArrayLen) throw std::runtime_error("GGUF: array too long: " + key);
            if (v.elem_type == T_STRING) {
                v.strs.reserve((size_t)n);
                for (uint64_t j = 0; j < n; ++j) v.strs.push_back(r.str());
            } else if (v.elem_type == T_F32 || v.elem_type == T_F64) {
                v.floats.reserve((size_t)n);
                for (uint64_t j = 0; j < n; ++j)
                    v.floats.push_back(v.elem_type == T_F32 ? (double)r.get<float>() : r.get<double>());
            } else if (scalarSize(v.elem_type) != 0) {
                v.ints.reserve((size_t)n);
                for (uint64_t j = 0; j < n; ++j) v.ints.push_back(readIntElem(r, v.elem_type));
            } else {
                throw std::runtime_error("GGUF: unsupported array element type in " + key);
            }
        } else {
            readScalar(r, t, v);
        }
        if (!kv_.emplace(key, std::move(v)).second) throw std::runtime_error("GGUF: duplicate key " + key);
    }

    if (has("general.alignment")) {
        alignment_ = getUInt("general.alignment");
        if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) throw std::runtime_error("GGUF: bad alignment");
    }

    tensors_.reserve((size_t)n_tensors);
    for (uint64_t k = 0; k < n_tensors; ++k) {
        GgufTensor t;
        t.name = r.str();
        const uint32_t nd = r.get<uint32_t>();
        if (nd == 0 || nd > kMaxDims) throw std::runtime_error("GGUF: bad dimension count for " + t.name);
        for (uint32_t d = 0; d < nd; ++d) {
            const int64_t n = (int64_t)r.get<uint64_t>();
            if (n <= 0 || n > (1ll << 40)) throw std::runtime_error("GGUF: bad dimension for " + t.name);
            t.ne.push_back(n);
        }
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();  // relative to the data section for now
        t.size = ggmlTensorBytes(t.type, t.ne);
        if (t.offset % alignment_ != 0) throw std::runtime_error("GGUF: misaligned tensor " + t.name);
        if (!tensor_index_.emplace(t.name, tensors_.size()).second)
            throw std::runtime_error("GGUF: duplicate tensor " + t.name);
        tensors_.push_back(std::move(t));
    }

    data_offset_ = (r.pos() + alignment_ - 1) / alignment_ * alignment_;
    for (auto& t : tensors_) {
        if (t.offset > file_size_ || data_offset_ > file_size_ - t.offset || t.size > file_size_ - data_offset_ - t.offset)
            throw std::runtime_error("GGUF: tensor data out of file bounds: " + t.name);
        t.offset += data_offset_;
    }
}

const GgufTensor* GgufFile::findTensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

static const GgufValue& need(const std::map<std::string, GgufValue>& kv, const std::string& key) {
    auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error("GGUF: missing metadata key " + key);
    return it->second;
}

uint64_t GgufFile::getUInt(const std::string& key) const {
    const GgufValue& v = need(kv_, key);
    if (v.kind == GgufValue::UINT) return v.u;
    if (v.kind == GgufValue::INT && v.i >= 0) return (uint64_t)v.i;
    throw std::runtime_error("GGUF: key is not a non-negative integer: " + key);
}

double GgufFile::getFloat(const std::string& key) const {
    const GgufValue& v = need(kv_, key);
    if (v.kind == GgufValue::FLOAT) return v.f;
    if (v.kind == GgufValue::UINT) return (double)v.u;
    if (v.kind == GgufValue::INT) return (double)v.i;
    throw std::runtime_error("GGUF: key is not a number: " + key);
}

std::string GgufFile::getString(const std::string& key) const {
    const GgufValue& v = need(kv_, key);
    if (v.kind != GgufValue::STRING) throw std::runtime_error("GGUF: key is not a string: " + key);
    return v.s;
}

const std::vector<std::string>& GgufFile::getStringArray(const std::string& key) const {
    const GgufValue& v = need(kv_, key);
    if (v.kind != GgufValue::ARRAY || v.elem_type != T_STRING)
        throw std::runtime_error("GGUF: key is not a string array: " + key);
    return v.strs;
}

const std::vector<int64_t>& GgufFile::getIntArray(const std::string& key) const {
    const GgufValue& v = need(kv_, key);
    if (v.kind != GgufValue::ARRAY || v.elem_type == T_STRING || v.elem_type == T_F32 || v.elem_type == T_F64)
        throw std::runtime_error("GGUF: key is not an integer array: " + key);
    return v.ints;
}

}  // namespace friday
