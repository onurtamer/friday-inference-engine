#ifndef FRIDAY_MODEL_LAYOUT_H
#define FRIDAY_MODEL_LAYOUT_H

// Tensor table and per-layer packing of a model file, shared by every backend.
// Each layer's blk.<l>.* tensors are read one by one into a packed buffer
// (GGUF does not keep a layer's tensors contiguous), each at a 256-byte
// aligned offset. Pure C++: no CUDA dependency.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

class AsyncIOManager;

struct TensorInfo {
    std::string name;
    size_t absolute_offset = 0;      // in the model file
    size_t size_bytes = 0;
    std::vector<long long> shape;    // GGML order, shape[0] innermost
    std::string dtype;               // "Q2_K", "F32", ...
};

struct LayerChunk {
    size_t file_offset;   // absolute offset in the model file
    size_t size;          // bytes
    size_t rel_offset;    // offset inside the packed layer buffer
};

struct LayerInfo {
    size_t size_bytes = 0;                            // packed size
    std::map<std::string, size_t> tensor_rel_offsets; // tensor name -> offset in the packed buffer
    std::vector<LayerChunk> chunks;                   // file ranges to read, in file order
};

namespace friday {

class GgufFile;

class ModelLayout {
public:
    std::map<std::string, TensorInfo> tensors;
    std::vector<LayerInfo> layers;
    size_t max_layer_bytes = 0;

    void loadGguf(const GgufFile& g);
    void loadTensorMap(const std::string& json_path);   // deprecated converter output
    void buildLayers(int num_layers);                    // throws if a layer has no tensors

    const TensorInfo* find(const std::string& name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }
    // Fills dst (layers[l].size_bytes bytes) from the model file.
    bool readLayer(AsyncIOManager& io, int l, unsigned char* dst) const;
    // Whole tensor into dst (size_bytes bytes).
    bool readTensor(AsyncIOManager& io, const TensorInfo& t, unsigned char* dst) const;
};

}  // namespace friday

#endif  // FRIDAY_MODEL_LAYOUT_H
