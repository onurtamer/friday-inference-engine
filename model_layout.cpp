#include "model_layout.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "gguf_file.h"
#include "io_manager.h"

namespace friday {

void ModelLayout::loadGguf(const GgufFile& g) {
    tensors.clear();
    for (const auto& t : g.tensors()) {
        const char* type_name = ggmlTypeName(t.type);
        TensorInfo info;
        info.name = t.name;
        info.absolute_offset = (size_t)t.offset;
        info.size_bytes = (size_t)t.size;
        info.shape.assign(t.ne.begin(), t.ne.end());
        info.dtype = type_name ? type_name : "UNKNOWN";
        tensors[info.name] = info;
    }
    printf("[ModelLayout] %zu tensors from GGUF v%u '%s'\n", tensors.size(), g.version(), g.path().c_str());
}

void ModelLayout::loadTensorMap(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f) throw std::runtime_error("cannot open tensor map " + json_path);
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("tensor map " + json_path + ": " + e.what());
    }
    tensors.clear();
    for (const auto& item : j) {
        TensorInfo info;
        info.name = item.at("name").get<std::string>();
        info.absolute_offset = item.at("absolute_offset").get<size_t>();
        info.size_bytes = item.at("size_bytes").get<size_t>();
        info.shape = item.at("shape").get<std::vector<long long>>();
        info.dtype = item.at("dtype").get<std::string>();
        tensors[info.name] = info;
    }
    printf("[ModelLayout] %zu tensors from tensor map '%s'\n", tensors.size(), json_path.c_str());
}

void ModelLayout::buildLayers(int num_layers) {
    layers.assign(num_layers, LayerInfo());
    std::vector<std::vector<TensorInfo>> per_layer(num_layers);
    for (const auto& [name, info] : tensors) {
        if (name.rfind("blk.", 0) != 0) continue;
        const size_t dot1 = name.find('.');
        const size_t dot2 = name.find('.', dot1 + 1);
        if (dot2 == std::string::npos) continue;
        const int l = std::stoi(name.substr(dot1 + 1, dot2 - dot1 - 1));
        if (l >= 0 && l < num_layers) per_layer[l].push_back(info);
    }
    max_layer_bytes = 0;
    for (int l = 0; l < num_layers; ++l) {
        auto& lt = per_layer[l];
        if (lt.empty()) throw std::runtime_error("no tensors for layer " + std::to_string(l));
        std::sort(lt.begin(), lt.end(), [](const TensorInfo& a, const TensorInfo& b) {
            return a.absolute_offset < b.absolute_offset;
        });
        size_t cur = 0;
        for (const auto& t : lt) {
            cur = (cur + 255) / 256 * 256;
            layers[l].chunks.push_back({t.absolute_offset, t.size_bytes, cur});
            layers[l].tensor_rel_offsets[t.name] = cur;
            cur += t.size_bytes;
        }
        layers[l].size_bytes = cur;
        max_layer_bytes = std::max(max_layer_bytes, cur);
    }
    printf("[ModelLayout] %d layers, largest %.2f MB\n", num_layers, (double)max_layer_bytes / (1024.0 * 1024.0));
}

bool ModelLayout::readLayer(AsyncIOManager& io, int l, unsigned char* dst) const {
    const auto& li = layers[l];
    size_t filled = 0;
    for (const auto& c : li.chunks) {
        if (c.rel_offset > filled) std::memset(dst + filled, 0, c.rel_offset - filled);  // alignment padding
        if (!io.readLayerToHost(c.file_offset, c.size, dst + c.rel_offset)) return false;
        filled = c.rel_offset + c.size;
    }
    return true;
}

bool ModelLayout::readTensor(AsyncIOManager& io, const TensorInfo& t, unsigned char* dst) const {
    return io.readLayerToHost(t.absolute_offset, t.size_bytes, dst);
}

}  // namespace friday
