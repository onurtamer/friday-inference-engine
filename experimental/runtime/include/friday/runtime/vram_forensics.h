#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

namespace friday {
namespace runtime {

struct MemoryCategoryAllocation {
    size_t weights = 0;
    size_t kv_cache = 0;
    size_t activations = 0;
    size_t scratch = 0;
    size_t runtime_metadata = 0;
    size_t cuda_backend = 0;
    size_t temporary_buffers = 0;
};

struct VramSnapshotStage {
    std::string stage_id;
    std::string stage_description;
    size_t total_vram_bytes = 0;
    size_t free_vram_bytes = 0;
    size_t used_vram_bytes = 0;
    MemoryCategoryAllocation categories;
    size_t peak_vram_bytes = 0;
    size_t allocation_count = 0;
    size_t free_count = 0;
    double fragmentation_estimate = 0.0;
};

class VramForensics {
public:
    static VramForensics& getInstance() {
        static VramForensics instance;
        return instance;
    }

    void recordSnapshot(const std::string& stage_id, 
                        const std::string& desc,
                        const MemoryCategoryAllocation& cat = {});

    void exportReport(const std::string& json_path);

    const std::vector<VramSnapshotStage>& getSnapshots() const { return snapshots_; }

private:
    VramForensics() = default;
    std::vector<VramSnapshotStage> snapshots_;
    size_t global_peak_vram_bytes_ = 0;
    size_t alloc_counter_ = 0;
    size_t free_counter_ = 0;
};

} // namespace runtime
} // namespace friday
