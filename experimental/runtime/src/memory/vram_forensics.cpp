#include "friday/runtime/vram_forensics.h"
#include <fstream>
#include <iostream>
#include <iomanip>

namespace friday {
namespace runtime {

void VramForensics::recordSnapshot(const std::string& stage_id, 
                                   const std::string& desc,
                                   const MemoryCategoryAllocation& cat) 
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    size_t used_bytes = total_bytes - free_bytes;
    if (used_bytes > global_peak_vram_bytes_) {
        global_peak_vram_bytes_ = used_bytes;
    }

    VramSnapshotStage stage;
    stage.stage_id = stage_id;
    stage.stage_description = desc;
    stage.total_vram_bytes = total_bytes;
    stage.free_vram_bytes = free_bytes;
    stage.used_vram_bytes = used_bytes;
    stage.categories = cat;
    stage.peak_vram_bytes = global_peak_vram_bytes_;
    stage.allocation_count = ++alloc_counter_;
    stage.free_count = free_counter_;

    // Fragmentation estimate based on categorized accounted memory vs raw device used bytes
    size_t accounted = cat.weights + cat.kv_cache + cat.activations + cat.scratch + cat.runtime_metadata;
    if (used_bytes > accounted && accounted > 0) {
        stage.fragmentation_estimate = (double)(used_bytes - accounted) / (double)used_bytes;
    } else {
        stage.fragmentation_estimate = 0.0;
    }

    snapshots_.push_back(stage);

    std::cout << "[VRAM Forensics] Snapshot " << stage_id << " (" << desc << "): "
              << std::fixed << std::setprecision(2) << (double)used_bytes / (1024.0*1024.0*1024.0) << " GB Used / "
              << (double)total_bytes / (1024.0*1024.0*1024.0) << " GB Total (Peak: "
              << (double)global_peak_vram_bytes_ / (1024.0*1024.0*1024.0) << " GB)\n";
}

void VramForensics::exportReport(const std::string& json_path) {
    nlohmann::json j;
    j["device"] = "NVIDIA GeForce RTX 3060 (12GB sm_86)";
    j["global_peak_vram_bytes"] = global_peak_vram_bytes_;
    j["global_peak_vram_gb"] = (double)global_peak_vram_bytes_ / (1024.0*1024.0*1024.0);

    std::vector<nlohmann::json> snap_list;
    for (const auto& s : snapshots_) {
        nlohmann::json item;
        item["stage_id"] = s.stage_id;
        item["stage_description"] = s.stage_description;
        item["total_vram_bytes"] = s.total_vram_bytes;
        item["free_vram_bytes"] = s.free_vram_bytes;
        item["used_vram_bytes"] = s.used_vram_bytes;
        item["used_vram_gb"] = (double)s.used_vram_bytes / (1024.0*1024.0*1024.0);
        item["peak_vram_bytes"] = s.peak_vram_bytes;
        item["fragmentation_estimate"] = s.fragmentation_estimate;
        item["categories"] = {
            {"weights_bytes", s.categories.weights},
            {"kv_cache_bytes", s.categories.kv_cache},
            {"activations_bytes", s.categories.activations},
            {"scratch_bytes", s.categories.scratch},
            {"runtime_metadata_bytes", s.categories.runtime_metadata},
            {"cuda_backend_bytes", s.categories.cuda_backend},
            {"temporary_buffers_bytes", s.categories.temporary_buffers}
        };
        snap_list.push_back(item);
    }
    j["snapshots"] = snap_list;

    std::ofstream out(json_path);
    if (out.is_open()) {
        out << j.dump(2);
        out.close();
        std::cout << "[OK] VRAM Forensics report written to: " << json_path << "\n";
    } else {
        std::cerr << "[HATA] VRAM Forensics report yazilamadi: " << json_path << "\n";
    }
}

} // namespace runtime
} // namespace friday
