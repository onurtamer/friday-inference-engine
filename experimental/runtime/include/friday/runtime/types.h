#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace friday {
namespace runtime {

using ContextId = int32_t;
using BlockId = int32_t;
using TokenId = int32_t;

enum class DeviceType {
    CUDA,
    CPU
};

enum class QuantType {
    UNKNOWN = 0,
    FP32 = 1,
    FP16 = 2,
    INT8 = 3,
    INT4 = 4,
    Q4_0 = 5,
    Q4_K = 6,
    Q4_1 = 7,
    Q6_K = 8,
    Q2_K = 9,
    Q3_K = 10,
    TERNARY = 11
};

struct ModelConfig {
    int hidden_dim = 5120;
    int num_heads = 40;
    int num_kv_heads = 8;
    int num_layers = 48; // Default 14B Qwen2.5 (48 layers)
    int max_position_embeddings = 32768;
    int intermediate_dim = 13824;
    int vocab_size = 152064;
    float rms_norm_epsilon = 1e-6f;
    float rope_freq_base = 1000000.0f;
    std::string quant_type_str = "Q4_0";
    QuantType quant_type = QuantType::Q4_0;
};

struct VramBudgetProfile {
    size_t total_vram_bytes = 0;
    size_t weights_bytes = 0;
    size_t activation_arena_bytes = 0;
    size_t kv_page_arena_bytes = 0;
    size_t scratch_workspace_bytes = 0;
    size_t free_margin_bytes = 0;

    double weights_gb() const { return (double)weights_bytes / (1024.0 * 1024.0 * 1024.0); }
    double kv_arena_gb() const { return (double)kv_page_arena_bytes / (1024.0 * 1024.0 * 1024.0); }
    double total_used_gb() const { 
        return (double)(weights_bytes + activation_arena_bytes + kv_page_arena_bytes + scratch_workspace_bytes) / (1024.0 * 1024.0 * 1024.0); 
    }
};

struct LatencyStats {
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p95_ms = 0.0;
};

} // namespace runtime
} // namespace friday
