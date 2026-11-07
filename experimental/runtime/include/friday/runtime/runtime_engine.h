#pragma once

#include "types.h"
#include "paged_kv_cache.h"
#include "draft_provider.h"
#include <string>
#include <vector>
#include <memory>

namespace friday {
namespace runtime {

// Invariant 1: Runtime Isolation
// Pure inference engine. No persona, identity, chat templates, or agentic workflow knowledge.
class FridayRuntime {
public:
    FridayRuntime();
    ~FridayRuntime();

    // Initialization with deterministic VRAM plan (Invariant 3)
    bool init(const std::string& model_path, 
              const std::string& tensor_map_path, 
              const ModelConfig& config, 
              double vram_limit_gb = 12.0);

    // Context Management (COW Enabled - Invariant 4)
    ContextId create_context();
    ContextId fork_context(ContextId parent_ctx);
    void destroy_context(ContextId ctx_id);

    // Batched Prefill (Invariant 10 - Empirical Benchmarking)
    // Computes all prompt tokens in a single/batched forward pass
    bool prefill_batched(ContextId ctx_id, 
                         const TokenId* tokens, 
                         size_t num_tokens, 
                         half* d_output_logits, 
                         cudaStream_t stream = 0);

    // Single-step Autoregressive Decode
    bool decode_step(ContextId ctx_id, 
                     TokenId token, 
                     int seq_pos, 
                     half* d_output_logits, 
                     cudaStream_t stream = 0);

    const ModelConfig& get_config() const { return config_; }
    const VramBudgetProfile& get_vram_profile() const { return vram_profile_; }
    PagedKvCache* get_kv_cache() { return kv_cache_.get(); }

private:
    ModelConfig config_;
    VramBudgetProfile vram_profile_;
    std::unique_ptr<PagedKvCache> kv_cache_;
    std::unique_ptr<IDraftProvider> draft_provider_;
    bool initialized_ = false;
};

} // namespace runtime
} // namespace friday
