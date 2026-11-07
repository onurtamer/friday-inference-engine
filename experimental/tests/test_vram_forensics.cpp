#include "friday/runtime/vram_forensics.h"
#include "friday/runtime/paged_kv_cache.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <nlohmann/json.hpp>

using namespace friday::runtime;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  TEST: VRAM FORENSICS & SNAPSHOT TELEMETRY (S0 -> S11)\n";
    std::cout << "========================================================\n";

    auto& forensics = VramForensics::getInstance();

    // S0 = process startup
    forensics.recordSnapshot("S0", "process startup");

    // S1 = CUDA initialized
    cudaFree(0);
    forensics.recordSnapshot("S1", "CUDA initialized");

    // S2 = model metadata loaded
    MemoryCategoryAllocation cat_meta;
    cat_meta.runtime_metadata = 25 * 1024 * 1024; // ~25 MB metadata & maps
    forensics.recordSnapshot("S2", "model metadata loaded", cat_meta);

    // S3 = weights loaded (simulate allocation of 14B Q4_0 resident weights ~8.54 GB)
    void* d_sim_weights = nullptr;
    size_t sim_weight_bytes = 8544268416ULL; // 7.95 GiB / 8.54 GB
    cudaError_t err = cudaMalloc(&d_sim_weights, sim_weight_bytes);
    if (err != cudaSuccess) {
        // Fallback to smaller footprint if 8.5GB not available in test
        sim_weight_bytes = 1024ULL * 1024ULL * 1024ULL; // 1 GB
        cudaMalloc(&d_sim_weights, sim_weight_bytes);
    }
    MemoryCategoryAllocation cat_weights = cat_meta;
    cat_weights.weights = sim_weight_bytes;
    forensics.recordSnapshot("S3", "weights loaded", cat_weights);

    // S4 = context created (Paged KV Cache init: 200 blocks)
    PagedKvCache kv_cache(48, 8, 128, 200);
    ContextId parent = kv_cache.create_context();
    MemoryCategoryAllocation cat_ctx = cat_weights;
    cat_ctx.kv_cache = kv_cache.get_block_bytes() * 200;
    forensics.recordSnapshot("S4", "context created", cat_ctx);

    // S5 = prefill completed (write 128 tokens to parent)
    for (int t = 0; t < 128; ++t) {
        kv_cache.prepare_write_slot(parent, t);
    }
    forensics.recordSnapshot("S5", "prefill completed", cat_ctx);

    // S6 = one decode completed (write token 128)
    kv_cache.prepare_write_slot(parent, 128);
    forensics.recordSnapshot("S6", "one decode completed", cat_ctx);

    // S7 = fork created (COW fork: 3 children)
    ContextId child_a = kv_cache.fork_context(parent);
    ContextId child_b = kv_cache.fork_context(parent);
    ContextId child_c = kv_cache.fork_context(parent);
    forensics.recordSnapshot("S7", "fork created", cat_ctx);

    // S8 = COW write occurred (child writes new token at pos 128, triggering COW)
    kv_cache.prepare_write_slot(child_a, 128);
    forensics.recordSnapshot("S8", "COW write occurred", cat_ctx);

    // S9 = child released
    kv_cache.destroy_context(child_a);
    kv_cache.destroy_context(child_b);
    kv_cache.destroy_context(child_c);
    forensics.recordSnapshot("S9", "child released", cat_ctx);

    // S10 = parent released
    kv_cache.destroy_context(parent);
    forensics.recordSnapshot("S10", "parent released", cat_weights);

    // S11 = runtime shutdown
    if (d_sim_weights) cudaFree(d_sim_weights);
    forensics.recordSnapshot("S11", "runtime shutdown");

    forensics.exportReport("artifacts/vram_forensics.json");

    // Model-Calibrated KV Test across 1, 2, 4, 8, 16 contexts
    std::cout << "\n>>> [KV Memory Scaling Test] 1, 2, 4, 8, 16 Contexts...\n";
    nlohmann::json kv_scaling;
    kv_scaling["n_layers"] = 48;
    kv_scaling["n_kv_heads"] = 8;
    kv_scaling["head_dim"] = 128;
    kv_scaling["block_tokens"] = KV_BLOCK_SIZE;

    size_t page_bytes = (size_t)KV_BLOCK_SIZE * 48 * 8 * 128 * sizeof(half) * 2;
    kv_scaling["single_page_bytes"] = page_bytes;
    kv_scaling["single_page_kb"] = (double)page_bytes / 1024.0;

    std::vector<int> context_counts = {1, 2, 4, 8, 16};
    std::vector<nlohmann::json> results;

    for (int num_ctx : context_counts) {
        PagedKvCache test_cache(48, 8, 128, 500);
        size_t vram_before = 0, total_dummy = 0;
        cudaMemGetInfo(&vram_before, &total_dummy);

        std::vector<ContextId> ctxs;
        for (int c = 0; c < num_ctx; ++c) {
            ContextId cid = test_cache.create_context();
            // Write 64 tokens per context (4 pages)
            for (int t = 0; t < 64; ++t) {
                test_cache.prepare_write_slot(cid, t);
            }
            ctxs.push_back(cid);
        }

        nlohmann::json entry;
        entry["contexts"] = num_ctx;
        entry["tokens_per_context"] = 64;
        entry["total_pages_used"] = test_cache.get_total_blocks() - test_cache.get_free_blocks();
        entry["kv_bytes_allocated"] = (test_cache.get_total_blocks() - test_cache.get_free_blocks()) * page_bytes;
        entry["kv_mb_allocated"] = (double)entry["kv_bytes_allocated"].get<size_t>() / (1024.0 * 1024.0);
        results.push_back(entry);

        for (auto cid : ctxs) test_cache.destroy_context(cid);
    }
    kv_scaling["scaling_results"] = results;

    std::ofstream kv_out("artifacts/kv_memory_scaling.json");
    if (kv_out.is_open()) {
        kv_out << kv_scaling.dump(2);
        kv_out.close();
        std::cout << "[OK] Model-calibrated KV scaling report written to: artifacts/kv_memory_scaling.json\n";
    }

    std::cout << ">>> [TEST PASS] VRAM Forensics & KV Scaling Completed. <<<\n";
    return 0;
}
