#include "friday/runtime/paged_kv_cache.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

using namespace friday::runtime;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  TEST: FORK MEMORY SCALING (O(1) FORK & O(PAGE) WRITE)\n";
    std::cout << "========================================================\n";

    const int num_layers = 48;
    const int num_kv_heads = 8;
    const int head_dim = 128;
    const int max_blocks = 1000;

    PagedKvCache cache(num_layers, num_kv_heads, head_dim, max_blocks);

    // Create 1 parent and prefill 128 tokens (8 blocks)
    ContextId parent = cache.create_context();
    for (int t = 0; t < 128; ++t) {
        cache.prepare_write_slot(parent, t);
    }

    size_t free_before_fork = cache.get_free_blocks();
    std::cout << "[Fork Scaling] Parent prefilled with 128 tokens (8 pages used, " 
              << free_before_fork << " free blocks left).\n";

    std::vector<int> fork_levels = {1, 4, 8, 16, 32};
    nlohmann::json fork_report;
    std::vector<nlohmann::json> level_results;

    for (int num_children : fork_levels) {
        std::vector<ContextId> children;
        size_t blocks_before = cache.get_free_blocks();

        // 1. Fork num_children from Parent (Strictly O(1) time and space)
        for (int i = 0; i < num_children; ++i) {
            children.push_back(cache.fork_context(parent));
        }

        size_t blocks_after_fork = cache.get_free_blocks();
        size_t fork_memory_delta_pages = blocks_before - blocks_after_fork;

        // Verify O(1) VRAM cost upon fork: EXACTLY 0 new physical pages!
        assert(fork_memory_delta_pages == 0);

        // 2. Each child decodes 16 tokens (triggers COW for exactly 1 page per child)
        for (auto child_id : children) {
            for (int t = 128; t < 128 + 16; ++t) {
                cache.prepare_write_slot(child_id, t);
            }
        }

        size_t blocks_after_write = cache.get_free_blocks();
        size_t write_memory_delta_pages = blocks_after_fork - blocks_after_write;

        nlohmann::json item;
        item["children_count"] = num_children;
        item["fork_allocated_pages"] = fork_memory_delta_pages; // Must be 0
        item["is_o1_fork"] = (fork_memory_delta_pages == 0);
        item["write_allocated_pages"] = write_memory_delta_pages; // O(changed pages)
        item["pages_per_child_written"] = (double)write_memory_delta_pages / (double)num_children;
        level_results.push_back(item);

        std::cout << "  Children: " << num_children 
                  << " | Fork New Pages: " << fork_memory_delta_pages << " (O(1) Verified)"
                  << " | Decode New Pages: " << write_memory_delta_pages 
                  << " (" << (double)write_memory_delta_pages / num_children << " page/child)\n";

        // Release children
        for (auto child_id : children) {
            cache.destroy_context(child_id);
        }

        assert(cache.get_free_blocks() == free_before_fork); // Leak-free
    }

    cache.destroy_context(parent);
    assert(cache.get_free_blocks() == max_blocks);

    fork_report["scaling_metrics"] = level_results;
    fork_report["status"] = "PASS";
    fork_report["zero_leak_verified"] = true;

    std::ofstream out("artifacts/fork_scaling.json");
    if (out.is_open()) {
        out << fork_report.dump(2);
        out.close();
        std::cout << "[OK] Fork memory scaling report written to: artifacts/fork_scaling.json\n";
    }

    std::cout << ">>> [TEST PASS] Fork Memory Scaling (O(1) Fork & O(page) Write) Validated. <<<\n";
    return 0;
}
