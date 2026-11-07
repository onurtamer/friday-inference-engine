#include "friday/runtime/paged_kv_cache.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace friday::runtime;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  TEST: KV CONTINUITY ACROSS BLOCK BOUNDARIES\n";
    std::cout << "========================================================\n";

    const int num_layers = 48;
    const int num_kv_heads = 8;
    const int head_dim = 128;
    const int num_blocks = 50;

    PagedKvCache cache(num_layers, num_kv_heads, head_dim, num_blocks);

    ContextId ctx = cache.create_context();

    // Write 64 tokens across exactly 4 pages of 16 tokens each:
    // Page 0: tokens 0..15
    // Page 1: tokens 16..31  <- Boundary 1 (15->16)
    // Page 2: tokens 32..47  <- Boundary 2 (31->32)
    // Page 3: tokens 48..63  <- Boundary 3 (47->48)
    for (int t = 0; t < 64; ++t) {
        cache.prepare_write_slot(ctx, t);
    }

    const auto& block_table = cache.get_block_table(ctx);
    assert(block_table.size() == 4);

    std::cout << "[KV Continuity] Block Table for 64 tokens: ["
              << block_table[0] << ", " << block_table[1] << ", "
              << block_table[2] << ", " << block_table[3] << "]\n";

    // Verify boundary transitions
    bool boundaries_valid = true;
    for (int page = 0; page < 3; ++page) {
        BlockId b_curr = block_table[page];
        BlockId b_next = block_table[page + 1];
        if (b_curr == b_next) {
            boundaries_valid = false;
            std::cerr << "[Hata] Page " << page << " and " << page + 1 << " share same physical block without COW!\n";
        }
    }

    // Verify all 4 blocks have ref_count == 1
    for (int page = 0; page < 4; ++page) {
        assert(cache.get_physical_block(block_table[page]).ref_count == 1);
    }

    cache.destroy_context(ctx);
    assert(cache.get_free_blocks() == num_blocks); // 100% leak free

    if (boundaries_valid) {
        std::cout << ">>> [TEST PASS] KV Continuity Across Block Boundaries Validated. <<<\n";
        return 0;
    } else {
        std::cerr << ">>> [TEST FAIL] Boundary Continuity Failed! <<<\n";
        return 1;
    }
}
