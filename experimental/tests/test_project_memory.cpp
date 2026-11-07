#include "friday/cognition/project_memory.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <nlohmann/json.hpp>

using namespace friday::cognition;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  TEST: PHASE 4 PERSISTENT PROJECT MEMORY & SYMBOL GRAPH\n";
    std::cout << "========================================================\n";

    ProjectMemoryManager memory;

    // 1. Index Core Architecture Symbols
    CodeSymbol sym1;
    sym1.name = "PagedKvCache";
    sym1.type = SymbolType::CLASS;
    sym1.file_path = "runtime/src/kv_cache/paged_kv_cache.cpp";
    sym1.line_number = 15;
    sym1.signature = "class PagedKvCache";
    sym1.dependencies = {"cudaMalloc", "cudaFree", "cudaMemcpyAsync"};
    memory.register_symbol(sym1);

    CodeSymbol sym2;
    sym2.name = "GpuMemoryArena";
    sym2.type = SymbolType::CLASS;
    sym2.file_path = "runtime/src/memory/memory_arena.cpp";
    sym2.line_number = 10;
    sym2.signature = "class GpuMemoryArena";
    sym2.dependencies = {"PagedKvCache", "VramBudgetProfile"};
    memory.register_symbol(sym2);

    CodeSymbol sym3;
    sym3.name = "DialecticalChoir";
    sym3.type = SymbolType::CLASS;
    sym3.file_path = "cognition/src/dialectical_choir.cpp";
    sym3.line_number = 8;
    sym3.signature = "class DialecticalChoir";
    sym3.dependencies = {"PagedKvCache", "fork_context"};
    memory.register_symbol(sym3);

    CodeSymbol sym4;
    sym4.name = "launchPagedFusedAttention";
    sym4.type = SymbolType::KERNEL;
    sym4.file_path = "runtime/src/kernels/paged_attention.cu";
    sym4.line_number = 140;
    sym4.signature = "void launchPagedFusedAttention(...)";
    sym4.dependencies = {"pagedFusedAttentionKernel", "updatePagedKvCacheKernel"};
    memory.register_symbol(sym4);

    assert(memory.get_symbol_count() == 4);
    std::cout << "[Step 1 PASS] 4 Key Architectural Symbols Indexed into Graph.\n";

    // 2. Test Exact Match Symbol Lookup
    const auto* s = memory.lookup_symbol("PagedKvCache");
    assert(s != nullptr);
    assert(s->type == SymbolType::CLASS);
    assert(s->file_path == "runtime/src/kv_cache/paged_kv_cache.cpp");
    std::cout << "[Step 2 PASS] Exact Match Lookup Verified: " << s->name << " at " << s->file_path << "\n";

    // 3. Test Prefix Search
    auto prefix_res = memory.search_by_prefix("launch");
    assert(prefix_res.size() == 1);
    assert(prefix_res[0].name == "launchPagedFusedAttention");
    std::cout << "[Step 3 PASS] Prefix Search Verified: Found " << prefix_res[0].name << "\n";

    // 4. Test Dependency Graph Traversal
    auto deps = memory.get_dependencies("DialecticalChoir");
    assert(deps.size() == 2);
    assert(deps[0] == "PagedKvCache");
    assert(deps[1] == "fork_context");
    std::cout << "[Step 4 PASS] Dependency Graph Traversal Verified: DialecticalChoir -> PagedKvCache, fork_context.\n";

    // Export Phase 4 Report
    nlohmann::json rep;
    rep["test_name"] = "Phase 4: Persistent Project Memory & Symbol Graph";
    rep["status"] = "PASS";
    rep["indexed_symbols_count"] = memory.get_symbol_count();
    rep["exact_lookup_verified"] = true;
    rep["prefix_search_verified"] = true;
    rep["dependency_graph_traversal_verified"] = true;

    std::ofstream out("artifacts/phase4_project_memory_report.json");
    if (out.is_open()) {
        out << rep.dump(2);
        out.close();
        std::cout << "[OK] Report written to: artifacts/phase4_project_memory_report.json\n";
    }

    std::cout << "\n>>> [FAZ 4 PERSISTENT PROJECT MEMORY TEST PASS] <<<\n";
    return 0;
}
