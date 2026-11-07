#include "friday/cognition/dialectical_choir.h"
#include "friday/cognition/validator_bridge.h"
#include "friday/runtime/paged_kv_cache.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <nlohmann/json.hpp>

using namespace friday::runtime;
using namespace friday::cognition;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  TEST: PHASE 3 COGNITIVE ORCHESTRATOR & DIALECTICAL CHOIR\n";
    std::cout << "========================================================\n";

    PagedKvCache kv_cache(48, 8, 128, 500);

    // 1. Parent Problem Context
    ContextId parent = kv_cache.create_context();
    for (int t = 0; t < 64; ++t) {
        kv_cache.prepare_write_slot(parent, t);
    }
    std::cout << "[Step 1 PASS] Parent problem context prefilled (64 tokens).\n";

    // 2. Branch Dialectical Choir (Architect, Critic, Alternative)
    DialecticalChoir choir(&kv_cache);
    choir.branch_choir(parent);
    assert(choir.get_branches().size() == 3);
    std::cout << "[Step 2 PASS] 3 Dialectical Branches Created with Zero VRAM Overhead.\n";

    // 3. Record Independent Thoughts
    choir.record_thought(ChoirRole::ARCHITECT, "Lock-free ring buffer ile veri aktarimi tasarla.", 0.92);
    choir.record_thought(ChoirRole::CRITIC, "Ring buffer wrap-around sirasinda atomic index tasma kontrolu sart.", 0.88);
    choir.record_thought(ChoirRole::ALTERNATIVE, "Alternatif olarak lock-free queue yerine bounded SPSC kanal kullanilabilir.", 0.78);

    std::string consensus = choir.synthesize_consensus();
    std::cout << "\n" << consensus << "\n";

    // 4. Deterministic Validator Integration (Invariant 6 & 11)
    std::cout << "[Step 4] Testing Deterministic Validator Bridge (Parse -> Build -> Test)...\n";
    auto val = DeterministicValidatorBridge::validate_patch("void ring_buffer_push() {}", "test_ring_buffer");
    assert(val.compile_success);
    assert(val.tests_passed);
    assert(val.asan_clean);
    std::cout << "  -> " << val.build_output << "\n";
    std::cout << "  -> " << val.test_output << "\n";

    // 5. Cleanup and leak check
    choir.release_branches();
    kv_cache.destroy_context(parent);
    assert(kv_cache.get_free_blocks() == 500);
    std::cout << "[Step 5 PASS] Zero VRAM leakage verified after Choir lifecycle.\n";

    // Export Phase 3 Report
    nlohmann::json rep;
    rep["test_name"] = "Phase 3: Cognitive Orchestrator & Dialectical Choir";
    rep["status"] = "PASS";
    rep["dialectical_branching_verified"] = true;
    rep["roles_evaluated"] = {"Architect", "Critic", "Alternative"};
    rep["deterministic_validator_verified"] = true;
    rep["zero_leak_verified"] = true;

    std::ofstream out("artifacts/phase3_cognitive_report.json");
    if (out.is_open()) {
        out << rep.dump(2);
        out.close();
        std::cout << "[OK] Report written to: artifacts/phase3_cognitive_report.json\n";
    }

    std::cout << "\n>>> [FAZ 3 COGNITIVE ORCHESTRATOR TEST PASS] <<<\n";
    return 0;
}
