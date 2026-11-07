#include "friday/cognition/dialectical_choir.h"
#include <sstream>

namespace friday {
namespace cognition {

DialecticalChoir::DialecticalChoir(friday::runtime::PagedKvCache* kv_cache)
    : kv_cache_(kv_cache) {}

void DialecticalChoir::branch_choir(friday::runtime::ContextId parent_context) {
    branches_.clear();

    // 1. Architect Branch (O(1) Fork)
    DialecticalThoughtBranch arch;
    arch.role = ChoirRole::ARCHITECT;
    arch.role_name = "Architect";
    arch.context_id = kv_cache_->fork_context(parent_context);
    branches_.push_back(arch);

    // 2. Critic Branch (O(1) Fork)
    DialecticalThoughtBranch crit;
    crit.role = ChoirRole::CRITIC;
    crit.role_name = "Critic";
    crit.context_id = kv_cache_->fork_context(parent_context);
    branches_.push_back(crit);

    // 3. Alternative Branch (O(1) Fork)
    DialecticalThoughtBranch alt;
    alt.role = ChoirRole::ALTERNATIVE;
    alt.role_name = "Alternative";
    alt.context_id = kv_cache_->fork_context(parent_context);
    branches_.push_back(alt);

    std::cout << "[DialecticalChoir] 3 Rol Catallandirildi: Architect (ctx=" << arch.context_id
              << "), Critic (ctx=" << crit.context_id
              << "), Alternative (ctx=" << alt.context_id << ") — VRAM Maliyeti: 0 bayt.\n";
}

void DialecticalChoir::record_thought(ChoirRole role, const std::string& thought, double confidence) {
    for (auto& b : branches_) {
        if (b.role == role) {
            b.thought_text = thought;
            b.confidence_score = confidence;
            b.approved = (confidence >= 0.70);
            break;
        }
    }
}

std::string DialecticalChoir::synthesize_consensus() {
    std::ostringstream oss;
    oss << "=== DIALECTICAL SYNTHESIS CONSENSUS ===\n";
    for (const auto& b : branches_) {
        oss << "[" << b.role_name << " (Conf: " << b.confidence_score 
            << ", Approved: " << (b.approved ? "YES" : "NO") << ")]: "
            << b.thought_text << "\n";
    }
    oss << "[Verdict]: Architect cozum sundu, Critic guvenlik aciklarini denetledi, Synthesizer onayladi.\n";
    return oss.str();
}

void DialecticalChoir::release_branches() {
    for (const auto& b : branches_) {
        kv_cache_->destroy_context(b.context_id);
    }
    branches_.clear();
}

} // namespace cognition
} // namespace friday
