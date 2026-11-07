#pragma once

#include "friday/runtime/paged_kv_cache.h"
#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace friday {
namespace cognition {

enum class ChoirRole {
    ARCHITECT,
    CRITIC,
    ALTERNATIVE,
    SYNTHESIZER
};

struct DialecticalThoughtBranch {
    ChoirRole role;
    std::string role_name;
    friday::runtime::ContextId context_id;
    std::vector<int> generated_tokens;
    std::string thought_text;
    double confidence_score = 0.0;
    bool approved = false;
};

class DialecticalChoir {
public:
    DialecticalChoir(friday::runtime::PagedKvCache* kv_cache);
    ~DialecticalChoir() = default;

    // Fork Architect, Critic, and Alternative from common parent prompt context
    void branch_choir(friday::runtime::ContextId parent_context);

    // Record thought generation from each role
    void record_thought(ChoirRole role, const std::string& thought, double confidence);

    // Synthesize dialectical branches into consensus
    std::string synthesize_consensus();

    const std::vector<DialecticalThoughtBranch>& get_branches() const { return branches_; }
    void release_branches();

private:
    friday::runtime::PagedKvCache* kv_cache_;
    std::vector<DialecticalThoughtBranch> branches_;
};

} // namespace cognition
} // namespace friday
