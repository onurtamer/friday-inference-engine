#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace friday {
namespace runtime {

class IDraftProvider {
public:
    virtual ~IDraftProvider() = default;
    
    // Propose gamma candidate tokens based on current context history
    virtual std::vector<TokenId> propose_draft(ContextId ctx_id, const std::vector<TokenId>& history) = 0;
    
    // Feedback loop to track empirical acceptance rate
    virtual void record_feedback(ContextId ctx_id, int accepted_count) = 0;
    
    virtual double get_acceptance_rate() const = 0;
};

class NoneDraftProvider : public IDraftProvider {
public:
    std::vector<TokenId> propose_draft(ContextId, const std::vector<TokenId>&) override {
        return {};
    }
    void record_feedback(ContextId, int) override {}
    double get_acceptance_rate() const override { return 0.0; }
};

} // namespace runtime
} // namespace friday
