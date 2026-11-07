#pragma once

#include <string>
#include <vector>

namespace friday {
namespace cognition {

struct ValidationResult {
    bool compile_success = false;
    bool tests_passed = false;
    bool asan_clean = false;
    std::string build_output;
    std::string test_output;
    int exit_code = 0;
};

class DeterministicValidatorBridge {
public:
    static ValidationResult validate_patch(const std::string& source_code, const std::string& test_name);
};

} // namespace cognition
} // namespace friday
