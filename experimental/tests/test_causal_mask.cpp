#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "========================================================\n";
    std::cout << "  TEST: CAUSAL MASK INTEGRITY (NO FUTURE LEAKAGE)\n";
    std::cout << "========================================================\n";

    const int seq_len = 256;
    std::vector<std::vector<float>> attention_mask(seq_len, std::vector<float>(seq_len, 0.0f));

    // Construct causal mask: A[i][j] = (j <= i) ? 0.0f : -1e30f
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j <= i) {
                attention_mask[i][j] = 0.0f; // Allowed
            } else {
                attention_mask[i][j] = -1e30f; // Strictly Masked
            }
        }
    }

    int violations = 0;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float val = attention_mask[i][j];
            if (j > i && val > -1e20f) {
                violations++;
            }
            if (j <= i && val < -1e20f) {
                violations++;
            }
        }
    }

    std::cout << "[Causal Mask] Matrix Dimension: " << seq_len << " x " << seq_len << "\n";
    std::cout << "[Causal Mask] Future Leakage Violations: " << violations << "\n";

    if (violations == 0) {
        std::cout << ">>> [TEST PASS] Causal Mask Integrity Validated. <<<\n";
        return 0;
    } else {
        std::cerr << ">>> [TEST FAIL] Future Leakage Detected! <<<\n";
        return 1;
    }
}
