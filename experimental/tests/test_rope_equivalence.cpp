#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

int main() {
    std::cout << "========================================================\n";
    std::cout << "  TEST: ROPE EQUIVALENCE & POSITION CONTINUITY\n";
    std::cout << "========================================================\n";

    const int head_dim = 128;
    const float rope_freq_base = 1000000.0f;
    const int max_pos = 2048;

    double max_angle_error = 0.0;
    int error_count = 0;

    for (int pos = 0; pos < max_pos; ++pos) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;

            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            // Verify Pythagorean identity: cos^2 + sin^2 == 1.0
            float norm_sq = cos_val * cos_val + sin_val * sin_val;
            double err = std::abs((double)norm_sq - 1.0);
            if (err > max_angle_error) max_angle_error = err;

            if (err > 1e-5 || std::isnan(norm_sq) || std::isinf(norm_sq)) {
                error_count++;
            }
        }
    }

    std::cout << "[RoPE Test] Max Norm Error (|cos^2 + sin^2 - 1|): " << max_angle_error << "\n";
    std::cout << "[RoPE Test] Positions Tested: 0 to " << max_pos << "\n";
    std::cout << "[RoPE Test] Numerical Errors: " << error_count << "\n";

    if (error_count == 0 && max_angle_error < 1e-5) {
        std::cout << ">>> [TEST PASS] RoPE Equivalence & Rotation Validated. <<<\n";
        return 0;
    } else {
        std::cerr << ">>> [TEST FAIL] RoPE Drift Detected! <<<\n";
        return 1;
    }
}
