#ifndef FRIDAY_MODEL_CONFIG_H
#define FRIDAY_MODEL_CONFIG_H

// Model hyper-parameters read from GGUF metadata ("<arch>.block_count", ...).

#include <string>

namespace friday {

class GgufFile;

struct ModelConfig {
    std::string architecture;   // general.architecture, e.g. "qwen2"
    std::string name;           // general.name (may be empty)
    int num_layers = 0;
    int hidden_dim = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int ffn_dim = 0;
    int context_length = 0;
    int vocab_size = 0;
    float rms_norm_epsilon = 0.0f;
    float rope_freq_base = 10000.0f;
    std::string file_type;      // dominant quantization ("Q2_K", ...) from general.file_type, if known

    static ModelConfig fromGguf(const GgufFile& f);
    // Throws std::runtime_error if this engine cannot run the model yet
    // (architecture, RoPE settings, head dim), with a message saying why.
    void checkSupported() const;
};

}  // namespace friday

#endif  // FRIDAY_MODEL_CONFIG_H
