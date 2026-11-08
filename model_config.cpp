#include "model_config.h"

#include <cmath>
#include <stdexcept>

#include "gguf_file.h"

namespace friday {

namespace {
// general.file_type (llama_ftype) -> name, for the common types only.
const char* fileTypeName(uint64_t ft) {
    switch (ft) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 7: return "Q8_0";
        case 8: return "Q5_0";
        case 9: return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        default: return "";
    }
}
}  // namespace

ModelConfig ModelConfig::fromGguf(const GgufFile& f) {
    ModelConfig c;
    c.architecture = f.architecture();
    c.name = f.getString("general.name", "");
    const std::string a = c.architecture + ".";
    c.num_layers = (int)f.getUInt(a + "block_count");
    c.hidden_dim = (int)f.getUInt(a + "embedding_length");
    c.num_heads = (int)f.getUInt(a + "attention.head_count");
    c.num_kv_heads = (int)f.getUInt(a + "attention.head_count_kv", (uint64_t)c.num_heads);
    c.ffn_dim = (int)f.getUInt(a + "feed_forward_length", 0);
    c.context_length = (int)f.getUInt(a + "context_length", 0);
    c.rms_norm_epsilon = (float)f.getFloat(a + "attention.layer_norm_rms_epsilon");
    c.rope_freq_base = (float)f.getFloat(a + "rope.freq_base", 10000.0);
    if (c.num_heads <= 0 || c.hidden_dim <= 0 || c.num_layers <= 0 || c.num_kv_heads <= 0)
        throw std::runtime_error("GGUF: invalid model dimensions");
    c.head_dim = (int)f.getUInt(a + "attention.key_length", (uint64_t)(c.hidden_dim / c.num_heads));
    if (f.has("tokenizer.ggml.tokens")) c.vocab_size = (int)f.getStringArray("tokenizer.ggml.tokens").size();
    c.file_type = fileTypeName(f.getUInt("general.file_type", 999));
    return c;
}

void ModelConfig::checkSupported() const {
    if (architecture != "qwen2" && architecture != "llama")
        throw std::runtime_error("unsupported architecture '" + architecture + "' (supported: qwen2, llama)");
    if (head_dim != 128 || head_dim * num_heads != hidden_dim)
        throw std::runtime_error("unsupported head layout: head_dim " + std::to_string(head_dim) + " (need 128)");
    if (num_heads % num_kv_heads != 0) throw std::runtime_error("num_heads must be a multiple of num_kv_heads");
    if (!(rope_freq_base > 1.0f)) throw std::runtime_error("invalid rope.freq_base");
}

}  // namespace friday
