#ifndef FRIDAY_TOKENIZER_H
#define FRIDAY_TOKENIZER_H

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Byte-level BPE tokenizer (GPT-2 / Qwen2 style).
//
// encode(): split on special tokens (<|...|>), pre-tokenize with the Qwen2
// pattern, map bytes to the GPT-2 byte alphabet, then apply merges by rank.
// Configs without "merges" fall back to the legacy greedy longest-match
// encoder (with a warning); that path does not match the reference tokenizer.
// Not implemented: Unicode NFC normalization of the input (input is assumed
// to be NFC already).
namespace friday { class GgufFile; }

class Tokenizer {
public:
    explicit Tokenizer(const std::string& config_path);
    explicit Tokenizer(const friday::GgufFile& gguf);   // tokenizer.ggml.* metadata

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    int getVocabSize() const { return (int)id_to_token_.size(); }
    int getUnkTokenId() const { return unk_token_id_; }
    int getEosTokenId() const { return eos_token_id_; }
    bool hasMerges() const { return !merge_ranks_.empty(); }

    // Qwen special token IDs
    static constexpr int IM_START_ID = 151644;
    static constexpr int IM_END_ID   = 151645;

    // Applies Qwen ChatML template around the user's raw prompt
    std::string applyChatTemplate(const std::string& prompt) const;

    // Qwen2 pre-tokenizer: splits UTF-8 text into pre-tokens (exposed for tests).
    // max_digits: 1 for qwen2, 3 for llama-bpe.
    static std::vector<std::string> preTokenize(const std::string& text, int max_digits = 1);
    int getBosTokenId() const { return bos_token_id_; }
    bool addsBos() const { return add_bos_ && bos_token_id_ >= 0; }  // GGUF tokenizer.ggml.add_bos_token

    void parseConfigFile(const std::string& config_path);

private:
    void finishInit(const std::string& source);
    void encodeBpe(const std::string& piece, std::vector<int>& out) const;
    void encodeGreedyLegacy(const std::string& segment, std::vector<int>& out) const;

    int max_digits_ = 1;
    int bos_token_id_ = -1;
    bool add_bos_ = false;
    bool ignore_merges_ = false;  // llama-bpe: a pre-token that is a whole vocab entry is used as is
    std::map<int, std::string> id_to_token_;
    std::map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merge_ranks_;          // "left right" -> rank
    std::vector<std::pair<std::string, int>> special_tokens_;   // longest first
    std::unordered_set<int> special_ids_;
    std::string byte_encoder_[256];
    std::unordered_map<std::string, unsigned char> byte_decoder_;
    int unk_token_id_ = -1;
    int eos_token_id_ = -1;
};

#endif // FRIDAY_TOKENIZER_H
