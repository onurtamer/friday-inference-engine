#include "tokenizer.h"
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp> // JSON desteği eklendi

// Helper to trim whitespace from a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \\t\\n\\r");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \\t\\n\\r");
    return str.substr(first, (last - first + 1));
}

// Minimal JSON-like parser for key-value pairs (replaced with nlohmann::json)
bool Tokenizer::extractKeyValue(const std::string& line, std::string& key, std::string& value) {
    size_t colon_pos = line.find(':');
    if (colon_pos == std::string::npos) {
        return false;
    }
    key = line.substr(0, colon_pos);
    value = line.substr(colon_pos + 1);

    key = trim(key);
    value = trim(value);

    // Remove surrounding quotes from key
    if (key.length() >= 2 && key.front() == '\"' && key.back() == '\"' && key[1] != '\\') {
        key = key.substr(1, key.length() - 2);
    }
    // Remove surrounding quotes from value if it's a string, and trailing comma
    if (value.length() >= 2 && value.front() == '\"' && value.back() == '\"' && value[1] != '\\') {
        value = value.substr(1, value.length() - 2);
    } else if (value.length() >= 1 && value.back() == ',') { // For integer values with trailing comma
        value = value.substr(0, value.length() - 1);
    }
    return true;
}

void Tokenizer::parseConfigFile(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open tokenizer config file: " + config_path);
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Failed to parse tokenizer config file: " + std::string(e.what()));
    }

    std::string current_map_type = ""; // "id_to_token" or "token_to_id"
    // unk_token_id_ and eos_token_id_ refer to class members implicitly

    for (auto& item : j.items()) {
        std::string key = item.key();
        if (key == "unk_token_id") {
            this->unk_token_id_ = item.value().get<int>();
        } else if (key == "eos_token_id") {
            this->eos_token_id_ = item.value().get<int>();
        } else if (key == "id_to_token") {
            current_map_type = "id_to_token";
            for (auto& token_item : item.value().items()) { // changed from item to item.value().items()
                std::string id_str = token_item.key();
                std::string token_str = token_item.value().get<std::string>();
                id_to_token_[std::stoi(id_str)] = token_str;
            }
        } else if (key == "token_to_id") {
            current_map_type = "token_to_id";
            for (auto& token_item : item.value().items()) { // changed from item to item.value().items()
                std::string token_str = token_item.key();
                int token_id = token_item.value().get<int>();
                token_to_id_[token_str] = token_id;
            }
        }
    }

    // Fallback for unk_token_id if not explicitly set
    if (this->unk_token_id_ == -1 && !id_to_token_.empty()) {
        if (token_to_id_.count("<s>")) {
            this->unk_token_id_ = token_to_id_["<s>"];
        } else if (token_to_id_.count("<<UNK>>")) {
            this->unk_token_id_ = token_to_id_["<<UNK>>"];
        } else if (token_to_id_.count("unknown")) {
            this->unk_token_id_ = token_to_id_["unknown"];
        } else if (!id_to_token_.empty()) {
            // If unk_token_id is not explicitly set, try to find "unk_token_id" in the map keys
            for (const auto& pair : id_to_token_) {
                if (pair.second.find("unk") != std::string::npos || pair.second.find("unknown") != std::string::npos) {
                    this->unk_token_id_ = pair.first;
                    break;
                }
            }
        }

        if (unk_token_id_ == -1) {
            std::cerr << "Warning: unk_token_id not found in config. Defaulting to -1 which might cause issues." << std::endl;
        }

        std::cout << "Tokenizer config loaded from " << config_path << ". Vocab size: " << id_to_token_.size() << std::endl;
    }
}

Tokenizer::Tokenizer(const std::string& config_path) {
    parseConfigFile(config_path);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    size_t i = 0;

    const std::string IM_START_STR = "<|im_start|>";
    const std::string IM_END_STR   = "<|im_end|>";
    const std::string ENDOFTEXT_STR= "<|endoftext|>";

    while (i < text.length()) {
        // 1. Check for Qwen special tokens first
        if (text.compare(i, IM_START_STR.length(), IM_START_STR) == 0) {
            tokens.push_back(IM_START_ID);
            i += IM_START_STR.length();
            continue;
        }
        if (text.compare(i, IM_END_STR.length(), IM_END_STR) == 0) {
            tokens.push_back(IM_END_ID);
            i += IM_END_STR.length();
            continue;
        }
        if (text.compare(i, ENDOFTEXT_STR.length(), ENDOFTEXT_STR) == 0) {
            tokens.push_back(151643);
            i += ENDOFTEXT_STR.length();
            continue;
        }

        // Find next special token to segment normal text
        size_t next_special = text.length();
        size_t p_start = text.find(IM_START_STR, i);
        size_t p_end   = text.find(IM_END_STR, i);
        size_t p_eot   = text.find(ENDOFTEXT_STR, i);
        if (p_start != std::string::npos && p_start < next_special) next_special = p_start;
        if (p_end != std::string::npos && p_end < next_special) next_special = p_end;
        if (p_eot != std::string::npos && p_eot < next_special) next_special = p_eot;

        // Substring to encode
        std::string segment = text.substr(i, next_special - i);
        i = next_special;

        // Map whitespace to BPE unicode characters
        // ' ' -> \xc4\xa0 (Ġ)
        // '\n' -> \xc4\x8a (Ċ)
        // '\t' -> \xc4\x89 (ĉ)
        std::string mapped_segment;
        for (char c : segment) {
            if (c == ' ') {
                mapped_segment += "\xc4\xa0";
            } else if (c == '\n') {
                mapped_segment += "\xc4\x8a";
            } else if (c == '\t') {
                mapped_segment += "\xc4\x89";
            } else {
                mapped_segment += c;
            }
        }

        // Greedy longest match on mapped segment
        size_t seg_i = 0;
        while (seg_i < mapped_segment.length()) {
            size_t longest_match_len = 0;
            int longest_match_id = unk_token_id_;
            size_t max_look = std::min((size_t)64, mapped_segment.length() - seg_i);

            for (size_t j = 1; j <= max_look; ++j) {
                std::string sub = mapped_segment.substr(seg_i, j);
                auto it = token_to_id_.find(sub);
                if (it != token_to_id_.end()) {
                    if (j > longest_match_len) {
                        longest_match_len = j;
                        longest_match_id = it->second;
                    }
                }
            }

            if (longest_match_len > 0) {
                tokens.push_back(longest_match_id);
                seg_i += longest_match_len;
            } else {
                tokens.push_back(unk_token_id_);
                seg_i++;
            }
        }
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::stringstream ss;
    for (int token_id : tokens) {
        if (id_to_token_.count(token_id)) {
            ss << id_to_token_.at(token_id);
        } else {
            // Handle unknown token IDs during decoding, e.g., print a placeholder
            ss << "[UNK]";
            std::cerr << "Warning: Unknown token ID encountered during decoding: " << token_id << std::endl;
        }
    }
    return ss.str();
}

std::string Tokenizer::applyChatTemplate(const std::string& prompt) const {
    // Qwen ChatML format
    std::string templated;
    templated += "<|im_start|>system\nYou are FRIDAY, a helpful AI assistant.<|im_end|>\n";
    templated += "<|im_start|>user\n";
    templated += prompt;
    templated += "<|im_end|>\n";
    templated += "<|im_start|>assistant\n";
    return templated;
}