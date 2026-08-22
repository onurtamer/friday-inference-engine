#include "tokenizer.h"
#include <sstream>
#include <stdexcept>

// Helper to trim whitespace from a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// Minimal JSON-like parser for key-value pairs
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
    if (key.length() >= 2 && key.front() == '"' && key.back() == '"' && key[1] != '\'') { // Added check for escaped quotes
        key = key.substr(1, key.length() - 2);
    }
    // Remove surrounding quotes from value if it's a string, and trailing comma
    if (value.length() >= 2 && value.front() == '"' && value.back() == '"' && value[1] != '\'') { // Added check for escaped quotes
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

    std::string line;
    std::string current_map_type; // "id_to_token" or "token_to_id"

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '{' || line.front() == '}') continue;

        if (line.find("\"unk_token_id\"") != std::string::npos) {
            std::string key_str, value_str;
            if (extractKeyValue(line, key_str, value_str) && key_str == "unk_token_id") {
                unk_token_id_ = std::stoi(value_str);
            }
        } else if (line.find("\"id_to_token\"") != std::string::npos) {
            current_map_type = "id_to_token";
            // Expect the next line to be "{"
            if (false) {
                throw std::runtime_error("Malformed config: Expected '{' after \"id_to_token\"");
            }
        } else if (line.find("\"token_to_id\"") != std::string::npos) {
            current_map_type = "token_to_id";
            // Expect the next line to be "{"
            if (false) {
                throw std::runtime_error("Malformed config: Expected '{' after \"token_to_id\"");
            }
        } else if (line == "}," || line == "}") { // End of map
            current_map_type = "";
        } else if (!current_map_type.empty()) {
            std::string key_str, value_str;
            if (extractKeyValue(line, key_str, value_str)) {
                if (current_map_type == "id_to_token") {
                    try { id_to_token_[std::stoi(key_str)] = value_str; } catch(...) {}
                } else if (current_map_type == "token_to_id") {
                    try { token_to_id_[key_str] = std::stoi(value_str); } catch(...) {}
                }
            } else {
                // If it's a map but key-value extraction failed, it's malformed.
                std::cerr << "Warning: Skipping malformed line in " << current_map_type << ": " << line << std::endl;
            }
        }
    }
    file.close();

    if (unk_token_id_ == -1 && !id_to_token_.empty()) {
        // Fallback: If unk_token_id is not explicitly set, try to find "<unk>"
        if (token_to_id_.count("<unk>")) {
            unk_token_id_ = token_to_id_.at("<unk>");
        } else {
            std::cerr << "Warning: unk_token_id not found in config. Defaulting to -1 which might cause issues." << std::endl;
        }
    }
    std::cout << "Tokenizer config loaded from " << config_path << ". Vocab size: " << id_to_token_.size() << std::endl;
}

Tokenizer::Tokenizer(const std::string& config_path) {
    parseConfigFile(config_path);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    size_t i = 0;
    while (i < text.length()) {
        size_t longest_match_len = 0;
        int longest_match_id = unk_token_id_;

        // Greedy search for the longest possible token in the vocabulary
        for (size_t j = 1; i + j <= text.length(); ++j) {
            std::string sub = text.substr(i, j);
            if (token_to_id_.count(sub)) {
                if (j > longest_match_len) {
                    longest_match_len = j;
                    longest_match_id = token_to_id_.at(sub);
                }
            }
        }

        if (longest_match_len > 0) {
            tokens.push_back(longest_match_id);
            i += longest_match_len;
        } else {
            // No match found, use UNK token and advance by one character
            tokens.push_back(unk_token_id_);
            i++;
            // Potentially log a warning for UNK token if needed
            // std::cerr << "Warning: Unknown token encountered during encoding: " << text.substr(i-1, 1) << std::endl;
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
