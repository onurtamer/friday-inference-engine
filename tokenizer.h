#ifndef FRIDAY_TOKENIZER_H
#define FRIDAY_TOKENIZER_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <algorithm> // For std::remove_if

class Tokenizer {
public:
    // Constructor: Loads vocabulary from a tokenizer configuration file.
    explicit Tokenizer(const std::string& config_path);

    // Encodes a given text string into a vector of token IDs.
    std::vector<int> encode(const std::string& text) const;

    // Decodes a vector of token IDs back into a human-readable string.
    std::string decode(const std::vector<int>& tokens) const;

private:
    std::map<int, std::string> id_to_token_;     // Maps token IDs to their string representations.
    std::map<std::string, int> token_to_id_;     // Maps string tokens to their integer IDs.
    int unk_token_id_ = -1;                     // ID for unknown tokens.

    // Helper function to parse a simple JSON-like configuration file.
    // This is a minimal parser, not robust for complex JSON.
    void parseConfigFile(const std::string& config_path);

    // Helper to extract key and value from a JSON-like line (e.g., \"\"key\": value,\" or \"\"key\": \"value\",\")
    bool extractKeyValue(const std::string& line, std::string& key, std::string& value);
};

#endif // FRIDAY_TOKENIZER_H
