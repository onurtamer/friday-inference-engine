#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "unicode_tables.h"

namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------
struct CodePoint {
    uint32_t cp;     // 0xFFFFFFFF for an invalid byte
    size_t offset;   // byte offset in the source string
    size_t len;      // byte length
};

std::vector<CodePoint> decodeUtf8(const std::string& s) {
    std::vector<CodePoint> out;
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        unsigned char c = p[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        bool ok = len > 0 && i + len <= n;
        for (size_t k = 1; ok && k < len; ++k) ok = (p[i + k] & 0xC0) == 0x80;
        if (!ok) { out.push_back({0xFFFFFFFFu, i, 1}); ++i; continue; }
        uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
        for (size_t k = 1; k < len; ++k) cp = (cp << 6) | (p[i + k] & 0x3F);
        out.push_back({cp, i, len});
        i += len;
    }
    return out;
}

std::string encodeUtf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    return s;
}

template <size_t N>
bool inRanges(const uint32_t (&r)[N][2], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

bool isLetter(uint32_t cp) { return cp != 0xFFFFFFFFu && inRanges(kUnicodeLetterRanges, cp); }
bool isNumber(uint32_t cp) { return cp != 0xFFFFFFFFu && inRanges(kUnicodeNumberRanges, cp); }
bool isNewline(uint32_t cp) { return cp == '\r' || cp == '\n'; }
// Unicode White_Space (matches \s of the regex engines used by HF tokenizers).
bool isSpace(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}
uint32_t asciiLower(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp; }

}  // namespace

// ---------------------------------------------------------------------------
// Qwen2 pre-tokenizer. Implements, at each position, the first matching
// alternative of:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//   | ' '?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// ---------------------------------------------------------------------------
std::vector<std::string> Tokenizer::preTokenize(const std::string& text) {
    const std::vector<CodePoint> cps = decodeUtf8(text);
    const size_t n = cps.size();
    std::vector<std::string> out;
    size_t i = 0;
    auto emit = [&](size_t a, size_t b) {  // codepoint range [a, b)
        const size_t start = cps[a].offset;
        const size_t end = cps[b - 1].offset + cps[b - 1].len;
        out.push_back(text.substr(start, end - start));
    };
    while (i < n) {
        const uint32_t c = cps[i].cp;
        size_t end = 0;

        // 1. contractions
        if (c == '\'' && i + 1 < n) {
            const uint32_t a = asciiLower(cps[i + 1].cp);
            const uint32_t b = (i + 2 < n) ? asciiLower(cps[i + 2].cp) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') end = i + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) end = i + 3;
        }
        // 2. optional non-letter/number/newline prefix + letters
        if (!end) {
            size_t j = i;
            if (!isNewline(c) && !isLetter(c) && !isNumber(c) && i + 1 < n && isLetter(cps[i + 1].cp)) j = i + 1;
            if (isLetter(cps[j].cp)) {
                while (j < n && isLetter(cps[j].cp)) ++j;
                end = j;
            }
        }
        // 3. single number
        if (!end && isNumber(c)) end = i + 1;
        // 4. optional space + punctuation/symbols + trailing newlines
        if (!end) {
            size_t j = (c == ' ') ? i + 1 : i;
            size_t k = j;
            while (k < n && !isSpace(cps[k].cp) && !isLetter(cps[k].cp) && !isNumber(cps[k].cp)) ++k;
            if (k > j) {
                while (k < n && isNewline(cps[k].cp)) ++k;
                end = k;
            }
        }
        // 5-7. whitespace
        if (!end && isSpace(c)) {
            size_t e = i;
            while (e < n && isSpace(cps[e].cp)) ++e;
            size_t last_nl = SIZE_MAX;
            for (size_t k = i; k < e; ++k) if (isNewline(cps[k].cp)) last_nl = k;
            if (last_nl != SIZE_MAX) end = last_nl + 1;          // \s*[\r\n]+
            else if (e == n) end = e;                            // \s+(?!\S) at end of text
            else if (e - i >= 2) end = e - 1;                    // \s+(?!\S): leave one space
            else end = e;                                        // \s+
        }
        if (!end) end = i + 1;  // unreachable in practice; never stall
        emit(i, end);
        i = end;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
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
    if (j.contains("unk_token_id") && j["unk_token_id"].is_number_integer()) unk_token_id_ = j["unk_token_id"];
    if (j.contains("eos_token_id") && j["eos_token_id"].is_number_integer()) eos_token_id_ = j["eos_token_id"];
    if (j.contains("id_to_token")) {
        for (auto& it : j["id_to_token"].items()) id_to_token_[std::stoi(it.key())] = it.value().get<std::string>();
    }
    if (j.contains("token_to_id")) {
        for (auto& it : j["token_to_id"].items()) token_to_id_[it.key()] = it.value().get<int>();
    }
    if (token_to_id_.empty()) {
        for (const auto& p : id_to_token_) token_to_id_[p.second] = p.first;
    }
    if (j.contains("merges")) {
        int rank = 0;
        for (const auto& m : j["merges"]) merge_ranks_[m.get<std::string>()] = rank++;
    }

    // Special tokens: explicit list, or every vocab entry of the form <|...|>.
    if (j.contains("special_tokens")) {
        for (const auto& s : j["special_tokens"]) {
            const std::string t = s.get<std::string>();
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end()) special_tokens_.push_back({t, it->second});
        }
    } else {
        for (const auto& p : token_to_id_) {
            const std::string& t = p.first;
            if (t.size() > 4 && t.compare(0, 2, "<|") == 0 && t.compare(t.size() - 2, 2, "|>") == 0)
                special_tokens_.push_back({t, p.second});
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    for (const auto& s : special_tokens_) special_ids_.insert(s.second);

    // GPT-2 byte <-> unicode alphabet.
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        const uint32_t cp = printable ? (uint32_t)b : (uint32_t)(256 + extra++);
        byte_encoder_[b] = encodeUtf8(cp);
        byte_decoder_[byte_encoder_[b]] = (unsigned char)b;
    }

    if (unk_token_id_ == -1) {
        auto it = token_to_id_.find("<unk>");
        if (it != token_to_id_.end()) unk_token_id_ = it->second;
    }
    if (merge_ranks_.empty()) {
        std::cerr << "Warning: tokenizer config has no merges; using the legacy greedy encoder, "
                     "which does not match the reference tokenizer.\n";
    }
    std::cout << "Tokenizer config loaded from " << config_path << ". Vocab size: " << id_to_token_.size()
              << ", merges: " << merge_ranks_.size() << ", special tokens: " << special_tokens_.size() << std::endl;
}

Tokenizer::Tokenizer(const std::string& config_path) {
    parseConfigFile(config_path);
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------
void Tokenizer::encodeBpe(const std::string& piece, std::vector<int>& out) const {
    std::vector<std::string> sym;
    sym.reserve(piece.size());
    for (unsigned char b : piece) sym.push_back(byte_encoder_[b]);

    while (sym.size() > 1) {
        int best_rank = INT_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merge_ranks_.find(sym[i] + " " + sym[i + 1]);
            if (it != merge_ranks_.end() && it->second < best_rank) { best_rank = it->second; best_i = i; }
        }
        if (best_rank == INT_MAX) break;
        // Merge every occurrence of the best pair, left to right.
        const std::string left = sym[best_i], right = sym[best_i + 1];
        std::vector<std::string> merged;
        merged.reserve(sym.size());
        for (size_t i = 0; i < sym.size();) {
            if (i + 1 < sym.size() && sym[i] == left && sym[i + 1] == right) { merged.push_back(left + right); i += 2; }
            else merged.push_back(sym[i++]);
        }
        sym.swap(merged);
    }
    for (const auto& s : sym) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) out.push_back(it->second);
        else if (unk_token_id_ >= 0) out.push_back(unk_token_id_);
    }
}

// Legacy encoder kept for configs without merges (greedy longest match, only
// space/newline/tab mapped to the byte alphabet).
void Tokenizer::encodeGreedyLegacy(const std::string& segment, std::vector<int>& out) const {
    std::string mapped;
    for (char c : segment) {
        if (c == ' ') mapped += "\xc4\xa0";
        else if (c == '\n') mapped += "\xc4\x8a";
        else if (c == '\t') mapped += "\xc4\x89";
        else mapped += c;
    }
    size_t i = 0;
    while (i < mapped.size()) {
        size_t best_len = 0;
        int best_id = unk_token_id_;
        const size_t max_look = std::min((size_t)64, mapped.size() - i);
        for (size_t j = 1; j <= max_look; ++j) {
            auto it = token_to_id_.find(mapped.substr(i, j));
            if (it != token_to_id_.end()) { best_len = j; best_id = it->second; }
        }
        out.push_back(best_id);
        i += best_len ? best_len : 1;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    size_t i = 0;
    while (i < text.size()) {
        // Find the earliest special token at or after i (longest wins on ties).
        size_t next = text.size();
        const std::pair<std::string, int>* hit = nullptr;
        for (const auto& s : special_tokens_) {
            const size_t p = text.find(s.first, i);
            if (p != std::string::npos && p < next) { next = p; hit = &s; }
        }
        if (next > i) {
            const std::string segment = text.substr(i, next - i);
            if (merge_ranks_.empty()) encodeGreedyLegacy(segment, tokens);
            else for (const auto& piece : preTokenize(segment)) encodeBpe(piece, tokens);
        }
        if (hit) { tokens.push_back(hit->second); i = next + hit->first.size(); }
        else i = next;
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    for (int id : tokens) {
        auto it = id_to_token_.find(id);
        if (it == id_to_token_.end()) {
            out += "[UNK]";
            std::cerr << "Warning: Unknown token ID encountered during decoding: " << id << std::endl;
            continue;
        }
        if (special_ids_.count(id)) { out += it->second; continue; }
        for (const auto& c : decodeUtf8(it->second)) {
            const std::string ch = it->second.substr(c.offset, c.len);
            auto b = byte_decoder_.find(ch);
            if (b != byte_decoder_.end()) out += (char)b->second;
            else out += ch;
        }
    }
    return out;
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
