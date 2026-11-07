// Engine tokenizer vs. reference Qwen2 tokenizer ids (CPU only; needs the model's
// tokenizer_config.json with merges). Also checks decode(encode(text)) == text.
// Usage: test_tokenizer_qwen <tokenizer_config.json> <qwen2_encode_reference.json>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "check.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: test_tokenizer_qwen <config> <reference>\n"); return 2; }
    Tokenizer tok(argv[1]);
    CHECK(tok.hasMerges());
    std::ifstream f(argv[2]);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    const auto ref = nlohmann::json::parse(f);
    CHECK(tok.getVocabSize() == ref["vocab_size"].get<int>());
    int n = 0, exact = 0;
    for (const auto& c : ref["cases"]) {
        const std::string text = c["text"];
        const auto expected = c["ids"].get<std::vector<int>>();
        const auto got = tok.encode(text);
        ++n;
        if (got == expected) ++exact;
        else {
            size_t k = 0;
            while (k < got.size() && k < expected.size() && got[k] == expected[k]) ++k;
            std::fprintf(stderr, "case %d: first difference at token %zu (got %zu ids, expected %zu)\n",
                         n, k, got.size(), expected.size());
        }
        CHECK(got == expected);
        CHECK(tok.decode(expected) == text);
    }
    std::printf("encode exact matches: %d / %d\n", exact, n);
    return CHECK_RESULT();
}
