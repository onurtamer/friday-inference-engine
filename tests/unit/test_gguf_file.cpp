// GGUF reader vs. gguf-py on a tiny file, and rejection of malformed files.
// Usage: test_gguf_file <fixture_dir>   (written by make_gguf_fixtures.py)
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "check.h"
#include "gguf_file.h"
#include "model_config.h"

using friday::GgufFile;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_gguf_file <fixture_dir>\n"); return 2; }
    const std::string dir = argv[1];
    std::ifstream ef(dir + "/expected.json");
    if (!ef) { std::fprintf(stderr, "missing expected.json\n"); return 2; }
    const auto exp = nlohmann::json::parse(ef);

    const GgufFile g(dir + "/tiny.gguf");
    CHECK(g.version() == exp["version"].get<uint32_t>());
    CHECK(g.alignment() == exp["alignment"].get<uint64_t>());
    CHECK(g.dataOffset() == exp["data_offset"].get<uint64_t>());
    CHECK(g.architecture() == "qwen2");
    CHECK(g.getString("general.name") == "tiny-test");
    CHECK(g.getUInt("qwen2.block_count") == 2);
    CHECK(g.getUInt("test.u64") == (1ull << 40));
    CHECK(g.getFloat("test.i32") == -7.0);
    CHECK(std::fabs(g.getFloat("qwen2.attention.layer_norm_rms_epsilon") - 1e-6) < 1e-12);
    CHECK(g.getFloat("test.f64") == 0.125);
    CHECK(g.metadata().at("test.bool").b);
    const auto& toks = g.getStringArray("tokenizer.ggml.tokens");
    CHECK(toks.size() == 4 && toks[2] == "<|x|>" && toks[3] == "\xc4\x9f" "z");
    const auto& ints = g.getIntArray("test.ints");
    CHECK(ints.size() == 3 && ints[1] == -2);
    CHECK(g.getUInt("missing.key", 5) == 5);
    bool threw = false;
    try { g.getUInt("general.name"); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);

    const auto& et = exp["tensors"];
    CHECK(g.tensors().size() == et.size());
    for (size_t i = 0; i < et.size() && i < g.tensors().size(); ++i) {
        const auto& t = g.tensors()[i];
        CHECK(t.name == et[i]["name"].get<std::string>());
        CHECK(t.type == et[i]["type"].get<uint32_t>());
        CHECK(t.offset == et[i]["offset"].get<uint64_t>());
        CHECK(t.size == et[i]["size"].get<uint64_t>());
        CHECK(t.ne == et[i]["ne"].get<std::vector<int64_t>>());
    }
    CHECK(g.findTensor("blk.0.ffn_up.weight") != nullptr && g.findTensor("nope") == nullptr);
    CHECK(std::string(friday::ggmlTypeName(friday::GGML_Q2_K)) == "Q2_K");

    // ModelConfig: missing required keys must throw, not default silently.
    threw = false;
    try { friday::ModelConfig::fromGguf(g); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);  // tiny.gguf has no embedding_length / head_count

    int rejected = 0, total = 0;
    for (const char* bad : {"bad_magic", "bad_version", "bad_kv_count", "bad_trunc_0", "bad_trunc_1", "bad_trunc_2",
                            "bad_trunc_3", "bad_trunc_4", "bad_trunc_5"}) {
        ++total;
        try {
            GgufFile b(dir + "/" + bad + ".gguf");
            std::printf("NOT rejected: %s\n", bad);
        } catch (const std::runtime_error& e) {
            ++rejected;
            std::printf("rejected %-13s %s\n", bad, e.what());
        }
    }
    CHECK(rejected == total);
    return CHECK_RESULT();
}
