// SHA-256 test vectors and the model registry (add / cache / verify / remove)
// on the gguf fixtures. Usage: test_model_registry <gguf_fixture_dir>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "check.h"
#include "model_registry.h"
#include "sha256.h"

namespace fs = std::filesystem;
using namespace friday;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_model_registry <fixture_dir>\n"); return 2; }
    const fs::path dir = argv[1];

    // FIPS 180-4 / NIST vectors
    CHECK(sha256Hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256Hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(sha256Hex(two.data(), two.size()) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    {
        Sha256 h;  // one million 'a', fed in odd-sized pieces
        const std::string chunk(997, 'a');
        size_t left = 1000000;
        while (left) { const size_t n = left < chunk.size() ? left : chunk.size(); h.update(chunk.data(), n); left -= n; }
        CHECK(h.hexDigest() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    std::ifstream ef(dir / "model_expected.json");
    if (!ef) { std::fprintf(stderr, "missing model_expected.json\n"); return 2; }
    const auto exp = nlohmann::json::parse(ef);
    CHECK(sha256File((dir / "model.gguf").string()) == exp["sha256"].get<std::string>());

    const fs::path reg = dir / "registry_test" / "models.json";
    fs::remove_all(reg.parent_path());
    const fs::path copy = dir / "registry_test_copy.gguf";
    fs::copy_file(dir / "model.gguf", copy, fs::copy_options::overwrite_existing);
    {
        ModelRegistry r(reg.string());
        CHECK(r.entries().empty());
        const ModelEntry& e = r.add((dir / "model.gguf").string(), "", false);
        CHECK(e.id == "model");
        CHECK(e.sha256 == exp["sha256"].get<std::string>());
        CHECK(e.architecture == "qwen2" && e.name == "Tiny Model" && e.license == "apache-2.0");
        CHECK(e.quant == "Q2_K" && e.layers == 2 && e.context_length == 4096);
        CHECK(e.parameters == exp["parameters"].get<uint64_t>());
        CHECK(e.size == exp["size"].get<uint64_t>());
        r.add(copy.string(), "copy", false);
        CHECK(r.entries().size() == 2);
        bool threw = false;
        try { r.add((dir / "tiny.gguf").string(), "copy", false); } catch (const std::exception&) { threw = true; }
        CHECK(threw);  // id in use (or tiny.gguf lacks model keys): must not add
        CHECK(r.entries().size() == 2);
    }
    {
        ModelRegistry r(reg.string());  // reload from disk
        CHECK(r.entries().size() == 2);
        CHECK(r.find("model") != nullptr && r.find((dir / "model.gguf").string()) != nullptr);
        CHECK(r.find("nope") == nullptr);
        r.add((dir / "model.gguf").string(), "", false);  // same file again: refreshed, not duplicated
        CHECK(r.entries().size() == 2);
        CHECK(r.verify("copy"));
        { std::ofstream f(copy, std::ios::app | std::ios::binary); f << 'x'; }  // modify the copy
        std::string actual;
        CHECK(!r.verify("copy", &actual));
        CHECK(actual.size() == 64);
        const ModelEntry& e2 = r.add(copy.string(), "", false);  // size changed -> re-hashed
        CHECK(e2.id == "copy" && e2.sha256 == actual);
        CHECK(r.remove("copy") && !r.remove("copy"));
        CHECK(r.entries().size() == 1);
    }
    CHECK(ModelRegistry(reg.string()).entries().size() == 1);
    fs::remove_all(reg.parent_path());
    fs::remove(copy);
    return CHECK_RESULT();
}
