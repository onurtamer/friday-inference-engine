#include "model_registry.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "gguf_file.h"
#include "model_config.h"
#include "sha256.h"

namespace fs = std::filesystem;

namespace friday {

namespace {

std::string envOr(const char* name, const std::string& def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : def;
}

std::string utcNow() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string idFromPath(const std::string& path) {
    std::string stem = fs::path(path).stem().string();
    std::string id;
    for (char c : stem) id += std::isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_' ? (char)std::tolower((unsigned char)c) : '-';
    return id.empty() ? "model" : id;
}

int64_t mtimeOf(const fs::path& p) {
    return (int64_t)fs::last_write_time(p).time_since_epoch().count();
}

void printProgress(uint64_t done, uint64_t total) {
    std::fprintf(stderr, "\r  hashing %.1f / %.1f GB", done / 1e9, total / 1e9);
}

nlohmann::json toJson(const ModelEntry& e) {
    return {{"id", e.id}, {"path", e.path}, {"size", e.size}, {"mtime", e.mtime}, {"sha256", e.sha256},
            {"architecture", e.architecture}, {"name", e.name}, {"license", e.license}, {"quant", e.quant},
            {"parameters", e.parameters}, {"layers", e.layers}, {"context_length", e.context_length},
            {"added", e.added}};
}

ModelEntry fromJson(const nlohmann::json& j) {
    ModelEntry e;
    e.id = j.at("id").get<std::string>();
    e.path = j.at("path").get<std::string>();
    e.size = j.value("size", (uint64_t)0);
    e.mtime = j.value("mtime", (int64_t)0);
    e.sha256 = j.value("sha256", std::string());
    e.architecture = j.value("architecture", std::string());
    e.name = j.value("name", std::string());
    e.license = j.value("license", std::string());
    e.quant = j.value("quant", std::string());
    e.parameters = j.value("parameters", (uint64_t)0);
    e.layers = j.value("layers", 0);
    e.context_length = j.value("context_length", 0);
    e.added = j.value("added", std::string());
    return e;
}

}  // namespace

std::string ModelRegistry::defaultPath() {
    const std::string over = envOr("FRIDAY_REGISTRY", "");
    if (!over.empty()) return over;
#ifdef _WIN32
    const std::string base = envOr("APPDATA", envOr("USERPROFILE", "."));
#else
    const std::string base = envOr("XDG_CONFIG_HOME", envOr("HOME", ".") + "/.config");
#endif
    return (fs::path(base) / "friday" / "models.json").string();
}

ModelRegistry::ModelRegistry(const std::string& file) : file_(file) { load(); }

void ModelRegistry::load() {
    entries_.clear();
    std::ifstream f(file_);
    if (!f) return;  // no registry yet
    nlohmann::json j;
    try {
        f >> j;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("model registry " + file_ + " is not valid JSON: " + e.what());
    }
    for (const auto& m : j.value("models", nlohmann::json::array())) entries_.push_back(fromJson(m));
}

void ModelRegistry::save() const {
    const fs::path p(file_);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    nlohmann::json j;
    j["version"] = 1;
    j["models"] = nlohmann::json::array();
    for (const auto& e : entries_) j["models"].push_back(toJson(e));
    const fs::path tmp = p.string() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write " + tmp.string());
        f << j.dump(2) << "\n";
        if (!f) throw std::runtime_error("write failed: " + tmp.string());
    }
    fs::rename(tmp, p);  // replaces the old file atomically on the same volume
}

const ModelEntry& ModelRegistry::add(const std::string& gguf_path, const std::string& id_in, bool verbose) {
    const fs::path abs = fs::absolute(gguf_path).lexically_normal();
    if (!fs::is_regular_file(abs)) throw std::runtime_error("not a file: " + abs.string());
    const GgufFile g(abs.string());
    const ModelConfig cfg = ModelConfig::fromGguf(g);

    ModelEntry e;
    e.path = abs.string();
    e.size = (uint64_t)fs::file_size(abs);
    e.mtime = mtimeOf(abs);
    e.architecture = cfg.architecture;
    e.name = cfg.name;
    e.license = g.getString("general.license", "");
    e.quant = cfg.file_type;
    e.layers = cfg.num_layers;
    e.context_length = cfg.context_length;
    for (const auto& t : g.tensors()) {
        uint64_t n = 1;
        for (int64_t d : t.ne) n *= (uint64_t)d;
        e.parameters += n;
    }

    // Existing entry for this path: keep its id and reuse the hash if unchanged.
    ModelEntry* existing = nullptr;
    for (auto& x : entries_) if (fs::path(x.path) == abs) existing = &x;
    if (existing && existing->size == e.size && existing->mtime == e.mtime && !existing->sha256.empty()) {
        e.sha256 = existing->sha256;
    } else {
        if (verbose) std::fprintf(stderr, "Computing SHA-256 of %s ...\n", abs.string().c_str());
        e.sha256 = sha256File(abs.string(), verbose ? &printProgress : nullptr);
        if (verbose) std::fprintf(stderr, "\n");
    }

    if (existing) {
        e.id = existing->id;
        e.added = existing->added;
        *existing = e;
        save();
        return *existing;
    }
    std::string id = id_in.empty() ? idFromPath(abs.string()) : id_in;
    if (!id_in.empty() && find(id)) throw std::runtime_error("model id already in use: " + id);
    for (int k = 2; find(id); ++k) id = idFromPath(abs.string()) + "-" + std::to_string(k);
    e.id = id;
    e.added = utcNow();
    entries_.push_back(e);
    save();
    return entries_.back();
}

bool ModelRegistry::remove(const std::string& id) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->id == id) { entries_.erase(it); save(); return true; }
    }
    return false;
}

bool ModelRegistry::verify(const std::string& id, std::string* actual) {
    for (auto& e : entries_) {
        if (e.id != id) continue;
        const std::string h = sha256File(e.path);
        if (actual) *actual = h;
        return h == e.sha256;
    }
    throw std::runtime_error("unknown model id: " + id);
}

const ModelEntry* ModelRegistry::find(const std::string& id_or_path) const {
    for (const auto& e : entries_) if (e.id == id_or_path) return &e;
    std::error_code ec;
    const fs::path abs = fs::absolute(id_or_path, ec).lexically_normal();
    if (!ec) for (const auto& e : entries_) if (fs::path(e.path) == abs) return &e;
    return nullptr;
}

}  // namespace friday
