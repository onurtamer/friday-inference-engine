#ifndef FRIDAY_MODEL_REGISTRY_H
#define FRIDAY_MODEL_REGISTRY_H

// Local model registry: one JSON file listing the GGUF models the user added,
// with what is needed to identify and describe them (hash, license,
// architecture, quantization, size). Default location:
//   Windows: %APPDATA%\friday\models.json
//   Linux:   $XDG_CONFIG_HOME/friday/models.json or ~/.config/friday/models.json
// FRIDAY_REGISTRY=<file> overrides it (used by tests).

#include <cstdint>
#include <string>
#include <vector>

namespace friday {

struct ModelEntry {
    std::string id;            // short name, unique in the registry
    std::string path;          // absolute path of the .gguf file
    uint64_t size = 0;         // bytes
    int64_t mtime = 0;         // last write time (filesystem clock ticks), for hash caching
    std::string sha256;
    std::string architecture;  // general.architecture
    std::string name;          // general.name
    std::string license;       // general.license (empty if the file does not say)
    std::string quant;         // general.file_type name, e.g. "Q2_K"
    uint64_t parameters = 0;   // sum of tensor element counts
    int layers = 0;
    int context_length = 0;
    std::string added;         // UTC, ISO 8601
};

class ModelRegistry {
public:
    explicit ModelRegistry(const std::string& file = defaultPath());
    static std::string defaultPath();

    const std::vector<ModelEntry>& entries() const { return entries_; }
    const std::string& file() const { return file_; }

    // Adds or refreshes a model. The SHA-256 is reused when path, size and mtime
    // are unchanged. `id` empty: derived from the file name. Saves the file.
    const ModelEntry& add(const std::string& gguf_path, const std::string& id = "", bool verbose = true);
    bool remove(const std::string& id);  // saves; false if not found
    // Re-hashes the file; returns true if it still matches the stored hash.
    bool verify(const std::string& id, std::string* actual = nullptr);
    // Entry by id, or by path; nullptr if absent.
    const ModelEntry* find(const std::string& id_or_path) const;

private:
    void load();
    void save() const;
    std::string file_;
    std::vector<ModelEntry> entries_;
};

}  // namespace friday

#endif  // FRIDAY_MODEL_REGISTRY_H
