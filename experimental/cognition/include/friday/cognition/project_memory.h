#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace friday {
namespace cognition {

enum class SymbolType {
    CLASS,
    STRUCT,
    FUNCTION,
    KERNEL,
    ENUM,
    VARIABLE
};

struct CodeSymbol {
    std::string name;
    SymbolType type;
    std::string file_path;
    int line_number;
    std::string signature;
    std::vector<std::string> dependencies;
};

class ProjectMemoryManager {
public:
    ProjectMemoryManager() = default;
    ~ProjectMemoryManager() = default;

    // Symbol Graph Indexing
    void register_symbol(const CodeSymbol& sym);

    // Precise Symbol Lookup (Tier 3 / Tier 4 FTS Style)
    const CodeSymbol* lookup_symbol(const std::string& name) const;
    std::vector<CodeSymbol> search_by_prefix(const std::string& prefix) const;

    // Graph Dependency Traversal
    std::vector<std::string> get_dependencies(const std::string& symbol_name) const;

    size_t get_symbol_count() const { return symbols_.size(); }

private:
    std::unordered_map<std::string, CodeSymbol> symbols_;
};

} // namespace cognition
} // namespace friday
