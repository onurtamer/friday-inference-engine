#include "friday/cognition/project_memory.h"
#include <algorithm>

namespace friday {
namespace cognition {

void ProjectMemoryManager::register_symbol(const CodeSymbol& sym) {
    symbols_[sym.name] = sym;
}

const CodeSymbol* ProjectMemoryManager::lookup_symbol(const std::string& name) const {
    auto it = symbols_.find(name);
    return (it != symbols_.end()) ? &it->second : nullptr;
}

std::vector<CodeSymbol> ProjectMemoryManager::search_by_prefix(const std::string& prefix) const {
    std::vector<CodeSymbol> results;
    for (const auto& pair : symbols_) {
        if (pair.first.rfind(prefix, 0) == 0) {
            results.push_back(pair.second);
        }
    }
    return results;
}

std::vector<std::string> ProjectMemoryManager::get_dependencies(const std::string& symbol_name) const {
    auto sym = lookup_symbol(symbol_name);
    return sym ? sym->dependencies : std::vector<std::string>();
}

} // namespace cognition
} // namespace friday
