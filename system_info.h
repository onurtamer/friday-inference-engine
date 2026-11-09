#ifndef FRIDAY_SYSTEM_INFO_H
#define FRIDAY_SYSTEM_INFO_H

// Host capability report (CPU features, cores, RAM). GPU details come from
// Engine::gpuInfo() in CUDA builds. Pure C++.

#include <cstdint>
#include <string>
#include <vector>

namespace friday {

struct CpuInfo {
    std::string arch;            // "x86_64", "aarch64", ...
    std::string brand;           // CPU model string, if known
    int logical_cpus = 0;
    std::vector<std::string> features;   // e.g. AVX2 FMA F16C AVX512F / NEON DOTPROD I8MM SVE
    std::string cpu_kernels;     // kernel path the CPU backend uses ("AVX2" / "scalar")
};

struct MemoryInfo {
    uint64_t total_bytes = 0;
    uint64_t available_bytes = 0;
};

CpuInfo cpuInfo();
MemoryInfo memoryInfo();

// System-wide CPU time counters; busy % between two snapshots measures the
// load of everything running on the machine (including this process).
struct CpuTimes { uint64_t idle = 0, total = 0; };
CpuTimes cpuTimes();
double cpuBusyPercent(const CpuTimes& a, const CpuTimes& b);

}  // namespace friday

#endif  // FRIDAY_SYSTEM_INFO_H
