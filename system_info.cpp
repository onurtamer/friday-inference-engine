#include "system_info.h"

#include <cstring>
#include <fstream>
#include <thread>

#include "cpu_kquants.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace friday {

namespace {
#if defined(_M_X64) || defined(__x86_64__)
void cpuidex(int leaf, int sub, int out[4]) {
#if defined(_MSC_VER)
    __cpuidex(out, leaf, sub);
#else
    unsigned a, b, c, d;
    __cpuid_count(leaf, sub, a, b, c, d);
    out[0] = (int)a; out[1] = (int)b; out[2] = (int)c; out[3] = (int)d;
#endif
}
uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
#endif
}
#endif
}  // namespace

CpuInfo cpuInfo() {
    CpuInfo c;
    c.logical_cpus = (int)std::thread::hardware_concurrency();
    c.cpu_kernels = friday_cpu::hasAvx2() ? "AVX2" : "scalar";
#if defined(_M_X64) || defined(__x86_64__)
    c.arch = "x86_64";
    int r[4];
    cpuidex(0x80000000, 0, r);
    if ((unsigned)r[0] >= 0x80000004u) {
        char brand[49] = {};
        for (int i = 0; i < 3; ++i) { cpuidex(0x80000002 + i, 0, r); std::memcpy(brand + 16 * i, r, 16); }
        c.brand = brand;
        while (!c.brand.empty() && c.brand.back() == ' ') c.brand.pop_back();
        size_t s = c.brand.find_first_not_of(' ');
        if (s != std::string::npos) c.brand = c.brand.substr(s);
    }
    cpuidex(1, 0, r);
    const bool osxsave = (r[2] >> 27) & 1;
    const bool ymm = osxsave && ((xgetbv0() & 0x6) == 0x6);
    const bool zmm = osxsave && ((xgetbv0() & 0xE6) == 0xE6);
    if ((r[2] >> 19) & 1) c.features.push_back("SSE4.1");
    if (ymm && ((r[2] >> 28) & 1)) c.features.push_back("AVX");
    if (ymm && ((r[2] >> 12) & 1)) c.features.push_back("FMA");
    if (ymm && ((r[2] >> 29) & 1)) c.features.push_back("F16C");
    cpuidex(7, 0, r);
    if (ymm && ((r[1] >> 5) & 1)) c.features.push_back("AVX2");
    if (zmm && ((r[1] >> 16) & 1)) c.features.push_back("AVX512F");
    if (zmm && ((r[2] >> 11) & 1)) c.features.push_back("AVX512_VNNI");
    cpuidex(7, 1, r);
    if (ymm && ((r[0] >> 4) & 1)) c.features.push_back("AVX_VNNI");
#elif defined(__aarch64__)
    c.arch = "aarch64";
    c.features.push_back("NEON");
#if defined(__linux__)
    const unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
    if (hw & HWCAP_ASIMDDP) c.features.push_back("DOTPROD");
    if (hw & HWCAP_SVE) c.features.push_back("SVE");
#ifdef HWCAP2_I8MM
    if (hw2 & HWCAP2_I8MM) c.features.push_back("I8MM");
#endif
#ifdef HWCAP2_SVE2
    if (hw2 & HWCAP2_SVE2) c.features.push_back("SVE2");
#endif
    (void)hw2;
#endif
#else
    c.arch = "unknown";
#endif
#if defined(__linux__)
    if (c.brand.empty()) {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line))
            if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0) {
                c.brand = line.substr(line.find(':') + 2);
                break;
            }
    }
#endif
    return c;
}

MemoryInfo memoryInfo() {
    MemoryInfo m;
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) { m.total_bytes = st.ullTotalPhys; m.available_bytes = st.ullAvailPhys; }
#else
    std::ifstream f("/proc/meminfo");
    std::string k;
    uint64_t v;
    std::string unit;
    while (f >> k >> v >> unit) {
        if (k == "MemTotal:") m.total_bytes = v * 1024;
        else if (k == "MemAvailable:") m.available_bytes = v * 1024;
    }
#endif
    return m;
}

CpuTimes cpuTimes() {
    CpuTimes t;
#if defined(_WIN32)
    FILETIME idle, kernel, user;
    if (GetSystemTimes(&idle, &kernel, &user)) {
        auto u64 = [](const FILETIME& f) { return ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime; };
        t.idle = u64(idle);
        t.total = u64(kernel) + u64(user);  // kernel time includes idle time
    }
#else
    std::ifstream f("/proc/stat");
    std::string cpu;
    uint64_t v[10] = {};
    f >> cpu;
    for (int i = 0; i < 10 && (f >> v[i]); ++i) {}
    for (uint64_t x : v) t.total += x;
    t.idle = v[3] + v[4];  // idle + iowait
#endif
    return t;
}

double cpuBusyPercent(const CpuTimes& a, const CpuTimes& b) {
    const double total = (double)(b.total - a.total), idle = (double)(b.idle - a.idle);
    return total > 0 ? 100.0 * (1.0 - idle / total) : 0.0;
}

}  // namespace friday
