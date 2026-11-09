#include "gpu_monitor.h"

#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace friday {

namespace {
// NVML ABI subset (nvml.h); return value 0 = NVML_SUCCESS.
struct NvmlUtil { unsigned gpu, memory; };
enum { kClockGraphics = 0, kClockSm = 1, kClockMem = 2, kTempGpu = 0 };
}  // namespace

struct GpuMonitor::Fns {
    int (*init)() = nullptr;
    int (*shutdown)() = nullptr;
    int (*count)(unsigned*) = nullptr;
    int (*by_index)(unsigned, void**) = nullptr;
    int (*by_bus)(const char*, void**) = nullptr;
    int (*name)(void*, char*, unsigned) = nullptr;
    int (*driver)(char*, unsigned) = nullptr;
    int (*clock)(void*, int, unsigned*) = nullptr;
    int (*max_clock)(void*, int, unsigned*) = nullptr;
    int (*temp)(void*, int, unsigned*) = nullptr;
    int (*power)(void*, unsigned*) = nullptr;
    int (*util)(void*, NvmlUtil*) = nullptr;
    int (*gen_cur)(void*, unsigned*) = nullptr;
    int (*width_cur)(void*, unsigned*) = nullptr;
    int (*gen_max)(void*, unsigned*) = nullptr;
    int (*width_max)(void*, unsigned*) = nullptr;
    int (*mem)(void*, void*) = nullptr;  // nvmlMemory_t {total, free, used}
};

GpuMonitor::GpuMonitor() : f_(new Fns) {
#if defined(_WIN32)
    lib_ = (void*)LoadLibraryA("nvml.dll");
    auto sym = [&](const char* n) -> void* { return lib_ ? (void*)GetProcAddress((HMODULE)lib_, n) : nullptr; };
#else
    lib_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    auto sym = [&](const char* n) -> void* { return lib_ ? dlsym(lib_, n) : nullptr; };
#endif
    if (!lib_) return;
    f_->init = (int (*)())sym("nvmlInit_v2");
    f_->shutdown = (int (*)())sym("nvmlShutdown");
    f_->count = (int (*)(unsigned*))sym("nvmlDeviceGetCount_v2");
    f_->by_index = (int (*)(unsigned, void**))sym("nvmlDeviceGetHandleByIndex_v2");
    f_->by_bus = (int (*)(const char*, void**))sym("nvmlDeviceGetHandleByPciBusId_v2");
    f_->name = (int (*)(void*, char*, unsigned))sym("nvmlDeviceGetName");
    f_->driver = (int (*)(char*, unsigned))sym("nvmlSystemGetDriverVersion");
    f_->clock = (int (*)(void*, int, unsigned*))sym("nvmlDeviceGetClockInfo");
    f_->max_clock = (int (*)(void*, int, unsigned*))sym("nvmlDeviceGetMaxClockInfo");
    f_->temp = (int (*)(void*, int, unsigned*))sym("nvmlDeviceGetTemperature");
    f_->power = (int (*)(void*, unsigned*))sym("nvmlDeviceGetPowerUsage");
    f_->util = (int (*)(void*, NvmlUtil*))sym("nvmlDeviceGetUtilizationRates");
    f_->gen_cur = (int (*)(void*, unsigned*))sym("nvmlDeviceGetCurrPcieLinkGeneration");
    f_->width_cur = (int (*)(void*, unsigned*))sym("nvmlDeviceGetCurrPcieLinkWidth");
    f_->gen_max = (int (*)(void*, unsigned*))sym("nvmlDeviceGetMaxPcieLinkGeneration");
    f_->width_max = (int (*)(void*, unsigned*))sym("nvmlDeviceGetMaxPcieLinkWidth");
    f_->mem = (int (*)(void*, void*))sym("nvmlDeviceGetMemoryInfo");
    if (!f_->init || !f_->shutdown || !f_->count || !f_->by_index || f_->init() != 0) return;
    unsigned n = 0;
    if (f_->count(&n) != 0) { f_->shutdown(); return; }
    count_ = (int)n;
    ok_ = true;
}

GpuMonitor::~GpuMonitor() {
    if (ok_ && f_->shutdown) f_->shutdown();
#if defined(_WIN32)
    if (lib_) FreeLibrary((HMODULE)lib_);
#else
    if (lib_) dlclose(lib_);
#endif
    delete f_;
}

void* GpuMonitor::handle(int index) const {
    void* h = nullptr;
    if (!ok_ || index < 0 || index >= count_ || f_->by_index((unsigned)index, &h) != 0) return nullptr;
    return h;
}

bool GpuMonitor::staticInfo(int index, GpuStatic* out) const {
    void* h = handle(index);
    if (!h) return false;
    unsigned v = 0;
    char buf[96] = {};
    if (f_->name && f_->name(h, buf, sizeof(buf)) == 0) out->name = buf;
    if (f_->driver && f_->driver(buf, sizeof(buf)) == 0) out->driver = buf;
    if (f_->max_clock && f_->max_clock(h, kClockSm, &v) == 0) out->sm_clock_max_mhz = (int)v;
    if (f_->max_clock && f_->max_clock(h, kClockMem, &v) == 0) out->mem_clock_max_mhz = (int)v;
    if (f_->gen_max && f_->gen_max(h, &v) == 0) out->pcie_gen_max = (int)v;
    if (f_->width_max && f_->width_max(h, &v) == 0) out->pcie_width_max = (int)v;
    unsigned long long mem[3] = {};
    if (f_->mem && f_->mem(h, mem) == 0) out->memory_total = mem[0];
    return true;
}

bool GpuMonitor::staticInfoByBusId(const std::string& bus_id, GpuStatic* out) const {
    if (!ok_ || !f_->by_bus) return false;
    void* h = nullptr;
    if (f_->by_bus(bus_id.c_str(), &h) != 0) return false;
    for (int i = 0; i < count_; ++i)
        if (handle(i) == h) return staticInfo(i, out);
    return false;
}

bool GpuMonitor::sample(int index, GpuSample* out) const {
    void* h = handle(index);
    if (!h) return false;
    unsigned v = 0;
    if (f_->clock && f_->clock(h, kClockSm, &v) == 0) out->sm_clock_mhz = (int)v;
    if (f_->clock && f_->clock(h, kClockMem, &v) == 0) out->mem_clock_mhz = (int)v;
    if (f_->temp && f_->temp(h, kTempGpu, &v) == 0) out->temperature_c = (int)v;
    if (f_->power && f_->power(h, &v) == 0) out->power_w = v / 1000.0;
    NvmlUtil u{};
    if (f_->util && f_->util(h, &u) == 0) { out->util_gpu_pct = (int)u.gpu; out->util_mem_pct = (int)u.memory; }
    if (f_->gen_cur && f_->gen_cur(h, &v) == 0) out->pcie_gen_cur = (int)v;
    if (f_->width_cur && f_->width_cur(h, &v) == 0) out->pcie_width_cur = (int)v;
    return true;
}

}  // namespace friday
