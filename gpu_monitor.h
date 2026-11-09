#ifndef FRIDAY_GPU_MONITOR_H
#define FRIDAY_GPU_MONITOR_H

// NVIDIA GPU telemetry through NVML, loaded at run time (nvml.dll /
// libnvidia-ml.so.1 ship with the driver), so nothing links against it and
// CPU-only builds work unchanged. Every query returns false when NVML or the
// device is unavailable. Pure C++.

#include <cstdint>
#include <string>

namespace friday {

struct GpuSample {
    int sm_clock_mhz = 0, mem_clock_mhz = 0;
    int temperature_c = 0;
    double power_w = 0.0;
    int util_gpu_pct = 0, util_mem_pct = 0;
    int pcie_gen_cur = 0, pcie_width_cur = 0;
};

struct GpuStatic {
    std::string name, driver;
    int sm_clock_max_mhz = 0, mem_clock_max_mhz = 0;
    int pcie_gen_max = 0, pcie_width_max = 0;
    uint64_t memory_total = 0;
};

class GpuMonitor {
public:
    GpuMonitor();
    ~GpuMonitor();
    GpuMonitor(const GpuMonitor&) = delete;
    GpuMonitor& operator=(const GpuMonitor&) = delete;

    bool available() const { return ok_; }
    int deviceCount() const { return count_; }
    // Device selection: NVML index, or PCI bus id ("0000:01:00.0", as from cudaDeviceGetPCIBusId).
    bool staticInfo(int index, GpuStatic* out) const;
    bool staticInfoByBusId(const std::string& bus_id, GpuStatic* out) const;
    bool sample(int index, GpuSample* out) const;

private:
    void* handle(int index) const;
    struct Fns;
    Fns* f_ = nullptr;
    void* lib_ = nullptr;
    bool ok_ = false;
    int count_ = 0;
};

}  // namespace friday

#endif  // FRIDAY_GPU_MONITOR_H
