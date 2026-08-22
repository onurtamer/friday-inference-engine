<div align="center">
  
# 🚀 FRIDAY Inference Engine
**Zero-Copy Asynchronous I/O LLM Architecture**

![C++17](https://img.shields.io/badge/C++-17-blue.svg?style=for-the-badge&logo=c%2B%2B)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg?style=for-the-badge&logo=nvidia)
![Windows](https://img.shields.io/badge/Windows-Native-0078D6.svg?style=for-the-badge&logo=windows)
![Python](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=for-the-badge&logo=python)

*An experimental, bare-metal inference engine designed to tackle I/O bottlenecks in data center infrastructure.*

</div>

---

## 📖 Overview
The **FRIDAY Engine** is an experimental, bare-metal C++17 and CUDA inference engine built from scratch. It is designed to tackle one of the most critical bottlenecks in data center infrastructure and cloud AI deployments: **I/O overhead and memory latency.**

Instead of relying on heavy frameworks (like PyTorch or TensorFlow), this engine bypasses the operating system's standard page cache entirely. It streams highly compressed **(1.58-bit Ternary) Mixture-of-Experts (MoE)** weights directly from NVMe SSD storage to GPU VRAM using Asynchronous I/O (`O_DIRECT` equivalents) and **Zero-Copy DMA** techniques.

---

## ⚡ Architectural Highlights

### 1. Operating System Bypass (Asynchronous `O_DIRECT`)
Standard file I/O operations inherently buffer data through the OS kernel, causing severe memory duplication and latency. 
FRIDAY employs low-level Windows APIs with unbuffered, sector-aligned sector reads (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`). This allows the architecture to bypass the OS page cache entirely, streaming expert weights directly into GPU-accessible memory.

### 2. Zero-Copy DMA & Memory Safety
To ensure absolute hardware isolation and memory safety, the engine utilizes a custom `ZeroCopyMemoryManager`. 
- **Isolated Pointers:** Device pointers (GPU) and Host pointers (CPU) are strictly isolated.
- **Auto-Regressive KV-Cache:** Maintained securely by leveraging asynchronous `cudaMemcpy` operations (Device-to-Host) to persist context across generation loops without risking Segmentation Faults or OS-level crashes.

### 3. 1.58-Bit Ternary Compute (Extreme Quantization)
To maximize throughput and minimize the storage footprint, the data pipeline forcibly quantizes standard 16-bit FP16 models down to a **1.58-bit Ternary format (-1, 0, 1)**. 
- **Storage Footprint:** Reduced a ~2.2 GB model down to ~190 MB.
- **Custom CUDA Kernels:** Matrix multiplications bypass heavy cuBLAS dependencies, relying instead on custom-built global CUDA kernels optimized for Ternary states.

---

## 🛠️ Technical Stack
- **Core Engine:** C++17 (MSVC)
- **Parallel Computing:** NVIDIA CUDA Toolkit
- **Data Ingestion/Pipeline:** Python 3.14 (Pandas, HuggingFace Hub, Safetensors)
- **Target OS:** Native Windows (Executable `.exe`)

---

## 🎯 Project Vision (Beta 1.0)
This Proof of Concept (Beta 1.0) was developed to demonstrate low-level systems programming capabilities, memory architecture optimization, and the theoretical limits of hardware when stripped of high-level software abstractions. 

It stands as a foundational study for future architectures in high-performance cloud networks and disaster recovery environments.

<div align="center">
  <i>Built with absolute precision. No high-level bloat. Just pure metal.</i>
</div>
