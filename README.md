<div align="center">
  
# 🚀 FRIDAY Engine 2.0 (V32)
### **Bare-Metal C++17 & Native CUDA LLM Inference Engine**
*Elastic Tiered Memory • Zero External Runtime Dependencies • Real-Time Token Streaming*

[![C++17](https://img.shields.io/badge/C++-17-00599C.svg?style=for-the-badge&logo=c%2B%2B)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Windows%20Native-0078D6.svg?style=for-the-badge&logo=windows)](https://microsoft.com)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--14B-8A2BE2.svg?style=for-the-badge)](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg?style=for-the-badge)](docs/ROADMAP.md)

<p align="center">
  <b>Run 14-billion-parameter LLMs locally on a consumer GPU within a 6 GB VRAM budget.</b><br>
  No PyTorch. No ONNX Runtime. No llama.cpp. No vLLM. Plain C++17 and CUDA (plus the header-only nlohmann/json).
</p>

</div>

---

## 📖 Overview

**FRIDAY Engine 2.0** is an experimental inference engine written from scratch in **C++17** and **NVIDIA CUDA**.

> **Project status:** experimental. The v1 plan (CUDA + ARM64 CPU backends, GGUF loading, chat, persistent memory, OpenAI-compatible API) is in [docs/ROADMAP.md](docs/ROADMAP.md); decisions are recorded in [docs/decisions/](docs/decisions/).

Modern cloud and edge LLM deployments are heavily constrained by high memory demands and runtime overhead. Running a **14-Billion parameter** model (like `Qwen2.5-14B-Instruct`) typically requires 10 GB – 16 GB of dedicated GPU VRAM. 

FRIDAY Engine breaks this barrier using **Elastic Tiered Memory (ETM)** combined with **native custom CUDA GEMV kernels** and **asynchronous PCIe ping-pong streaming**. It automatically partitions transformer layers between GPU VRAM and Host RAM, completely eliminates disk reading bottlenecks during autoregressive generation, and streams responses token-by-token in real time.

---

## ⚡ Key Innovations & Architectural Breakthroughs

```mermaid
flowchart TD
    subgraph Storage["1. Storage & Preprocessing"]
        GGUF["Qwen2.5-14B-Instruct-Q4_0.gguf"] -->|converter.py| BIN["friday_aligned_int4.bin (4KB Aligned)"]
        GGUF -->|converter.py| MAP["tensor_map.json + model_params.json"]
    end

    subgraph Initialization["2. Elastic Tiered Memory Initialization"]
        BIN --> ETM["Elastic Tiered Memory Manager"]
        ETM -->|Static Tensors| EMB["token_embd.weight & output.weight (GPU)"]
        ETM -->|VRAM Limit Filter| GPU_LAYERS["GPU-Resident Layers (0 .. N)"]
        ETM -->|Zero Disk I/O| HOST_CACHE["Host RAM Pinned Cache (N+1 .. 47)"]
    end

    subgraph Inference["3. Asynchronous Ping-Pong Inference Loop"]
        direction TB
        GPU_LAYERS -->|Zero-Copy VRAM Execution| COMPUTE["CUDA Stream 1: GEMV Compute"]
        HOST_CACHE -->|Overlapped PCIe DMA Transfer| DMA["CUDA Stream 2: Asynchronous Prefetch"]
        DMA -.->|Ping-Pong Ping/Pong Buffers| COMPUTE
    end

    subgraph Kernels["4. Native CUDA Kernels & Output"]
        COMPUTE --> RMS["RMSNorm Kernel (Warp Shuffle)"]
        RMS --> ATTN["Custom RoPE (Base 1M) on GPU; GQA attention on CPU"]
        ATTN --> SOFTMAX["FP32-Safe Softmax (Overflow-Proof)"]
        SOFTMAX --> MLP["SwiGLU Activation + Down Projection"]
        MLP --> STREAM["BPE UTF-8 Tokenizer -> Console Stream"]
    end
```

### 1. 🧠 Elastic Tiered Memory (ETM) Architecture
- **Dynamic VRAM Sizing (`--vram <gb>`):** The engine probes total and available physical VRAM dynamically (or obeys a user-specified limit such as `--vram 6.0`).
- **GPU-Resident Layer Priority:** Static critical structures (embedding table, RMSNorm weights, output LM head, and activation ping-pong buffers) are locked into VRAM. As many transformer layers as possible are placed permanently in high-speed GPU memory.
- **Pinned Host RAM Pre-Caching:** Non-resident layers are pre-loaded into Host RAM during startup. **No disk I/O occurs during token generation**, completely bypassing the storage latency wall.
- **Asynchronous Ping-Pong Double Buffering:** Layers streamed from Host RAM utilize two scratch device buffers (`Buffer A` and `Buffer B`) and two independent CUDA streams (`stream_compute` and `stream_io`). While the GPU computes Layer $L$ on Buffer A, CUDA asynchronous DMA (`cudaMemcpyAsync`) transfers Layer $L+1$ into Buffer B simultaneously.

### 2. ⚡ Native Custom CUDA GEMV Kernels
FRIDAY does not link against external BLAS or inference libraries. Every operation is hand-crafted in CUDA:
- **`Q4_0` GEMV Kernel:** Unpacks 4-bit nibbles on-the-fly from 32-element blocks with FP16 scales using warp-level reduction primitives (`__shfl_down_sync`).
- **`Q4_1` GEMV Kernel:** Handles 4-bit quantization with dual parameters (scale + min/bias FP16).
- **`Q6_K` GEMV Kernel:** Decodes 256-element super-blocks composed of 16 sub-scales, lower 4-bit nibbles, and upper 2-bit bitfields in register space.
- **`Q8_0` GEMV Kernel:** High-precision 8-bit quantized matrix-vector multiplication.
- **Fast RMSNorm:** Vectorized FP32/FP16 normalization with warp-shuffle reduction.
- **Custom RoPE (Rotary Position Embeddings):** Direct device rotation supporting rotary base frequencies up to $1,000,000$ (as required by modern Qwen2.5 models).
- **SwiGLU Non-Linearity:** Parallel fused kernel computing `silu(gate) * up` in place.

### 3. 🛡️ Numerically Stable FP32-Safe Softmax
> In the current version, self-attention and its softmax run on the **CPU** (Q/K/V are copied to host memory and the KV cache lives in host RAM). Moving attention and the KV cache to the GPU is planned (roadmap R4).

Under Grouped-Query Attention (GQA) with scaling factors ($1 / \sqrt{d_k}$), standard FP16 softmax accumulators often experience exponent overflow or catastrophic cancellation, producing `NaN` or infinite values.
FRIDAY Engine implements a **two-pass online Safe Softmax**:
1. Finds the maximum attention score per head across sequence positions.
2. Subtracts the maximum before exponentiation and computes intermediate exponentials and sum in **FP32 registers**, safely casting back to `half` for final score-value projection.

### 4. 🔤 Native Fast C++ BPE Tokenizer
- Clean, standalone C++ implementation of Byte Pair Encoding (BPE).
- Full UTF-8 byte-to-unicode remapping (decoding GPT-2 / Qwen artifacts like `Ġ` $\to$ space, `Ċ` $\to$ newline, `ĉ` $\to$ tab, `\xe2\x96\x81`).
- Native ChatML support (`<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant`).
- **Real-Time Token Streaming:** Tokens appear immediately on the console (`std::flush`) with negligible latency as they are decoded.

### 5. 📦 4KB Sector-Aligned Weight Serialization (`converter.py`)
Standard GGUF files place tensor payloads at variable byte alignments. FRIDAY includes a standalone Python utility (`converter.py`) that reads standard GGUF models and serializes them into **4096-byte sector-aligned binary files (`friday_aligned_int4.bin`)**. The alignment is a prerequisite for unbuffered I/O (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`); the current loader still uses buffered `fread`, and unbuffered I/O is planned (roadmap R4).

---

## 🏗️ Repository Structure

```
friday-inference-engine/
├── CMakeLists.txt              # Unified CMake build configuration (MSVC + NVCC)
├── main.cpp                    # Application entry point, CLI parser & streaming loop
├── transformer.h               # Transformer engine & Elastic Tiered Memory definitions
├── transformer.cpp             # Forward pass, GQA, SwiGLU, and layer dispatching
├── io_manager.h                # Asynchronous I/O manager interface
├── io_manager.cpp              # Sector-aligned asynchronous file operations
├── memory_manager.h            # Host and Device memory allocator
├── memory_manager.cpp          # Pinned host memory & device buffer management
├── kernel_dispatcher.h         # Quantization type dispatcher
├── kernel_dispatcher.cpp       # Runtime GEMV kernel selector
├── cuda_math_kernels.h         # CUDA kernel headers (Q4_0, Q4_1, Q6_K, Q8_0)
├── cuda_math_kernels.cu        # High-performance CUDA GEMV kernels & reductions
├── int4_kernel.cu              # Dedicated INT4 GEMV & RMSNorm kernels
├── tokenizer.h                 # BPE Tokenizer interface
├── tokenizer.cpp               # BPE encoding, ChatML parser, and UTF-8 mapper
├── friday_types.h              # Platform I/O types and alignment constants
├── converter.py                # Standalone GGUF -> 4KB Aligned BIN converter
├── model_params.json           # Model configuration parameters (dim, heads, layers)
├── tensor_map.json             # Offset and dimension lookup table for weights
├── include/
│   └── nlohmann/
│       └── json.hpp            # Header-only JSON library for parameter mapping
├── docs/
│   ├── ROADMAP.md              # Milestones and version scope
│   └── decisions/              # Architecture decision records (ADRs)
├── .gitignore                  # Strict ignore rules for large weights and build artifacts
├── LICENSE                     # Apache License 2.0
└── README.md                   # Technical documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Operating System:** Windows 10 / 11 (Linux builds are not yet tested)
- **Compiler:** Microsoft Visual Studio 2022 (MSVC C++17)
- **CUDA Toolkit:** NVIDIA CUDA Toolkit 12.0 or higher
- **Build System:** CMake 3.18 or higher
- **GPU:** NVIDIA GPU. The build targets `sm_86` (RTX 30-series) by default; other architectures need `CMAKE_CUDA_ARCHITECTURES` changed and are untested.

### Step 1: Clone the Repository
```bash
git clone https://github.com/onurtamer/friday-inference-engine.git
cd friday-inference-engine
```

### Step 2: Download & Convert a Model
Download the official GGUF weight for `Qwen2.5-14B-Instruct-Q4_0` (or any compatible model) and run the converter:
```bash
python converter.py --input Qwen2.5-14B-Instruct-Q4_0.gguf
```
*This produces `friday_aligned_int4.bin`, `tensor_map.json`, `model_params.json`, and `tokenizer_config.json`.*

### Step 3: Build the Engine
Build the project in `Release` mode using CMake:
```bash
cmake -B build -A x64
cmake --build build --config Release
```
*The compiled binary will be located at `build/Release/FridayEngine.exe`.*

### Step 4: Run Inference
Launch the engine with your preferred VRAM allocation:
```bash
# Auto-detect available VRAM:
.\build\Release\FridayEngine.exe

# Or cap the VRAM budget (e.g., 6.0 GB):
.\build\Release\FridayEngine.exe --vram 6.0 friday_aligned_int4.bin
```

### Step 5: Run the Tests
```bash
cmake -B build -A x64 -DFRIDAY_MODEL_DIR=<dir with friday_aligned_int4.bin, tensor_map.json, tokenizer_config.json>
cmake --build build --config Release
ctest --test-dir build -C Release -L unit    # CPU only, < 10 s
ctest --test-dir build -C Release -L gpu     # GPU kernels vs. gguf reference (needs Python gguf + numpy)
ctest --test-dir build -C Release -L model   # end-to-end regression snapshot, ~1 min
```
Tests labelled `known-failing` document open defects and are expected to fail until fixed. The `model` label also includes a CPU-only tokenizer test against reference Qwen2 token ids; it needs a `tokenizer_config.json` with BPE merges (written by `converter.py` or `tools/export_tokenizer.py`).

---

## ⚙️ Command-Line Interface (CLI)

| Option | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| `--vram` | `<float>` | Maximum GPU VRAM to allocate for weights and buffers (in GB). | `0.0` (Auto-detect free VRAM) |
| `<model_path>`| `<string>` | Path to the 4KB sector-aligned binary model file. | `friday_aligned_int4.bin` |

---

## 📊 Technical Verification & Results

Measured by the author on **NVIDIA GeForce RTX 3060 12 GB (desktop) / 16 GB DDR4 RAM**, with the engine's VRAM budget capped at **6.0 GB** (`--vram 6.0`). This emulates a 6 GB budget; it is not a run on a physical 6 GB card (driver reservations and display usage differ). A reproducible benchmark script is planned (roadmap R5); throughput figures will be published with it.
- **Model:** `Qwen2.5-14B-Instruct` (48 Layers, Hidden Dim: 5120, Intermediate: 13824, Heads: 40, KV Heads: 8, Vocab: 152064).
- **VRAM Footprint:** Runs within the **6.0 GB VRAM budget**.
- **Execution:** Layers 0–20 reside permanently in GPU VRAM; Layers 21–47 are streamed from Host Pinned RAM via dual-stream ping-pong buffers.
- **Latency:** Zero disk I/O during generation; smooth real-time token streaming to console with coherent, logically sound outputs.

---

## 🎯 Design Philosophy

> *"High-performance systems programming is not about adding more abstractions; it is about eliminating unnecessary ones until only compute and memory remain."*

FRIDAY Engine explores how far 14-billion-parameter models can be pushed on modest consumer hardware when framework overhead is removed.

---

## 📄 License
This project is licensed under the [Apache License 2.0](LICENSE).

## 🙏 Acknowledgements
AI-assisted tooling was used during development.

