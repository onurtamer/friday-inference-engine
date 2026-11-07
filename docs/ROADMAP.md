# FRIDAY roadmap

This is a living document: it will change as we learn, and its history is
tracked in git. Decisions that change v1 scope are recorded as ADRs in
`docs/decisions/` before this file is updated.

Every milestone ends with: tests passing (also under NDEBUG), measurements
recorded, docs updated, commit reviewed.

## v1.0 — public local inference engine

| ID | Milestone | Exit gate |
|----|-----------|-----------|
| R0 | Repo hygiene & reproducible build | Clean clone builds via CMake; tests run under CTest; LICENSE, honest README; stale scripts moved out; secret/personal-path scan clean |
| R1 | Correctness foundation | Per-model golden tests; tokenizer matches reference byte-for-byte; logits within tolerance of a reference runtime; KV layout bug fixed; ASAN + compute-sanitizer clean |
| R2 | Model loading | Direct GGUF parsing (no tensor_map/convert step); model registry with hash, license, architecture; Qwen2.5/3 + Llama/Mistral |
| R3 | Backend + platform HAL (Windows and Linux) | Backend interface; CUDA backend from existing kernels; CPU backend (AVX2 + NEON); capability report; ARM cross-compile + QEMU tests |
| R4 | Measured optimizations | GPU attention/KV (currently CPU), batched prefill, kernel fusion, kernel-bypass memory/IO; per-platform before/after measurements and numerical diff |
| R5 | Benchmarks & performance model | Measured table (RTX 3060, Ryzen 5600X); estimate table for other GPUs/ARM boards; published methodology |
| R6 | Chat & persistent memory | Per-model chat templates; sessions; SQLite + FTS memory with provenance, edit and delete |
| R7 | API server & router | OpenAI-compatible HTTP, SSE streaming, cancel, queue, multi-model routing; localhost default, token auth |
| R8 | Stability & release | Multi-hour soak; fuzzing (GGUF parser, API); structured logs/metrics; semantic versioning; GitHub release |

## v1.1

GPT-OSS (MoE, MXFP4, sliding-window attention), speculative decoding,
additional quantization formats, ARM measured results once a board is owned.

## v2

Council (President / Vice President / Experts) on top of v1 sessions;
COW/paged KV sharing; AMD GPUs (ROCm or Vulkan); NPU backends after SDK
review; external AI gateway per ADR 0002; kernel module only for a
measured bottleneck.

## v3

FRIDAY OS image; custom kernel, gated by measurements; large multi-model
orchestration.

## Hardware

- Development: RTX 3060 12 GB, Ryzen 5 5600X, 16 GB RAM (Windows).
  Linux measurements on this machine require a native Linux install
  (dual boot); WSL2 is not representative.
- Target: Radxa Orion O6 64 GB; fallback Orange Pi 6 Plus or an affordable
  ARM board with 64–128 GB RAM and sufficient memory bandwidth.
