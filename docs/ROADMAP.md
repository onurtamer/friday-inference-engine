# FRIDAY roadmap

This is a living document: it will change as we learn, and its history is
tracked in git. Decisions that change v1 scope are recorded as ADRs in
`docs/decisions/` before this file is updated.

Every milestone ends with: tests passing (also under NDEBUG), measurements
recorded, docs updated, commit reviewed.

## Priorities (2026-11-07)

- **P0 — correctness blockers (R1):** token embedding for K-quant types;
  a real BPE tokenizer matching the reference tokenizer; golden and kernel
  tests running under CTest.
- **P1 — decode throughput (R4, CUDA path first):** reach at least 60% of the
  computed ceiling (VRAM bandwidth for resident layers + PCIe for streamed
  layers). Baseline measured 2026-11-07 on RTX 3060 12 GB with
  Qwen2.5-32B Q2_K, 55/64 layers resident: 1.5–1.6 tok/s decode against an
  estimated ceiling of ~10 tok/s.
- **Order:** R0 → R1 → R4 (CUDA) → R5 → R2 → R3 → R6 → R7 → R8.
  ARM-side optimizations follow with the backend HAL in R3.

## Progress

- **R0 — done (2026-11-07).** Clean clone builds; CTest suite (unit/gpu/model).
- **R1 — in progress.**
  - [x] BPE tokenizer matches the reference Qwen2 tokenizer (20/20 ids).
  - [x] Kernel tests vs. gguf reference dequantization (Q2_K/Q3_K/Q4_K/Q6_K, Q2_K embedding).
  - [x] Logits vs. a reference runtime: llama.cpp b12189, Qwen2.5-32B Q2_K, 32/32
        greedy tokens identical, top-1 log-prob diff median 0.002 / max 0.28.
  - [ ] Embeddings for the remaining K-quant types; kernel launch error checks.
  - [ ] ASAN + compute-sanitizer clean; build warnings (C4297, LNK4098).
- **R4 baseline (measured 2026-11-07, RTX 3060 12 GB, Qwen2.5-32B Q2_K):**
  FRIDAY 1.5–1.7 tok/s decode, 1.3–1.9 tok/s prefill (55/64 layers on GPU);
  llama.cpp b12189 8.3 tok/s decode, 73.5 tok/s prefill (50/64 layers on GPU).
  Goal: exceed llama.cpp on the same hardware and model.

## v1.0 — public local inference engine

| ID | Milestone | Exit gate |
|----|-----------|-----------|
| R0 | Repo hygiene & reproducible build | Clean clone builds via CMake; tests run under CTest; LICENSE, honest README; stale scripts moved out; secret/personal-path scan clean |
| R1 | Correctness foundation | Per-model golden tests; tokenizer matches reference byte-for-byte; kernel tests against GGUF reference dequantization for every supported quant type; logits within tolerance of a reference runtime; KV layout bug fixed; ASAN + compute-sanitizer clean |
| R2 | Model loading | Direct GGUF parsing (no tensor_map/convert step); model registry with hash, license, architecture; Qwen2.5/3 + Llama/Mistral |
| R3 | Backend + platform HAL (Windows and Linux) | Backend interface; CUDA backend from existing kernels; CPU backend (AVX2 + NEON); capability report; ARM cross-compile + QEMU tests |
| R4 | Measured optimizations | Profile first; GPU attention/KV (currently CPU), non-blocking streams, bandwidth-efficient GEMV, CUDA Graphs and kernel fusion, batched prefill, kernel-bypass memory/IO; decode ≥60% of the computed ceiling; per-platform before/after tok/s and numerical diff |
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
