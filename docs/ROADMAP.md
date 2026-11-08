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
- **Order:** R0 → R1 → R4 steps 1–3 (done) → R2 → R3 → R4 → R5 → R4b → R6 → R7 → R8
  (re-ordered 2026-11-08 after R1: model loading and backends before the remaining tuning).
  ARM-side optimizations follow with the backend HAL in R3.

## Progress

- **R0 — done (2026-11-07).** Clean clone builds; CTest suite (unit/gpu/model).
- **R1 — done (2026-11-08).**
  - [x] BPE tokenizer matches the reference Qwen2 tokenizer (20/20 ids).
  - [x] Kernel tests vs. gguf reference dequantization (Q2_K/Q3_K/Q4_K/Q6_K, Q2_K embedding).
  - [x] Logits vs. a reference runtime: llama.cpp b12189, Qwen2.5-32B Q2_K, 32/32
        greedy tokens identical, top-1 log-prob diff median 0.002 / max 0.28.
  - [x] Embeddings for every K-quant type (Q2_K/Q3_K/Q4_K/Q6_K, plus Q4_0/F16),
        tested against gguf dequantization; GEMV and attention launches checked;
        unimplemented FP16/TERNARY GEMV now fail loudly instead of returning garbage.
  - [x] compute-sanitizer (memcheck, racecheck, initcheck, synccheck) clean on the
        GPU tests and on a 2-token model run (memcheck, racecheck; initcheck needs more host RAM than this 16 GB machine has with the 32B model); ASAN (MSVC) clean on unit/GPU tests and the full model golden test.
  - [x] Build warnings fixed (C4297, LNK4098, C4005); CMake sm_86 default was dead
        code (set after `project()`, everything built for sm_52) — fixed.
- **R2 — done (2026-11-08)** (Mistral deferred, see below).
  - [x] R2a: direct GGUF loading. Bounds-checked C++ GGUF reader (unit test vs.
        gguf-py, 9 malformed files rejected), hyper-parameters and tokenizer from
        metadata, per-tensor packed layer buffers (GGUF layers are not always
        contiguous). Qwen2.5-32B from the `.gguf`: golden PASS and per-step
        log-probs bit-identical to the converted `.bin`. converter.py deprecated.
  - [x] R2b: model registry (`models list/add/remove/verify`, run by id): path,
        size, SHA-256 (self-contained, NIST vectors; cached by size + mtime;
        matches certutil on the 12 GB model), architecture, name, license, quant,
        parameter count. JSON in the user config dir, atomic writes.
  - [x] R2c (Llama): `llama` architecture — GGML normal-mode RoPE with base and
        Llama 3 frequency factors from the GGUF, no qkv bias, `llama-bpe`
        pre-tokenizer (digit groups of 3, BOS, ignore_merges): 20/20 ids vs.
        llama-tokenize. Q5_K/Q6_K/Q8_0 on the CPU (AVX2) and as dp4a GPU kernels
        (Q6_K 176-205 -> ~318 GB/s), plus embedding rows. Golden vs. llama.cpp for
        Meta-Llama-3.1-8B-Instruct Q4_K_M (all-GPU and 9 GPU / 23 CPU layers):
        32/32, max top-1 diff 0.035. Golden check now uses teacher forcing on
        the reference tokens and reports reference near-ties (< 0.1 nats).
        Numerics: Q6_K now uses int8 activations; Qwen2.5 max diff 0.271 -> 0.278.
  - [x] R2c (Qwen3): per-head q/k RMSNorm (GPU kernel + CPU path). Tokenizer
        20/20 vs. llama-tokenize; golden vs. llama.cpp for Qwen3-8B Q4_K_M
        (all-GPU and 9 GPU / 27 CPU layers): 32/32, max top-1 diff 0.019.
        Mistral (needs a SentencePiece tokenizer) moved to after R3 (2026-11-08).
- **R3 — in progress.**
  - [x] Backend split: `friday::Engine` API (host logits, no device memory in
        callers); `ModelLayout` (tensors + packed layers) and `CpuLayerRunner`
        shared by both backends; `io_manager` CUDA-free. CUDA path bit-identical
        to before (Qwen2.5 57/7, Qwen3 9/27, Llama 9/23 GPU/CPU layers).
  - [x] CPU-only engine: embedding, layers and LM head on the CPU
        (`--backend cpu` / `FRIDAY_BACKEND=cpu`); CPU-only build with
        `-DFRIDAY_ENABLE_CUDA=OFF` (no CUDA DLLs). Golden vs. llama.cpp on the
        CPU: Llama 3.1 8B (max diff 0.052) and Qwen3 8B (0.020), ~6.9 tok/s.
  - [x] Capability report `FridayEngine --info` (CPU features, RAM, CUDA devices,
        VRAM, PCIe link via NVML loaded at run time).
  - [ ] Linux build and tests (x86-64, WSL): needs a toolchain in WSL.
  - [ ] ARM64: NEON kernels, cross-compile, QEMU and real-device tests —
        deferred to the end of R3 (no ARM device yet; Raspberry Pi as interim).
- **R4 baseline (measured 2026-11-07, RTX 3060 12 GB, Qwen2.5-32B Q2_K):**
  FRIDAY 1.5–1.7 tok/s decode, 1.3–1.9 tok/s prefill (55/64 layers on GPU);
  llama.cpp b12189 8.3 tok/s decode, 73.5 tok/s prefill (50/64 layers on GPU).
  Goal: exceed llama.cpp on the same hardware and model.
  - Step 1 (2026-11-08): warp-per-row Q6_K GEMV for the LM head (was 225 ms/token,
    thread-per-row, uncoalesced) → 2.42 tok/s decode, 2.78 tok/s prefill.
    Remaining per-token cost: CPU attention + copies ~190 ms, MLP GEMV ~171 ms
    (~16% of memory bandwidth), PCIe streaming ~42 ms.
  - Step 2 (2026-11-08): GPU decode attention + fp16 KV cache on the GPU
    (`--ctx`, default 2048 = 512 MiB) → 3.44 tok/s decode, 3.52 tok/s prefill
    (median of 3; 52/64 layers on GPU). Golden v2 unchanged (32/32, max 0.28).
  - Step 3a: Q2_K/Q3_K/Q4_K GEMV with int8 activations + `__dp4a` (80–90% of VRAM
    bandwidth, was 14–28%) → 5.36 tok/s. Numerics: activation quantized per 32
    values, as in llama.cpp's mmvq; golden unchanged.
  - Step 3b: measured PCIe 3.0 x16 at 13 GB/s vs host RAM ~38 GB/s, so streaming
    weights every token loses to computing on the CPU. Non-resident layers now run
    on the CPU (AVX2 K-quant GEMV, same int8 activations and fp16 rounding points);
    no DMA slots, KV cache per GPU layer only → 9.71 tok/s (56 GPU / 8 CPU).
  - Step 3c: embedding table in host RAM (one row uploaded per token, numerics
    identical), 8 CPU threads → **10.55 tok/s decode, 10.2 tok/s prefill** (median
    of 3, 57 GPU / 7 CPU). Golden 32/32, max top-1 diff 0.27. llama.cpp: 8.3 / 73.5.
  - Step 4 (2026-11-09), same-session medians on the RTX 3060:
    4a fused GPU layer (int8-activation pipeline: RMSNorm+quantize, SiLU*up+
    quantize, gate+up in one launch, bias/residual in the GEMV epilogue, RoPE+KV
    write; ~27 -> ~12 launches per layer), bit-identical; 4b CUDA Graph of the
    resident layers (position read on the device), bit-identical; together
    Llama 3.1 8B 47.1 -> 51.9 tok/s, Qwen3 8B 45.6 -> 50.0. 4c split-K decode
    attention: +13% decode at ~1500 context, neutral at short context;
    numerics change slightly (golden still passes, Qwen2.5 max diff 0.278 ->
    0.264). 4d optional int8 KV (`--kv q8`): Qwen2.5-32B 58/64 layers on the GPU,
    12.04 -> 12.73 tok/s; Llama max diff 0.035 -> 0.073 (default stays f16).
    Versus llama.cpp in the same session (engine decode time): Llama 3.1 8B
    59.6 vs 61.0 tok/s, Qwen3 8B 58.1 vs 58.9.
  - Evaluated next (2026-11-08, measured with microbenchmarks):
    CUDA Graph of the GPU phase −4.5 ms/token (45.3 → 40.8 ms; GEMV-only floor
    39 ms); split-K decode attention: +7 ms at 1024 context today, mostly
    recoverable; naive batched dp4a prefill only 1.4× at B=16 (compute-bound) —
    prefill needs tensor-core int8 GEMM; 8-bit KV would free ~220 MiB (+1 GPU
    layer, ~−5.5 ms/token).

## v1.0 — public local inference engine

| ID | Milestone | Exit gate |
|----|-----------|-----------|
| R0 | Repo hygiene & reproducible build | Clean clone builds via CMake; tests run under CTest; LICENSE, honest README; stale scripts moved out; secret/personal-path scan clean |
| R1 | Correctness foundation | Per-model golden tests; tokenizer matches reference byte-for-byte; kernel tests against GGUF reference dequantization for every supported quant type; logits within tolerance of a reference runtime; KV layout bug fixed; ASAN + compute-sanitizer clean |
| R2 | Model loading | Direct GGUF parsing (no tensor_map/convert step); model registry with hash, license, architecture; Qwen2.5/3 + Llama/Mistral |
| R3 | Backend + platform HAL (Windows and Linux) | Backend interface; CUDA backend from existing kernels; CPU backend (AVX2 + NEON); capability report; ARM cross-compile + QEMU tests |
| R4 | Measured optimizations (decode) | Done: GPU attention/KV, bandwidth-efficient GEMV, CPU-computed layers. Remaining: split-K decode attention, CUDA Graphs and kernel fusion, 8-bit KV cache; decode ≥60% of the computed ceiling (GPU + CPU layers); per-platform before/after tok/s and numerical diff |
| R4b | Prefill & I/O | Batched prefill (tensor-core int8 GEMM on the GPU, batched CPU GEMM, causal batched attention); kernel-bypass memory/IO (unbuffered/direct reads, large pages); prefill tok/s vs llama.cpp on the same machine |
| R5 | Benchmarks & performance model | Controlled benchmark tool (N repeats, median/p95, records free RAM, GPU clocks and background load); measured table (RTX 3060, Ryzen 5600X); estimate table for other GPUs/ARM boards; published methodology |
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
