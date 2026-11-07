# ADR 0003: v1 product definition

- Status: accepted
- Date: 2026-11-07
- Amends: ADR 0001 (items 1, 5); earlier internal scope notes (product
  definition, "no listening port" rule)
- Keeps: ADR 0002 (external gateway; deferred to v2)

## Context

FRIDAY v1 is redefined from an orchestration prototype into a publishable
local inference engine. The repository is public under Apache-2.0.

## Decision

1. v1 scope: model loading (GGUF), tokenizer, chat, persistent memory,
   measured optimizations, benchmark suite, and an API server/router.
2. Backends: NVIDIA CUDA and ARM64 CPU (NEON) are both first-class.
   A CPU reference path is the correctness baseline for every backend.
3. Models in v1.0: Qwen2.5/Qwen3 dense, Llama/Mistral family.
   GPT-OSS (MoE, MXFP4) is v1.1.
4. Execution model: user-space service using kernel-bypass techniques on
   both Linux and Windows, behind a platform layer with per-OS
   implementations:
   - Linux: hugepages, mlock, CPU/NUMA pinning, io_uring + O_DIRECT,
     VFIO/UIO where justified.
   - Windows: large pages (MEM_LARGE_PAGES), VirtualLock, CPU sets,
     IoRing / unbuffered overlapped I/O, MapViewOfFile.
   - Both: CUDA driver API, pinned memory, CUDA Graphs.
   Each feature is reported as a capability; if unavailable (e.g. missing
   privilege) the engine falls back and logs it instead of failing.
   All hardware access sits behind a HAL so a v2 kernel module or a v3
   FRIDAY OS kernel can plug in without rewriting the engine.
   No kernel-mode inference in v1.
5. API: OpenAI-compatible HTTP. Binds to 127.0.0.1 by default; network
   exposure is opt-in and requires token authentication.
6. Performance claims: "measured" only with hardware ID, model hash, config,
   and p50/p95. All other figures are labelled "estimate" and derived from a
   documented bandwidth/compute model calibrated on measured hardware.
7. The RTX 3060 development PC is an allowed build and benchmark machine.
8. Council/multi-agent: not in v1. Session, context, and backend interfaces
   must not preclude it (v2).
9. Performance is measured per platform; WSL2 numbers are not treated as
   native Linux results.

## Consequences

- Every optimization ships with a before/after benchmark and a reported
  numerical diff against the golden reference.
- ARM performance remains "estimate" until measured on a real board;
  ARM correctness is tested via cross-compilation/QEMU earlier.
- FRIDAY OS (v3) may use a custom kernel; that decision is gated on
  measurements showing user-space + kernel-bypass is insufficient.
