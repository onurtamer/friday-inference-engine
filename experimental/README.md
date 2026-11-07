# experimental/

Prototype code kept for reference. **None of it is built by the root CMake
project, and none of it is part of the engine.** Status below is from a source
review on 2026-11-07; items marked "not run" have not been executed.

| Path | What it is | Status |
|------|------------|--------|
| `runtime/` `PagedKvCache` | Paged KV cache with reference-counted copy-on-write pages | Page bookkeeping looks correct. CUDA-only, not thread-safe, copyable (double free risk). The COW copy assumes a `[token][layer][head][dim]` block layout while `paged_attention.cu` indexes `[token][head][dim]` without a layer dimension — inconsistent for >1 layer (not run). |
| `runtime/` `GpuMemoryArena`, `vram_forensics` | VRAM arena and diagnostics | Not reviewed in detail. |
| `runtime/` `FridayRuntime` | Runtime interface | Header only; no implementation. |
| `cognition/` `DialecticalChoir` | Architect / Critic / Alternative branches over `PagedKvCache` | No model calls; `synthesize_consensus()` concatenates strings and prints a fixed verdict. Calling `branch_choir()` twice leaks KV pages. |
| `cognition/` `DeterministicValidatorBridge` | Intended compiler/test/sanitizer bridge | **Placeholder: reports success for any non-empty input without running any tool.** Must not be used as a validator. |
| `cognition/` `ProjectMemoryManager` | Symbol store | In-memory `unordered_map` only; no persistence or provenance. |
| `tests/` | Prototype tests | Most use `assert` (removed under `NDEBUG`). `test_prefill_equivalence` compares two identical constant vectors. `test_cognitive_orchestrator` asserts the placeholder validator's success. `test_golden_equivalence` targets an engine API that no longer exists and contains a hard-coded token override. |
| `golden-legacy/` | Golden prompt and 32 output tokens for Qwen2.5-32B | Produced by an earlier engine version that is no longer available. The current engine generates coherent output for this prompt but diverges at token 3 (near-tied candidates), so this file is **not** used as a pass/fail oracle. A reference-runtime golden (tokens + logits) is planned in roadmap R1. |

Reuse of anything here requires review and tests first; see `docs/ROADMAP.md`.
The council/multi-agent work is scheduled for v2 (ADR 0003).
