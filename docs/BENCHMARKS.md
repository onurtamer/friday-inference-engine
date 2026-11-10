# Benchmarks

How FRIDAY's speed is measured, how to reproduce it, and what the numbers mean.

## Tools

| Tool | What it does |
| :--- | :--- |
| `friday-bench` (`tools/bench/friday_bench.cpp`) | Runs the engine through its public API (`friday::Engine`). Each run is a new sequence: a fixed prompt of `--prompt` tokens (one step per token), then `--gen` greedy tokens. Writes a table and a JSON with every raw value. |
| `tools/bench/llama_cpp_bench.py` | Starts `llama-server`, sends **the same prompt token ids** (taken from the friday-bench JSON), greedy, `n_predict = gen`, `ignore_eos`, and records llama-server's own timings. |
| `tools/bench/estimate.py` + `hardware.json` | Bandwidth model for hardware we have not measured (see [Estimates](#estimates)). |

```bash
# FRIDAY (5 runs + 1 warmup, 512-token prompt, 128 generated tokens)
friday-bench model.gguf --runs 5 --prompt 512 --gen 128 --json friday.json --label "rtx3060"
# llama.cpp on the same tokens
python tools/bench/llama_cpp_bench.py <path>/llama-server model.gguf friday.json llama.json --ngl 99
```

## What is recorded

Per run: prefill and decode tok/s, per-token decode latency (p50, p95, max),
whole-system CPU busy % during a 1 s window **before** the run (background
load) and during it, free RAM before and during, and every 200 ms the GPU SM and
memory clocks, temperature, power and utilization (NVML, loaded at run time).
Per file: date, git revision, model (name, architecture, quantization, file
size, SHA-256 if registered), options, engine summary (backend, GPU/CPU layer
split, KV format), CPU model and features, RAM, GPU name, driver, maximum
clocks and PCIe link, the prompt token ids, and whether the generated tokens
were identical across runs.

## Rules for published numbers

1. **Same session only.** Clocks of a desktop GPU vary with temperature and
   power state; on the RTX 3060 dev box decode differed by ~15% between days.
   FRIDAY and llama.cpp are measured back to back, on the same prompt tokens.
2. **Quiet machine.** Background CPU load (recorded before each run) should be
   under ~10%, the GPU idle before the run, no other GPU applications. Runs that
   violate this are discarded, not averaged.
3. **Median of at least 5 runs** after 1 warmup run; min/max and p95 token
   latency are reported with it.
4. **Same work.** Identical prompt tokens and generated length; greedy decoding
   for both. Output identity across runs is checked (FRIDAY's tool exits with
   status 1 if it differs).

## What the decode numbers include

- FRIDAY: wall time of `Engine::step` per generated token, i.e. GPU/CPU compute,
  the copy of the logits to host memory and their conversion to fp32.
  Sampling (greedy argmax) is outside the timed region.
- llama.cpp: `timings.predicted_per_second` reported by llama-server.
- Prefill: `Engine::prefill` (int8 tensor-core GEMM, compute capability 8.0+;
  otherwise one step per token). The tables below up to 2026-11-10 were measured
  one token at a time. First batched version (2026-11-10, Llama 3.1 8B, 512
  tokens, 3 runs): 465 tok/s vs 51 before; llama.cpp 1944. Attention still runs
  once per token and CPU layers of a split model still run token by token.

## Measured results

RTX 3060 12 GB (PCIe 3.0 x16, driver 610.74), Ryzen 5 5600X, 16 GB DDR4-3200,
Windows 11; 2026-11-09, background CPU load 5-13% before runs; FRIDAY at
git 29535da, llama.cpp b12189-80d2c0c2c. Prompt 512 tokens, 128 generated,
greedy, 1 warmup + 5 runs (3 where noted). Raw JSON: `docs/bench/2026-11-09-rtx3060/`.

| Model | Engine | Layers GPU/CPU | Decode tok/s (median, min-max) | p95 token | Prefill tok/s |
| :--- | :--- | :--- | :--- | ---: | ---: |
| Llama 3.1 8B Q4_K_M | FRIDAY | 32/0, fp16 KV | 50.0 (49.0-50.5) | 22.0 ms | 51 |
| | llama.cpp | 32/0 | 55.4 (54.2-55.7) | | 1944 |
| Qwen3 8B Q4_K_M | FRIDAY | 36/0, fp16 KV | 47.4 (46.4-47.6) | 23.1 ms | 49 |
| | llama.cpp | 36/0 | 53.2 (52.9-55.9) | | 1816 |
| Qwen2.5-32B Q2_K | FRIDAY `--kv auto` (int8), ctx 2048 | 58/6 | 10.66 (10.52-10.71) | 100.6 ms | 11.1 |
| | FRIDAY ctx 704, `--vram 11.8` (3 runs) | 58/6, fp16 KV | 10.85 (10.85-10.93) | 95.1 ms | |
| | llama.cpp `-ngl 58`, `-c 704` (3 runs) | 58/6 | 11.66 | | 325 |

llama.cpp `-ngl` sweep on Qwen2.5-32B (3 runs each, `-c 704`): 50: 8.27,
54: 9.61, 56: 10.99, **58: 11.66**, 60: 10.19, 62: 7.81 tok/s (above 58 the
weights no longer fit and it slows down). Comparisons must use each engine at
its best setting; an earlier comparison at `-ngl 50` overstated FRIDAY.

Short context (64-token prompt, 32 generated), Qwen2.5-32B, interleaved rounds
FRIDAY / llama.cpp `-ngl 58`, 3 runs each: 9.18 / 13.15, 12.20 / 13.26,
12.16 / 13.25 tok/s. FRIDAY's split GPU+CPU runs vary between rounds (cause not
identified yet; GPU clocks and CPU load were normal), llama.cpp's do not.

Follow-up 2026-11-10 (after GQA split-K attention, 8217fa4; same session,
512-token prompt): Llama 3.1 8B 57.6 vs 62.4 tok/s (92%), Qwen3 8B 55.7 vs 60.6
(92%), Qwen2.5-32B 11.8 vs 13.6 (87%, llama.cpp `-ngl 58`). `FRIDAY_PROFILE=1`
splits a Qwen2.5-32B token into GPU layers 42.7 ms (58 layers, ~250 GB/s), CPU
layers 36.2 ms (6 layers, ~29 GB/s) and tail 2.0 ms. The run-to-run instability
seen on 2026-11-09 did not reproduce in 6 processes on 2026-11-10 (12.4-12.7
tok/s at short context), including runs started right after llama-server.

Findings to act on: FRIDAY is ~10% slower than llama.cpp at a 512-640 token
context on all three models and closer at short context (~2% on the 8B models
in earlier same-session runs), which points at attention cost growing with
context; split GPU+CPU runs are not stable; prefill needs batching (R4b).

Split-run instability, 2026-11-10 (Qwen2.5-32B golden run, `FRIDAY_PROFILE=1`):
the slow runs are slow in the GPU phase (48-112 ms per token instead of ~45 ms)
and happen when VRAM is nearly full, whether the last ~340 MiB hold weights or
allocated but untouched buffers: 9 of 17 runs slow with VRAM full, 0 of 17 with
~340 MiB left free (same 56 GPU layers; cause inside the driver not verified).
The planner now keeps room for two layer-sized prefill staging buffers when a
model is split, which are allocated only while a batched prefill runs. On this
model the default split moves from 58/6 to 56/8: CPU phase ~35 -> ~49 ms, but
6 of 6 runs stable at ~9.8 tok/s (golden harness) instead of 6.3-10.5.
`FRIDAY_PREFILL_STAGE=0` restores the old split.

## Estimates

`estimate.py` predicts single-stream decode from memory traffic:

```
t_token = gpu_bytes / (gpu_bw * e_gpu) + cpu_bytes / (cpu_bw * e_cpu)
          + t_fixed + n_cpu_layers * t_cpu_layer (+ t_handoff if the model is split)
```

The layer split follows the engine's planner. Efficiencies and fixed costs are
calibrated on the RTX 3060 + Ryzen 5 5600X (see `hardware.json`); on that
machine the model reproduces the measured results within about 3% (Qwen2.5-32B
57/7 split 12.1 vs 12.25 tok/s; Llama 3.1 8B 59.7 vs 59.6; Qwen3 8B 59.0 vs
58.1 on the GPU and 6.7 vs 6.8 on the CPU). Bandwidths of other devices are
spec-sheet peaks. **These are estimates, not measurements**: expect +-20% on
other NVIDIA GPUs, and treat ARM numbers as an upper bound until the NEON
kernels exist and are measured on the device.

Decode tok/s, `--kv f16`, context 2048 (estimate unless marked measured):

| Hardware | Qwen2.5-32B Q2_K | Llama 3.1 8B Q4_K_M | Qwen3 8B Q4_K_M |
| :--- | ---: | ---: | ---: |
| RTX 3060 12 GB + DDR4-3200 (measured device) | 12.1 (57/7) | 59.7 | 59.0 |
| RTX 4060 Ti 16 GB + DDR5-6000 | 19.3 | 48.5 | 48.0 |
| RTX 4070 12 GB + DDR5-6000 | 18.1 (57/7) | 81.1 | 80.2 |
| RTX 3090 24 GB | 59.5 | 138 | 137 |
| RTX 4090 24 GB | 63.7 | 147 | 145 |
| CPU only, DDR4-3200 dual channel | 2.7 | 6.9 | 6.7 |
| CPU only, DDR5-6000 dual channel | 4.8 | 12.2 | 11.9 |
| Radxa Orion O6 (CIX P1, LPDDR5, 64 GB) | 4.8 | 12.2 | 11.9 |
| Raspberry Pi 5 8 GB | does not fit | 2.4 | 2.3 |

ARM rows assume spec bandwidth (unverified) and NEON kernels as efficient as the
AVX2 ones, which are not written yet (roadmap R3).
