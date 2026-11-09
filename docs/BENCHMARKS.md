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
- Prefill is not comparable yet: FRIDAY processes the prompt one token at a time
  (batched prefill is roadmap R4b), llama.cpp in batches.

## Measured results

To be filled from a quiet-machine session with the tools above (the numbers in
the README are from 2026-11-09, same session, golden-test prompt, median of 3).

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
