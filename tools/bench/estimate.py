"""Bandwidth-based decode estimate for FRIDAY on hardware we have not measured.

Single-stream decode reads every weight once per token, so its time is
dominated by memory traffic:
    t_token = gpu_bytes / (gpu_bw * e_gpu) + cpu_bytes / (cpu_bw * e_cpu)
              + t_fixed + n_cpu_layers * t_cpu_layer (+ t_handoff if split)
gpu_bytes: GPU-resident layers + LM head; cpu_bytes: layers that do not fit.
The layer split follows the engine's planner (VRAM minus a 384 MiB reserve,
the LM head and the KV cache of each GPU layer). Efficiencies and fixed costs
are calibrated on the measured RTX 3060 + Ryzen 5 5600X results in
hardware.json; other devices use spec-sheet bandwidths, so every result is an
ESTIMATE (expected error +-20%, larger for ARM until NEON kernels are measured).

Usage:
  python tools/bench/estimate.py <model.gguf> [--hw tools/bench/hardware.json] [--ctx 2048] [--kv f16|q8]
"""
import argparse
import json
import os
import re
import sys

from gguf import GGUFReader


def model_bytes(path):
    r = GGUFReader(path)
    layers = {}
    head = embd = other = 0
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()

    def u(key):
        f = r.fields.get(key)
        return None if f is None else int(f.parts[f.data[0]][0])

    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            layers[int(m.group(1))] = layers.get(int(m.group(1)), 0) + int(t.n_bytes)
        elif t.name == "output.weight":
            head = int(t.n_bytes)
        elif t.name == "token_embd.weight":
            embd = int(t.n_bytes)
        else:
            other += int(t.n_bytes)
    if head == 0:
        head = embd  # tied embeddings: the LM head reads the embedding table
    n_layers = len(layers)
    kv_heads = u(arch + ".attention.head_count_kv") or u(arch + ".attention.head_count")
    head_dim = u(arch + ".attention.key_length") or (u(arch + ".embedding_length") // u(arch + ".attention.head_count"))
    return {"layers": [layers[i] for i in range(n_layers)], "head": head, "other": other,
            "kv_dim": kv_heads * head_dim, "name": os.path.basename(path)}


def estimate(m, hw, cal, ctx, kv):
    kv_bytes_per_layer = 2 * ctx * m["kv_dim"] * (1 + 2 / 32 if kv == "q8" else 2)
    layers = m["layers"]
    if hw.get("vram_gb"):
        budget = hw["vram_gb"] * 2**30 * cal["usable_vram_fraction"] - 384 * 2**20 - m["head"] - m["other"]
        n_gpu = 0
        for b in layers:
            if budget < b + kv_bytes_per_layer:
                break
            budget -= b + kv_bytes_per_layer
            n_gpu += 1
        gpu_bytes = sum(layers[:n_gpu]) + m["head"]
        cpu_bytes = sum(layers[n_gpu:])
        n_cpu = len(layers) - n_gpu
        t = gpu_bytes / (hw["gpu_bw_gbs"] * 1e9 * cal["e_gpu"])
        t += cpu_bytes / (hw["cpu_bw_gbs"] * 1e9 * cal["e_cpu"]) if n_cpu else 0.0
        t += cal["t_fixed_gpu_ms"] / 1e3 + n_cpu * cal["t_cpu_layer_ms"] / 1e3
        t += cal["t_handoff_ms"] / 1e3 if n_cpu else 0.0
    else:  # CPU-only device (ARM boards, no CUDA)
        n_gpu, n_cpu = 0, len(layers)
        t = (sum(layers) + m["head"]) / (hw["cpu_bw_gbs"] * 1e9 * cal["e_cpu"])
        t += cal["t_fixed_cpu_ms"] / 1e3 + n_cpu * cal["t_cpu_layer_ms"] / 1e3
    return n_gpu, n_cpu, 1.0 / t


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--hw", default=os.path.join(here, "hardware.json"))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--kv", default="f16", choices=["f16", "q8"])
    a = ap.parse_args()
    data = json.load(open(a.hw, encoding="utf-8"))
    m = model_bytes(a.model)
    total = (sum(m["layers"]) + m["head"] + m["other"]) / 2**30
    print("%s: %d layers, %.2f GiB (estimates, see tools/bench/estimate.py)" % (m["name"], len(m["layers"]), total))
    print("%-40s %-13s %10s  %s" % ("hardware", "layers G/C", "decode", "basis"))
    for hw in data["hardware"]:
        if hw.get("min_ram_gb") and hw["min_ram_gb"] < total + 1:
            print("%-40s %-13s %10s  %s" % (hw["name"], "-", "n/a", "model does not fit in RAM"))
            continue
        n_gpu, n_cpu, tps = estimate(m, hw, data["calibration"], a.ctx, a.kv)
        print("%-40s %-13s %7.1f t/s  %s" % (hw["name"], "%d/%d" % (n_gpu, n_cpu), tps, hw.get("basis", "spec sheet")))


if __name__ == "__main__":
    sys.exit(main())
