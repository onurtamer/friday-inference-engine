"""Measure llama.cpp (llama-server) on the same prompt tokens as a friday-bench run.

Starts llama-server on 127.0.0.1, sends the prompt token ids recorded in a
friday-bench JSON file (greedy, n_predict = gen tokens), and writes a JSON with
llama-server's own timings (prompt and decode tok/s) for N runs plus warmup,
and nvidia-smi clocks/temperature before and after each run when available.

Usage:
  python tools/bench/llama_cpp_bench.py <llama-server> <model.gguf> <friday-bench.json> <out.json>
         [--ngl N] [--runs N] [--warmup N] [--port P] [--extra "args for llama-server"]

Defaults: --ngl 99, runs/warmup as in the friday-bench file, --port 8089.
Server flags always include -c (prompt + gen + 64), -np 1, --load-mode none
(memory-mapped loading can exhaust RAM on 16 GB machines with large models).
"""
import argparse
import json
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request


def http_json(url, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gpu_sample():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        sm, mem, temp, pw = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"sm_mhz": float(sm), "mem_mhz": float(mem), "temp_c": float(temp), "power_w": float(pw)}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server")
    ap.add_argument("model")
    ap.add_argument("friday_json")
    ap.add_argument("out")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--runs", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()

    fb = json.load(open(a.friday_json, encoding="utf-8"))
    prompt = fb["prompt_ids"]
    gen = fb["options"]["gen_tokens"]
    runs = a.runs or fb["options"]["runs"]
    warmup = a.warmup if a.warmup is not None else fb["options"]["warmup"]
    ctx = len(prompt) + gen + 64

    cmd = [a.server, "-m", a.model, "-ngl", str(a.ngl), "-c", str(ctx), "-np", "1", "--load-mode", "none",
           "--host", "127.0.0.1", "--port", str(a.port)] + shlex.split(a.extra)
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = "http://127.0.0.1:%d" % a.port
        for _ in range(600):
            try:
                if http_json(base + "/health", timeout=5).get("status") == "ok":
                    break
            except Exception:
                pass
            if srv.poll() is not None:
                sys.exit("llama-server exited during startup")
            time.sleep(1)
        else:
            sys.exit("llama-server did not become ready")
        props = {}
        try:
            props = http_json(base + "/props", timeout=10)
        except Exception:
            pass

        body = {"prompt": prompt, "n_predict": gen, "temperature": 0.0, "top_k": 1, "seed": 42,
                "cache_prompt": False, "return_tokens": True, "samplers": ["top_k"], "ignore_eos": True}
        results, outputs = [], []
        for r in range(-warmup, runs):
            g0 = gpu_sample()
            resp = http_json(base + "/completion", body)
            g1 = gpu_sample()
            t = resp.get("timings", {})
            if r < 0:
                print("warmup: prompt %.2f tok/s, decode %.2f tok/s" % (t.get("prompt_per_second", 0), t.get("predicted_per_second", 0)))
                continue
            outputs.append(resp.get("tokens"))
            results.append({"run": r, "prefill_tok_s": t.get("prompt_per_second"), "decode_tok_s": t.get("predicted_per_second"),
                            "decode_ms_per_token": t.get("predicted_per_token_ms"), "gpu_before": g0, "gpu_after": g1})
            print("run %d: prefill %.2f tok/s | decode %.2f tok/s" % (r + 1, results[-1]["prefill_tok_s"], results[-1]["decode_tok_s"]))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()

    dec = [x["decode_tok_s"] for x in results]
    pre = [x["prefill_tok_s"] for x in results]
    summary = {"decode_tok_s_median": statistics.median(dec), "decode_tok_s_min": min(dec), "decode_tok_s_max": max(dec),
               "prefill_tok_s_median": statistics.median(pre),
               "output_identical_across_runs": all(o == outputs[0] for o in outputs)}
    json.dump({"tool": "llama_cpp_bench.py", "server_cmd": cmd, "build": props.get("build_info", ""),
               "model": a.model, "prompt_tokens": len(prompt), "gen_tokens": gen, "runs": results, "summary": summary,
               "friday_bench_file": a.friday_json}, open(a.out, "w"), indent=2)
    print("decode: median %.2f tok/s | prefill: median %.2f tok/s" % (summary["decode_tok_s_median"], summary["prefill_tok_s_median"]))
    print("json:", a.out)


if __name__ == "__main__":
    main()
