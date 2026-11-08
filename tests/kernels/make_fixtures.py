"""Generate synthetic K-quant test fixtures (no model file needed).

Random quantized blocks with small finite fp16 scales are dequantized with the
gguf reference implementation; y_ref = dequant(W) @ x in float64.
Usage: python make_fixtures.py <out_dir>
"""
import json
import os
import sys

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import quants

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20261107)

# block size in bytes and byte offsets of fp16 scale fields (GGML layouts)
LAYOUT = {
    "q2_k": (T.Q2_K, 84, [80, 82]),    # scales[16] qs[64] d dmin
    "q3_k": (T.Q3_K, 110, [108]),      # hmask[32] qs[64] scales[12] d
    "q4_k": (T.Q4_K, 144, [0, 2]),     # d dmin scales[12] qs[128]
    "q6_k": (T.Q6_K, 210, [208]),      # ql[128] qh[64] scales[16] d
    "q5_k": (T.Q5_K, 176, [0, 2]),     # d dmin scales[12] qh[32] qs[128]
    "q8_0": (T.Q8_0, 34, [0]),         # d qs[32]; 32 values per block
}
VALUES_PER_BLOCK = {"q8_0": 32}


def random_rows(block_bytes, scale_offsets, rows, blocks):
    w = rng.integers(0, 256, size=(rows, blocks, block_bytes), dtype=np.uint8)
    for off in scale_offsets:
        s = rng.uniform(1e-3, 1e-2, size=(rows, blocks)).astype(np.float16)
        w[:, :, off:off + 2] = s.view(np.uint8).reshape(rows, blocks, 2)
    return w.reshape(rows, blocks * block_bytes)


meta = []
K, N = 1024, 64
for case, (qtype, bb, offs) in LAYOUT.items():
    w = random_rows(bb, offs, N, K // VALUES_PER_BLOCK.get(case, 256))
    deq = quants.dequantize(w, qtype).astype(np.float64).reshape(N, K)
    x = rng.standard_normal(K).astype(np.float16)
    (deq @ x.astype(np.float64)).tofile(os.path.join(out, case + "_yref.bin"))
    # Reference with the int8 activation used by the engine's dp4a GEMV:
    # blocks of 32, d = amax / 127 (fp32), q = round-half-even(x * (1 / d)).
    xb = x.astype(np.float32).reshape(-1, 32)
    d = (np.abs(xb).max(axis=1) / np.float32(127)).astype(np.float32)
    inv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, 1), 0).astype(np.float32)
    q = np.clip(np.rint(xb * inv[:, None]), -127, 127)
    xq8 = (q.astype(np.float64) * d.astype(np.float64)[:, None]).reshape(K)
    (deq @ xq8).tofile(os.path.join(out, case + "_yref_q8.bin"))
    w.tofile(os.path.join(out, case + "_w.bin"))
    x.tofile(os.path.join(out, case + "_x.bin"))
    meta.append({"case": case, "K": K, "N": N})

# Q2_K embedding table: dequantize selected rows.
# Embedding tables: dequantize selected rows, for every K-quant type.
V, D = 32, 1024
ids = [0, 7, 31]
for case, (qtype, bb, offs) in LAYOUT.items():
    table = random_rows(bb, offs, V, D // VALUES_PER_BLOCK.get(case, 256))
    ref = quants.dequantize(np.ascontiguousarray(table[ids]), qtype).astype(np.float32).reshape(len(ids), D)
    table.tofile(os.path.join(out, "emb_" + case + "_table.bin"))
    ref.tofile(os.path.join(out, "emb_" + case + "_ref.bin"))
    meta.append({"case": "emb_" + case, "dtype": case.upper(), "K": D, "N": V,
                 "row_bytes": int(table.shape[1]), "ids": ids})

with open(os.path.join(out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("fixtures written to", out)
