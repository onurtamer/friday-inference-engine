"""Write a tiny GGUF file plus malformed variants for test_gguf_file.

Usage: make_gguf_fixtures.py <out_dir>
Writes tiny.gguf, expected.json (what gguf-py reads back) and bad_*.gguf.
"""
import json
import os
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType as T, GGUFReader, GGUFWriter

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "tiny.gguf")

w = GGUFWriter(path, "qwen2")
w.add_string("general.name", "tiny-test")
w.add_uint32("qwen2.block_count", 2)
w.add_uint64("test.u64", 1 << 40)
w.add_int32("test.i32", -7)
w.add_float32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
w.add_float64("test.f64", 0.125)
w.add_bool("test.bool", True)
w.add_array("tokenizer.ggml.tokens", ["a", "b", "<|x|>", "ğz"])
w.add_array("test.ints", [1, -2, 3])
w.add_array("test.floats", [0.5, 1.5])

rng = np.random.default_rng(1)
w.add_tensor("blk.0.attn_norm.weight", np.ones(256, dtype=np.float32))
w.add_tensor("blk.0.ffn_up.weight", rng.integers(0, 256, size=(3, 2 * 84), dtype=np.uint8), raw_dtype=T.Q2_K)
w.add_tensor("token_embd.weight", rng.integers(0, 256, size=(4, 210), dtype=np.uint8), raw_dtype=T.Q6_K)
w.add_tensor("blk.1.attn_norm.weight", np.full(256, 2.0, dtype=np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

r = GGUFReader(path)
exp = {
    "version": 3,
    "alignment": int(r.alignment),
    "data_offset": int(r.data_offset),
    "tensors": [{"name": t.name, "ne": [int(x) for x in t.shape], "type": int(t.tensor_type),
                 "offset": int(t.data_offset), "size": int(t.n_bytes)} for t in r.tensors],
}
with open(os.path.join(out, "expected.json"), "w") as f:
    json.dump(exp, f, indent=1)

data = open(path, "rb").read()


def write(name, blob):
    with open(os.path.join(out, name), "wb") as f:
        f.write(blob)


write("bad_magic.gguf", b"GGUX" + data[4:])
write("bad_version.gguf", data[:4] + struct.pack("<I", 99) + data[8:])
write("bad_kv_count.gguf", data[:16] + struct.pack("<Q", 1 << 40) + data[24:])
for i, cut in enumerate([3, 20, 60, 200, exp["data_offset"] - 8, len(data) - 100]):
    write("bad_trunc_%d.gguf" % i, data[:cut])
print("gguf fixtures written to", out)
