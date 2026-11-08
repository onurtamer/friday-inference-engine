"""Token ids from llama.cpp's llama-tokenize for the texts of an existing
reference file (no BOS; special tokens parsed), e.g. for Llama 3.x.

Usage: make_llama_reference.py <llama-tokenize> <model.gguf> <texts_from.json> <out.json>
"""
import json
import os
import subprocess
import sys
import tempfile

exe, model, src, out = sys.argv[1:5]
cases = json.load(open(src, encoding="utf-8"))["cases"]
result = []
with tempfile.TemporaryDirectory() as tmp:
    for c in cases:
        path = os.path.join(tmp, "text.txt")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(c["text"])
        r = subprocess.run([exe, "-m", model, "-f", path, "--ids", "--no-bos", "--log-disable"],
                           capture_output=True, text=True, encoding="utf-8", check=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("[")][-1]
        result.append({"text": c["text"], "ids": json.loads(line)})
vocab = None
with open(out, "w", encoding="utf-8") as f:
    json.dump({"source": "llama.cpp llama-tokenize (" + os.path.basename(exe) + "), " + os.path.basename(model),
               "cases": result}, f, ensure_ascii=False, indent=1)
print("wrote %s (%d cases)" % (out, len(result)))
