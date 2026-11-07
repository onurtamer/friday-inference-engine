"""Export only the tokenizer from a GGUF file (does not touch model weights).

Usage: python tools/export_tokenizer.py <model.gguf> <tokenizer_config.json>
"""
import os
import sys

import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_tokenizer import write_tokenizer_config  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data = write_tokenizer_config(gguf.GGUFReader(sys.argv[1], "r"), sys.argv[2])
    print(f"vocab={data['vocab_size']} merges={len(data['merges'])} special={len(data['special_tokens'])} "
          f"bos={data['bos_token_id']} eos={data['eos_token_id']} -> {sys.argv[2]}")
