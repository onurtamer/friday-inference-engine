"""Generate tokenizer reference fixtures with the HF `tokenizers` library.

  pretokenize_reference.json  Qwen2 pre-tokenizer splits (no model needed)
  qwen2_encode_reference.json token ids from a Qwen2 tokenizer rebuilt from the
                              GGUF vocab + merges (NFC + Split + ByteLevel + BPE)

Usage: python make_reference.py <out_dir> [model.gguf]
"""
import json
import os
import sys

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

QWEN2_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
                 r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

CORPUS = [
    "Hello world",
    "I'm sure they'LL say it's fine, don't you THINK?",
    "Çok güzel bir gün, değil mi? İstanbul'da yağmur var.",
    "C++ dilinde RAII prensibini ve neden onemli oldugunu kisaca acikla.",
    "12345 + 3.14159 = 12348.14159; 1,000,000",
    "for (int i = 0; i < n; ++i) {\n\treturn x * 2;\n}",
    "  leading spaces",
    "trailing spaces   ",
    "a\n\n\nb",
    "line1\r\nline2\r\n",
    "tabs\t\tand   multiple    spaces",
    "emoji 🙂👍🏽 test",
    "你好，世界！こんにちは 세계",
    "مرحبا بالعالم",
    "!!!???...---***",
    "e-mail: user@example.com, url: https://example.com/a?b=c",
    "x²+y³ ½ ⅓ Ⅻ",
    "'s 'sun ' alone",
]
SPECIAL_CORPUS = [
    "hello<|im_end|>world",
    "<|im_start|>system\nYou are FRIDAY, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n"
    "C++ dilinde RAII prensibini ve neden onemli oldugunu kisaca acikla.<|im_end|>\n<|im_start|>assistant\n",
]

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

pre = pre_tokenizers.Split(Regex(QWEN2_PATTERN), behavior="isolated")
with open(os.path.join(out, "pretokenize_reference.json"), "w", encoding="utf-8") as f:
    json.dump([{"text": t, "pieces": [p for p, _ in pre.pre_tokenize_str(t)]} for t in CORPUS],
              f, ensure_ascii=False, indent=1)
print("wrote pretokenize_reference.json (%d cases)" % len(CORPUS))

if len(sys.argv) > 2:
    import gguf
    reader = gguf.GGUFReader(sys.argv[2], "r")

    def strings(name):
        fld = reader.fields[name]
        return [bytes(fld.parts[i]).decode("utf-8", errors="replace") for i in fld.data]

    tokens = strings("tokenizer.ggml.tokens")
    merges = [tuple(m.split(" ", 1)) for m in strings("tokenizer.ggml.merges")]
    tok = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=merges, byte_fallback=False))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tf = reader.fields["tokenizer.ggml.token_type"]
    types = [int(tf.parts[i][0]) for i in tf.data]
    tok.add_special_tokens([AddedToken(t, special=True) for t, ty in zip(tokens, types) if ty in (3, 4)])
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in CORPUS + SPECIAL_CORPUS]
    with open(os.path.join(out, "qwen2_encode_reference.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(tokens), "cases": cases}, f, ensure_ascii=False, indent=1)
    print("wrote qwen2_encode_reference.json (%d cases)" % len(cases))
