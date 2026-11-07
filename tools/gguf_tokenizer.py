"""Build the engine's tokenizer_config.json from a GGUF file (read-only).

String arrays are read through field.data indices; field.parts also contains
length/type headers, so iterating parts directly (as converter.py used to)
shifts every token id.
"""
import json

GGUF_TOKEN_CONTROL = 3
GGUF_TOKEN_USER_DEFINED = 4


def _strings(reader, name):
    f = reader.fields.get(name)
    if f is None:
        return None
    return [bytes(f.parts[i]).decode("utf-8", errors="replace") for i in f.data]


def _ints(reader, name):
    f = reader.fields.get(name)
    if f is None:
        return None
    return [int(f.parts[i][0]) for i in f.data]


def _scalar(reader, name):
    f = reader.fields.get(name)
    return None if f is None else int(f.parts[f.data[0]][0])


def tokenizer_config_from_gguf(reader):
    tokens = _strings(reader, "tokenizer.ggml.tokens")
    if not tokens:
        raise ValueError("GGUF has no tokenizer.ggml.tokens")
    merges = _strings(reader, "tokenizer.ggml.merges") or []
    types = _ints(reader, "tokenizer.ggml.token_type") or []
    special = [t for t, ty in zip(tokens, types) if ty in (GGUF_TOKEN_CONTROL, GGUF_TOKEN_USER_DEFINED)]
    return {
        "vocab_size": len(tokens),
        "bos_token_id": _scalar(reader, "tokenizer.ggml.bos_token_id"),
        "eos_token_id": _scalar(reader, "tokenizer.ggml.eos_token_id"),
        "unk_token_id": _scalar(reader, "tokenizer.ggml.unknown_token_id"),
        "special_tokens": special,
        "merges": merges,
        "id_to_token": {str(i): t for i, t in enumerate(tokens)},
        "token_to_id": {t: i for i, t in enumerate(tokens)},
    }


def write_tokenizer_config(reader, out_path):
    data = tokenizer_config_from_gguf(reader)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data
