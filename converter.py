import os
import sys
import json
import shutil
from pathlib import Path
from gguf import GGUFReader

PAGE_SIZE = 4096

def get_quant_type_id(dtype_str: str) -> int:
    dtype_upper = dtype_str.upper()
    if "Q4_1" in dtype_upper:
        return 4  # Q4_1
    elif "Q4" in dtype_upper or "INT4" in dtype_upper:
        return 2  # INT4 (Q4_0)
    elif "Q6_K" in dtype_upper or "Q6" in dtype_upper:
        return 5  # Q6_K
    elif "Q8" in dtype_upper or "INT8" in dtype_upper:
        return 1  # INT8
    elif "F16" in dtype_upper or "FP16" in dtype_upper or "F32" in dtype_upper:
        return 0  # FP16 / FP32
    elif "TERNARY" in dtype_upper:
        return 3  # TERNARY
    return 6  # UNKNOWN

def convert_gguf_to_aligned_bin(
    gguf_path: str = "Qwen2.5-14B-Instruct-Q4_0.gguf",
    output_bin_path: str = "friday_aligned_int4.bin",
    output_map_path: str = "tensor_map.json",
    output_params_path: str = "model_params.json",
    output_tokenizer_path: str = "tokenizer_config.json"
):
    print("============================================================")
    print("  FRIDAY ENGINE 2.0 - 14B GGUF -> Aligned BIN Converter     ")
    print("============================================================")
    print(f"Giris GGUF Dosyasi: {gguf_path}")
    print(f"Hedef BIN Dosyasi : {output_bin_path}")
    print(f"Hedef Harita (JSON): {output_map_path}")
    print(f"Sayfa Hizalamasi   : {PAGE_SIZE} bayt (4KB O_DIRECT)\n")

    if not os.path.exists(gguf_path):
        print(f"[Hata] GGUF dosyasi bulunamadi: {gguf_path}", file=sys.stderr)
        sys.exit(1)

    print("[1/4] GGUF Dosyasi taraniyor ve metadata okunuyor...")
    reader = GGUFReader(gguf_path, 'r')

    # 1. Model Parametrelerini Cikar
    model_params = {
        "vocab_size": 152064,
        "hidden_dim": 5120,
        "num_heads": 40,
        "num_kv_heads": 8,
        "num_layers": 48,
        "rms_norm_epsilon": 1e-6,
        "quant_type": "Q4_0",
        "max_position_embeddings": 32768,
        "rope_freq_base": 1000000.0
    }

    for key, field in reader.fields.items():
        try:
            val = field.parts[-1].tolist()
            value = val[0] if isinstance(val, list) and len(val) == 1 else val
        except Exception:
            continue

        if "block_count" in key:
            model_params["num_layers"] = int(value)
        elif "embedding_length" in key:
            model_params["hidden_dim"] = int(value)
        elif "attention.head_count_kv" in key:
            model_params["num_kv_heads"] = int(value)
        elif "attention.head_count" in key:
            model_params["num_heads"] = int(value)
        elif "context_length" in key:
            model_params["max_position_embeddings"] = int(value)
        elif "rope.freq_base" in key:
            model_params["rope_freq_base"] = float(value)
        elif "attention.layer_norm_rms_epsilon" in key:
            model_params["rms_norm_epsilon"] = float(value)

    print(f"  Tespit edilen parametreler:")
    print(f"    - Katman Sayisi   : {model_params['num_layers']}")
    print(f"    - Hidden Dim      : {model_params['hidden_dim']}")
    print(f"    - Heads (Q / KV)  : {model_params['num_heads']} / {model_params['num_kv_heads']}")
    print(f"    - RMSNorm Epsilon : {model_params['rms_norm_epsilon']}")
    print(f"    - RoPE Freq Base  : {model_params['rope_freq_base']}")

    # 2. Tokenizer Bilgisini Cikar
    print("\n[2/4] Tokenizer sozlugu kontrol ediliyor...")
    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    if tokens_field:
        try:
            tokens_list = tokens_field.parts
            id_to_token = {}
            token_to_id = {}
            for idx, part in enumerate(tokens_list):
                token_bytes = bytes(part)
                token_str = token_bytes.decode('utf-8', errors='replace')
                id_to_token[str(idx)] = token_str
                token_to_id[token_str] = idx

            model_params["vocab_size"] = len(tokens_list)
            tokenizer_data = {
                "vocab_size": len(tokens_list),
                "bos_token_id": 151644,
                "eos_token_id": 151645,
                "unk_token_id": 151643,
                "id_to_token": id_to_token,
                "token_to_id": token_to_id
            }
            with open(output_tokenizer_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_data, f, ensure_ascii=False)
            print(f"  Tokenizer sozlugu ({len(tokens_list)} token) '{output_tokenizer_path}' dosyasina yazildi.")
        except Exception as e:
            print(f"  [Uyari] Tokenizer cikarilirken hata: {e}, mevcut config korunacak.")

    with open(output_params_path, "w", encoding="utf-8") as f:
        json.dump(model_params, f, indent=2)
    print(f"  Model parametreleri '{output_params_path}' dosyasina kaydedildi.")

    # 3. Tensorleri Grupla ve Sirala
    print("\n[3/4] Tensorler katman sirasina gore duzenleniyor...")
    tensors = reader.tensors

    layer_tensors = {}
    static_tensors = []

    for t in tensors:
        name = t.name
        if name.startswith("blk."):
            parts = name.split(".")
            l_idx = int(parts[1])
            if l_idx not in layer_tensors:
                layer_tensors[l_idx] = []
            layer_tensors[l_idx].append(t)
        else:
            static_tensors.append(t)

    sorted_tensors = []
    for l_idx in sorted(layer_tensors.keys()):
        sorted_tensors.extend(layer_tensors[l_idx])
    sorted_tensors.extend(static_tensors)

    print(f"  Toplam {len(sorted_tensors)} tensor ({len(layer_tensors)} katman + {len(static_tensors)} statik tensor) siralandi.")

    # 4. 4096 Bayt Hizali BIN Dosyasini ve Tensor Haritasini Olustur
    print(f"\n[4/4] '{output_bin_path}' olusturuluyor (4096-bayt sayfa hizalamasi)...")
    tensor_map_data = []
    current_bin_offset = 0

    chunk_size = 64 * 1024 * 1024  # 64 MB okuma/yazma tamponu
    total_raw_bytes = sum(t.n_bytes for t in sorted_tensors)
    written_bytes = 0

    with open(gguf_path, "rb") as in_f, open(output_bin_path, "wb") as out_f:
        for idx, t in enumerate(sorted_tensors):
            padding = (PAGE_SIZE - (current_bin_offset % PAGE_SIZE)) % PAGE_SIZE
            if padding > 0:
                out_f.write(b'\x00' * padding)
                current_bin_offset += padding

            tensor_start_offset = current_bin_offset

            in_f.seek(t.data_offset)
            remaining = t.n_bytes
            while remaining > 0:
                n_read = min(remaining, chunk_size)
                buf = in_f.read(n_read)
                if not buf:
                    raise IOError(f"Beklenmedik dosya sonu: {t.name}")
                out_f.write(buf)
                remaining -= len(buf)

            current_bin_offset += t.n_bytes
            written_bytes += t.n_bytes

            dtype_str = t.tensor_type.name
            quant_id = get_quant_type_id(dtype_str)

            tensor_info = {
                "name": t.name,
                "absolute_offset": tensor_start_offset,
                "size_bytes": int(t.n_bytes),
                "shape": [int(x) for x in t.shape],
                "dtype": dtype_str,
                "quant_type": quant_id
            }
            tensor_map_data.append(tensor_info)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(sorted_tensors):
                pct = (written_bytes / total_raw_bytes) * 100
                print(f"  Islenen tensor: {idx + 1}/{len(sorted_tensors)} ({pct:.1f}%) - Guncel BIN boyutu: {current_bin_offset / (1024**3):.2f} GB", flush=True)

    with open(output_map_path, "w", encoding="utf-8") as f:
        json.dump(tensor_map_data, f, ensure_ascii=False, indent=2)

    final_bin_size = os.path.getsize(output_bin_path)
    print(f"\n[Basarili] BIN Donusturme Tamamlandi:")
    print(f"  - BIN Dosyasi : {output_bin_path} ({final_bin_size / (1024**3):.2f} GB)")
    print(f"  - Harita      : {output_map_path} ({len(tensor_map_data)} tensor)")
    print(f"  - Parametreler: {output_params_path}")

    # Build / Release klasorune kopyala
    release_dir = Path("build/Release")
    if release_dir.exists():
        print("\n[Senkronizasyon] Dosyalar 'build/Release' dizinine kopyalaniyor...")
        try:
            shutil.copy2(output_map_path, release_dir / output_map_path)
            shutil.copy2(output_params_path, release_dir / output_params_path)
            if os.path.exists(output_tokenizer_path):
                shutil.copy2(output_tokenizer_path, release_dir / output_tokenizer_path)
            print("  JSON dosyalari 'build/Release' dizinine basariyla aktarildi.")
        except Exception as e:
            print(f"  [Uyari] Release kopyalama hatasi: {e}")

    print("\n============================================================")
    print("  DONUSTURME BASARIYLA TAMAMLANDI!                          ")
    print("============================================================")

if __name__ == "__main__":
    gguf_file = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-14B-Instruct-Q4_0.gguf"
    convert_gguf_to_aligned_bin(gguf_file)
