import os
import json
import numpy as np
import ml_dtypes
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# --- Configuration ---
MODEL_REPO = "1bitLLM/bitnet_b1_58-large"
MODEL_FILENAME = "model.safetensors"
TOKENIZER_FILENAME = "tokenizer.json"

PAGE_SIZE = 4096  # 4KB Page alignment for O_DIRECT / FILE_FLAG_NO_BUFFERING compatibility
OUTPUT_BIN_FILE = "friday_model.bin"
OUTPUT_MAP_FILE = "expert_map.json"
OUTPUT_TOKENIZER_CONFIG = "tokenizer_config.json"
QUANTIZATION_THRESHOLD = 0.05

def download_model_files():
    print(f"[HuggingFace] Downloading {MODEL_FILENAME} and {TOKENIZER_FILENAME} from {MODEL_REPO}...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, force_download=True)
    tokenizer_path = hf_hub_download(repo_id=MODEL_REPO, filename=TOKENIZER_FILENAME, force_download=True)
    print(f"[HuggingFace] Download complete.\n  Model: {model_path}\n  Tokenizer: {tokenizer_path}")
    return model_path, tokenizer_path

def convert_tokenizer_config(tokenizer_json_path, output_config_path):
    print(f"[Tokenizer] Converting {tokenizer_json_path} to {output_config_path}...")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vocab = {}
    if "model" in data and "vocab" in data["model"]:
        vocab = data["model"]["vocab"]
    elif "vocab" in data:
        vocab = data["vocab"]
    else:
        # Fallback or general structure inspection
        raise ValueError("Unsupported tokenizer.json structure: 'vocab' key not found.")

    id_to_token = {}
    token_to_id = {}
    unk_token_id = 0

    # Handle different vocab formats (dict token->id or list/etc)
    if isinstance(vocab, dict):
        for token, idx in vocab.items():
            id_to_token[str(idx)] = token
            token_to_id[token] = int(idx)
    elif isinstance(vocab, list):
        for idx, token in enumerate(vocab):
            id_to_token[str(idx)] = token
            token_to_id[token] = idx

    # Check for unk token
    if "added_tokens" in data:
        for added in data["added_tokens"]:
            if added.get("content") == "<unk>":
                unk_token_id = added.get("id", 0)
                break

    config_data = {
        "unk_token_id": unk_token_id,
        "id_to_token": id_to_token,
        "token_to_id": token_to_id
    }

    with open(output_config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    print(f"[Tokenizer] Successfully generated {output_config_path} with vocab size {len(token_to_id)}.")

def quantize_ternary(tensor: np.ndarray, threshold: float = QUANTIZATION_THRESHOLD) -> np.ndarray:
    quantized_tensor = np.zeros_like(tensor, dtype=np.int8)
    quantized_tensor[tensor > threshold] = 1
    quantized_tensor[tensor < -threshold] = -1
    return quantized_tensor

def pack_ternary_to_bytes(ternary_array: np.ndarray) -> bytes:
    mapped_array = np.zeros_like(ternary_array, dtype=np.uint8)
    mapped_array[ternary_array == -1] = 0  # 00
    mapped_array[ternary_array == 0] = 1   # 01
    mapped_array[ternary_array == 1] = 2   # 10

    packed_bytes = bytearray()
    num_elements = len(mapped_array)

    for i in range(0, num_elements, 4):
        byte_val = 0
        for j in range(4):
            if i + j < num_elements:
                byte_val |= (mapped_array[i + j] << ((3 - j) * 2))
        packed_bytes.append(byte_val)
    return bytes(packed_bytes)

def process_and_serialize_weights(model_path):
    print(f"[Quantization] Opening model safetensors: {model_path}")
    expert_map = {}
    current_offset = 0

    bin_file_path = Path(OUTPUT_BIN_FILE)
    map_file_path = Path(OUTPUT_MAP_FILE)

    with open(bin_file_path, "wb") as f_bin:
        with safe_open(model_path, framework="np", device="cpu") as f:
            keys = f.keys()
            # Filter for MLP / linear projection weights to treat as virtual experts
            # TinyLlama layers: model.layers.{i}.mlp.gate_proj.weight, up_proj.weight, down_proj.weight
            virtual_expert_keys = [k for k in keys if "mlp" in k and "weight" in k]
            
            if not virtual_expert_keys:
                # Fallback: take any weight tensors if mlp is structured differently
                virtual_expert_keys = [k for k in keys if "weight" in k][:16]

            print(f"[Quantization] Found {len(virtual_expert_keys)} weight tensors to serialize as virtual experts.")

            for expert_id, key in enumerate(virtual_expert_keys):
                print(f"  Processing [{expert_id}] {key}...")
                tensor = f.get_tensor(key)
                tensor = tensor.astype(np.float32)
                
                # Quantize & pack
                quantized_data = quantize_ternary(tensor.flatten())
                packed_bytes = pack_ternary_to_bytes(quantized_data)
                
                data_byte_size = len(packed_bytes)
                padding_needed = (PAGE_SIZE - (data_byte_size % PAGE_SIZE)) % PAGE_SIZE
                padded_block_size = data_byte_size + padding_needed

                # Write page-aligned bytes
                f_bin.write(packed_bytes)
                if padding_needed > 0:
                    f_bin.write(b'\0' * padding_needed)

                expert_map[str(expert_id)] = {
                    "tensor_name": key,
                    "offset": current_offset,
                    "size": padded_block_size,
                    "original_shape": list(tensor.shape),
                    "quantized_element_count": len(quantized_data)
                }

                current_offset += padded_block_size

    print(f"[Quantization] Weights successfully written to {OUTPUT_BIN_FILE} ({bin_file_path.stat().st_size} bytes).")

    with open(map_file_path, "w", encoding='utf-8') as f_json:
        json.dump(expert_map, f_json, indent=4)
    print(f"[Quantization] Expert map successfully written to {OUTPUT_MAP_FILE}.")

def main():
    try:
        print("=== FRIDAY Real Model Converter (Path A) ===")
        model_path, tokenizer_path = download_model_files()
        convert_tokenizer_config(tokenizer_path, OUTPUT_TOKENIZER_CONFIG)
        process_and_serialize_weights(model_path)
        print("=== Conversion Completed Successfully ===")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during conversion: {e}")
        exit(1)

if __name__ == "__main__":
    main()
