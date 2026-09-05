#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#define INT8 WIN_INT8
#include <windows.h>
#undef INT8
#endif

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <exception> // Hatalari yakalamak icin eklendi
#include "transformer.h"
#include "io_manager.h"
#include "tokenizer.h"
#include "kernel_dispatcher.h"
#include <nlohmann/json.hpp>

// V32: Modelden üretilen token string'e ekrana basılmadan hemen önce uygulanan temizleme filtresi
static std::string cleanTokenString(std::string text) {
    // 0. <0x0A> -> "\n"
    const std::string hex_nl = "<0x0A>";
    size_t pos = 0;
    while ((pos = text.find(hex_nl, pos)) != std::string::npos) {
        text.replace(pos, hex_nl.length(), "\n");
        pos += 1;
    }

    // 1. Qwen BPE newline: "\xc4\x8a" (Ċ) -> "\n"
    const std::string bpe_nl = "\xc4\x8a";
    pos = 0;
    while ((pos = text.find(bpe_nl, pos)) != std::string::npos) {
        text.replace(pos, bpe_nl.length(), "\n");
        pos += 1;
    }

    // 2. Qwen BPE tab: "\xc4\x89" (ĉ) -> "\t"
    const std::string bpe_tab = "\xc4\x89";
    pos = 0;
    while ((pos = text.find(bpe_tab, pos)) != std::string::npos) {
        text.replace(pos, bpe_tab.length(), "\t");
        pos += 1;
    }

    // 3. Qwen BPE boşluk byte'ları: "\xe2\x96\x81" (U+2581) -> " "
    const std::string bpe_space = "\xe2\x96\x81";
    pos = 0;
    while ((pos = text.find(bpe_space, pos)) != std::string::npos) {
        text.replace(pos, bpe_space.length(), " ");
        pos += 1;
    }

    // 4. GPT-2 style BPE boşluğu: "Ġ" (UTF-8: \xc4\xa0) -> " "
    const std::string g_space_utf8 = "\xc4\xa0";
    pos = 0;
    while ((pos = text.find(g_space_utf8, pos)) != std::string::npos) {
        text.replace(pos, g_space_utf8.length(), " ");
        pos += 1;
    }

    const std::string g_space = "Ġ";
    pos = 0;
    while ((pos = text.find(g_space, pos)) != std::string::npos) {
        text.replace(pos, g_space.length(), " ");
        pos += 1;
    }

    return text;
}

QuantType stringToQuantType(const std::string& type_str) {
    if (type_str == "F16" || type_str == "F32") return FP16;
    if (type_str == "Q4_0" || type_str == "Q4_K") return INT4;
    if (type_str == "Q4_1") return Q4_1;
    if (type_str == "Q8_0" || type_str == "Q8_1") return INT8;
    if (type_str == "Q6_K") return Q6_K;
    if (type_str == "TERNARY") return TERNARY;
    return UNKNOWN;
}

void createDefaultTokenizerConfigIfNeeded(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::ofstream out(path);
        out << "{ \"vocab_size\": 11, \"bos_token_id\": 1, \"eos_token_id\": 2 }" << std::endl;
        out.close();
    }
}

int argmaxFromDeviceLogits(const half* d_logits, int vocab_size, cudaStream_t stream) {
    std::vector<half> h_logits(vocab_size);
    cudaMemcpyAsync(h_logits.data(), d_logits, vocab_size * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    int max_idx = -1;
    float max_val = -1e30f;
    int nan_count = 0;
    for (int i = 0; i < vocab_size; ++i) {
        float val = __half2float(h_logits[i]);
        if (std::isnan(val) || std::isinf(val)) {
            nan_count++;
            continue;
        }
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    if (max_idx < 0) {
        std::cerr << "\n[CRITICAL] All " << vocab_size << " logits are NaN/Inf! (" << nan_count << " NaNs)\n";
        return 151643;
    }
    return max_idx;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    try {
        std::cout << "===================================================\n";
        std::cout << "  FRIDAY Engine 2.0 - V32 (Elastic Tiered Memory)\n";
        std::cout << "===================================================\n";
        std::cout << std::flush;

        // Model yolu ve VRAM limiti kontrolü:
        std::string gguf_model_path = "friday_aligned_int4.bin";
        double vram_limit_gb = 0.0; // 0.0 = Otomatik algilama

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--vram" && i + 1 < argc) {
                vram_limit_gb = std::stod(argv[++i]);
            } else if (arg.rfind("--", 0) != 0) {
                // Sayısal değer ise vram limiti, değilse model dosyasıdır
                try {
                    size_t idx = 0;
                    double val = std::stod(arg, &idx);
                    if (idx == arg.length()) {
                        vram_limit_gb = val;
                        continue;
                    }
                } catch (...) {}
                gguf_model_path = arg;
            }
        }

        std::ifstream test_f(gguf_model_path, std::ios::binary);
        if (!test_f.is_open()) {
            std::string fallback = "qwen2.5-0.5b-instruct-q4_0.gguf";
            std::ifstream test_f2(fallback, std::ios::binary);
            if (test_f2.is_open()) {
                gguf_model_path = fallback;
            }
        }
        std::cout << "[Init] Secilen model yolu: " << gguf_model_path << "\n" << std::flush;

        std::string tokenizer_config_path = "tokenizer_config.json";
        
        createDefaultTokenizerConfigIfNeeded(tokenizer_config_path); 

        Tokenizer tokenizer(tokenizer_config_path);
        AsyncIOManager async_io_manager(gguf_model_path);

        int hidden_dim = 0;
        int num_heads = 0;
        int num_kv_heads = 0;
        int num_layers = 0;
        float rms_norm_epsilon = 0.0f;
        QuantType model_quant_type = UNKNOWN;

        std::ifstream params_file("model_params.json");
        if (!params_file.is_open()) {
            fprintf(stderr, "Hata: 'model_params.json' dosyasi bulunamadi.\n");
            return 1;
        }
        nlohmann::json j = nlohmann::json::parse(params_file);
        hidden_dim = j.at("hidden_dim").get<int>();
        num_heads = j.at("num_heads").get<int>();
        num_kv_heads = j.at("num_kv_heads").get<int>();
        num_layers = j.at("num_layers").get<int>();
        rms_norm_epsilon = j.at("rms_norm_epsilon").get<float>();
        model_quant_type = stringToQuantType(j.at("quant_type").get<std::string>());
        params_file.close();

        std::string tensor_map_file = "tensor_map.json";

        std::cout << "[Init] Allocating GPU device memory for activation buffers...\n" << std::flush;

        Transformer transformer(
            gguf_model_path,
            tensor_map_file,
            hidden_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            rms_norm_epsilon,
            model_quant_type,
            &async_io_manager,
            vram_limit_gb
        );

        int vocab_size = transformer.getVocabSize();
        if (vocab_size == 0) {
            vocab_size = tokenizer.getVocabSize();
        }

        if (vocab_size == 0) {
            fprintf(stderr, "Hata: vocab_size belirlenemedi.\n");
            return 1;
        }
        
        int eos_token_id = tokenizer.getEosTokenId();
        if (eos_token_id == -1) eos_token_id = 151645; // Qwen <|im_end|>
        
        printf("[Init] Determined vocab_size: %d, eos_token_id: %d\n", vocab_size, eos_token_id);
        fflush(stdout);

        half *d_output_logits = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_output_logits, vocab_size * sizeof(half)));
        CUDA_CHECK(cudaMemset(d_output_logits, 0, vocab_size * sizeof(half)));

        half* d_input_activations = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_input_activations, hidden_dim * sizeof(half)));
        CUDA_CHECK(cudaMemset(d_input_activations, 0, hidden_dim * sizeof(half)));

        std::cout << "[Init] FRIDAY Engine 2.0 (V32 - Elastic Tiered Memory & 14B) basariyla hazirlandi.\n";
        std::cout << "Sisteme soru sorabilirsiniz (Cikmak icin 'exit' veya 'quit' yazin):\n";
        std::cout << std::flush;

        std::string prompt;
        while (true) {
            std::cout << "\nUser> ";
            if (!std::getline(std::cin, prompt)) break;
            if (prompt == "exit" || prompt == "quit") break;
            if (prompt.empty()) continue;

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            // Apply Qwen ChatML template before encoding
            std::string templated_prompt = tokenizer.applyChatTemplate(prompt);
            std::vector<int> input_tokens = tokenizer.encode(templated_prompt);
            std::cout << "[Tokenizer] Prompt ChatML formatinda islendi (" << input_tokens.size() << " token).\n";

            if (input_tokens.empty()) {
                CUDA_CHECK(cudaStreamDestroy(stream));
                continue;
            }

            std::cout << "[Inference] Cevap uretiliyor...\n" << std::flush;

            bool prefill_success = true;
            // Prefill Phase: tum prompt tokenlarini isleyelim
            for (size_t i = 0; i < input_tokens.size(); ++i) {
                int current_token = input_tokens[i];
                
                if (!transformer.getEmbedding(current_token, d_input_activations, vocab_size, stream)) {
                    fprintf(stderr, "\n[Hata] Token %d icin embedding alinamadi. Durduruluyor...\n", current_token);
                    prefill_success = false;
                    break;
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));

                CUDA_CHECK(cudaMemsetAsync(d_output_logits, 0, vocab_size * sizeof(half), stream));
                transformer.forward(d_input_activations, d_output_logits, 1, 0, (int)i, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            if (!prefill_success) {
                CUDA_CHECK(cudaStreamDestroy(stream));
                continue;
            }

            int next_token_id = argmaxFromDeviceLogits(d_output_logits, vocab_size, stream);

            // Auto-regressive Generation Loop with Live Streaming
            const int max_new_tokens = 100;
            std::cout << "\nFRIDAY: " << std::flush;

            for (int step = 0; step < max_new_tokens; ++step) {
                if (next_token_id == eos_token_id || next_token_id == 151643 || next_token_id == 151645) {
                    break;
                }

                std::string token_text = cleanTokenString(tokenizer.decode({next_token_id}));
                std::cout << token_text << std::flush;
                input_tokens.push_back(next_token_id);

                if (!transformer.getEmbedding(next_token_id, d_input_activations, vocab_size, stream)) {
                    fprintf(stderr, "\n[Hata] Token %d icin embedding alinamadi. Durduruluyor...\n", next_token_id);
                    break;
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));

                int current_pos = (int)input_tokens.size() - 1;
                CUDA_CHECK(cudaMemsetAsync(d_output_logits, 0, vocab_size * sizeof(half), stream));
                transformer.forward(d_input_activations, d_output_logits, 1, 0, current_pos, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                next_token_id = argmaxFromDeviceLogits(d_output_logits, vocab_size, stream);
            }

            std::cout << "\n" << std::flush;
            CUDA_CHECK(cudaStreamDestroy(stream));
        }

        std::cout << "[Cleanup] Freeing GPU activation buffers...\n";
        CUDA_CHECK(cudaFree(d_input_activations));
        CUDA_CHECK(cudaFree(d_output_logits));

        std::cout << "[Shutdown] FRIDAY Engine 2.0 successfully shut down.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] C++ Exception Caught: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n[FATAL ERROR] Unknown Exception Caught!\n";
        return 1;
    }
}