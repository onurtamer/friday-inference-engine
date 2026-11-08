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
#include "engine.h"
#include "system_info.h"
#include "gguf_file.h"
#include "model_config.h"
#include "model_registry.h"
#include <cstdio>
#include <memory>
#include "tokenizer.h"
#include <cmath>
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


void createDefaultTokenizerConfigIfNeeded(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::ofstream out(path);
        out << "{ \"vocab_size\": 11, \"bos_token_id\": 1, \"eos_token_id\": 2 }" << std::endl;
        out.close();
    }
}

// Greedy selection: skip NaN/Inf, first maximum wins.
static int argmaxLogits(const std::vector<float>& logits) {
    int max_idx = -1;
    float max_val = -1e30f;
    int nan_count = 0;
    for (int i = 0; i < (int)logits.size(); ++i) {
        const float val = logits[i];
        if (std::isnan(val) || std::isinf(val)) { nan_count++; continue; }
        if (val > max_val) { max_val = val; max_idx = i; }
    }
    if (max_idx < 0) std::cerr << "\n[CRITICAL] All " << logits.size() << " logits are NaN/Inf! (" << nan_count << " NaNs)\n";
    return max_idx;
}

// `FridayEngine models <list|add|remove|verify> ...`: manage the local model registry.
static int runModelsCommand(int argc, char** argv) {
    friday::ModelRegistry reg;
    const std::string cmd = argc >= 3 ? argv[2] : "list";
    if (cmd == "list") {
        std::cout << "Registry: " << reg.file() << "\n";
        if (reg.entries().empty()) { std::cout << "  (no models; add one with: models add <file.gguf> [id])\n"; return 0; }
        for (const auto& e : reg.entries()) {
            char line[512];
            std::snprintf(line, sizeof(line), "  %-28s %-8s %-7s %6.2f GB  %5.1fB params  %-12s %s\n", e.id.c_str(),
                          e.architecture.c_str(), e.quant.c_str(), e.size / 1e9, e.parameters / 1e9,
                          e.license.empty() ? "(no license)" : e.license.c_str(), e.path.c_str());
            std::cout << line << "      sha256 " << e.sha256 << "\n";
        }
        return 0;
    }
    if (cmd == "add" && argc >= 4) {
        const auto& e = reg.add(argv[3], argc >= 5 ? argv[4] : "");
        std::cout << "Added '" << e.id << "' (" << e.architecture << ", " << e.quant << ", sha256 " << e.sha256 << ")\n";
        if (e.license.empty()) std::cout << "  Note: the file has no general.license; check the model card.\n";
        return 0;
    }
    if (cmd == "remove" && argc >= 4) {
        if (!reg.remove(argv[3])) { std::cerr << "unknown model id: " << argv[3] << "\n"; return 1; }
        std::cout << "Removed '" << argv[3] << "' (the file itself was not touched)\n";
        return 0;
    }
    if (cmd == "verify" && argc >= 4) {
        std::string actual;
        const bool ok = reg.verify(argv[3], &actual);
        std::cout << (ok ? "OK: hash matches\n" : "MISMATCH: file changed (sha256 " + actual + ")\n");
        return ok ? 0 : 1;
    }
    std::cerr << "usage: FridayEngine models [list | add <file.gguf> [id] | remove <id> | verify <id>]\n";
    return 2;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    if (argc >= 2 && (std::string(argv[1]) == "--info" || std::string(argv[1]) == "info")) {
        const friday::CpuInfo c = friday::cpuInfo();
        const friday::MemoryInfo m = friday::memoryInfo();
        std::cout << "FRIDAY Engine capability report\n";
        std::cout << "  CPU      : " << (c.brand.empty() ? "?" : c.brand) << " (" << c.arch << ", " << c.logical_cpus
                  << " logical CPUs)\n  features :";
        for (const auto& f : c.features) std::cout << " " << f;
        std::cout << "\n  CPU GEMV : " << c.cpu_kernels << " kernels\n";
        std::printf("  RAM      : %.1f GB total, %.1f GB available\n", m.total_bytes / 1e9, m.available_bytes / 1e9);
        std::cout << "  CUDA     : " << (friday::Engine::cudaCompiledIn() ? "compiled in" : "not in this build") << "\n";
        for (const auto& g : friday::Engine::gpuInfo()) {
            std::printf("  GPU      : %s, compute %d.%d, %d SMs, %.1f GB VRAM (%.1f GB free)", g.name.c_str(),
                        g.compute_major, g.compute_minor, g.multiprocessors, g.total_bytes / 1e9, g.free_bytes / 1e9);
            if (g.pcie_gen_max) std::printf(", PCIe gen %d x%d (max link)", g.pcie_gen_max, g.pcie_width_max);
            std::printf("\n");
        }
        std::cout << "  Backend  : " << (friday::Engine::gpuInfo().empty() ? "CPU (all layers)" : "CUDA + CPU for layers that do not fit in VRAM")
                  << " (override: --backend cpu|cuda)\n";
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "models") {
        try { return runModelsCommand(argc, argv); }
        catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }
    }
    try {
        std::cout << "===================================================\n";
        std::cout << "  FRIDAY Engine 2.0 - V32 (Elastic Tiered Memory)\n";
        std::cout << "===================================================\n";
        std::cout << std::flush;

        // Model yolu ve VRAM limiti kontrolü:
        std::string gguf_model_path = "friday_aligned_int4.bin";
        double vram_limit_gb = 0.0; // 0.0 = Otomatik algilama
        int max_context = 2048;      // --ctx N: KV cache capacity in tokens (GPU memory)
        int cpu_threads = 0;         // --threads N: CPU threads for CPU-computed layers (0 = auto)
        friday::BackendKind backend = friday::BackendKind::Auto;  // --backend auto|cuda|cpu
        bool kv_q8 = false;                                        // --kv f16|q8

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--vram" && i + 1 < argc) {
                vram_limit_gb = std::stod(argv[++i]);
            } else if (arg == "--ctx" && i + 1 < argc) {
                max_context = std::stoi(argv[++i]);
            } else if (arg == "--threads" && i + 1 < argc) {
                cpu_threads = std::stoi(argv[++i]);
            } else if (arg == "--kv" && i + 1 < argc) {
                const std::string k = argv[++i];
                if (k == "q8") kv_q8 = true;
                else if (k == "f16") kv_q8 = false;
                else { std::cerr << "unknown --kv '" << k << "' (f16, q8)\n"; return 2; }
            } else if (arg == "--backend" && i + 1 < argc) {
                const std::string b = argv[++i];
                if (b == "cpu") backend = friday::BackendKind::Cpu;
                else if (b == "cuda") backend = friday::BackendKind::Cuda;
                else if (b == "auto") backend = friday::BackendKind::Auto;
                else { std::cerr << "unknown --backend '" << b << "' (auto, cuda, cpu)\n"; return 2; }
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

        // A registered model id can be used instead of a path.
        if (!std::ifstream(gguf_model_path, std::ios::binary).is_open()) {
            friday::ModelRegistry reg;
            if (const friday::ModelEntry* e = reg.find(gguf_model_path)) {
                std::cout << "[Init] Registry: '" << e->id << "' -> " << e->path << "\n";
                gguf_model_path = e->path;
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

        const bool is_gguf = gguf_model_path.size() > 5 &&
                             gguf_model_path.compare(gguf_model_path.size() - 5, 5, ".gguf") == 0;
        int hidden_dim = 0;
        int num_heads = 0;
        int num_kv_heads = 0;
        int num_layers = 0;
        float rms_norm_epsilon = 0.0f;
        std::string model_quant_type;
        std::unique_ptr<Tokenizer> tokenizer_ptr;

        if (is_gguf) {
            // Everything comes from the GGUF file: hyper-parameters, tokenizer, tensors.
            const friday::GgufFile gguf(gguf_model_path);
            const friday::ModelConfig cfg = friday::ModelConfig::fromGguf(gguf);
            cfg.checkSupported();
            hidden_dim = cfg.hidden_dim;
            num_heads = cfg.num_heads;
            num_kv_heads = cfg.num_kv_heads;
            num_layers = cfg.num_layers;
            rms_norm_epsilon = cfg.rms_norm_epsilon;
            std::cout << "[Init] " << cfg.architecture << " '" << cfg.name << "': " << cfg.num_layers << " layers, hidden "
                      << cfg.hidden_dim << ", heads " << cfg.num_heads << "/" << cfg.num_kv_heads << ", "
                      << (cfg.file_type.empty() ? "?" : cfg.file_type) << "\n";
            tokenizer_ptr.reset(new Tokenizer(gguf));
        } else {
            std::cerr << "[Init] Deprecated: converted .bin + model_params.json + tensor_map.json + tokenizer_config.json. "
                         "Pass the .gguf file instead; this path will be removed in a later release.\n";
            std::string tokenizer_config_path = "tokenizer_config.json";
            createDefaultTokenizerConfigIfNeeded(tokenizer_config_path);
            tokenizer_ptr.reset(new Tokenizer(tokenizer_config_path));
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
            model_quant_type = j.at("quant_type").get<std::string>();
            params_file.close();
        }
        Tokenizer& tokenizer = *tokenizer_ptr;
        friday::EngineOptions eo;
        eo.model_path = gguf_model_path;
        eo.vram_gb = vram_limit_gb;
        eo.max_ctx = max_context;
        eo.cpu_threads = cpu_threads;
        eo.backend = backend;
        eo.kv_q8 = kv_q8;
        eo.tensor_map = "tensor_map.json";  // deprecated .bin path only
        eo.hidden_dim = hidden_dim;
        eo.num_heads = num_heads;
        eo.num_kv_heads = num_kv_heads;
        eo.num_layers = num_layers;
        eo.rms_norm_epsilon = rms_norm_epsilon;
        eo.quant_type = model_quant_type;
        friday::Engine engine(eo);

        const int vocab_size = engine.vocabSize();
        int eos_token_id = tokenizer.getEosTokenId();
        if (eos_token_id == -1) eos_token_id = 151645; // Qwen <|im_end|>
        printf("[Init] vocab_size: %d, eos_token_id: %d\n", vocab_size, eos_token_id);
        std::vector<float> logits(vocab_size);

        std::cout << "[Init] FRIDAY Engine ready.\n";
        std::cout << "Sisteme soru sorabilirsiniz (Cikmak icin 'exit' veya 'quit' yazin):\n";
        std::cout << std::flush;

        std::string prompt;
        while (true) {
            std::cout << "\nUser> ";
            if (!std::getline(std::cin, prompt)) break;
            if (prompt == "exit" || prompt == "quit") break;
            if (prompt.empty()) continue;

            // Each prompt starts a new sequence (multi-turn chat comes with roadmap R6).
            std::vector<int> input_tokens = tokenizer.encode(tokenizer.applyChatTemplate(prompt));
            if (tokenizer.addsBos()) input_tokens.insert(input_tokens.begin(), tokenizer.getBosTokenId());
            std::cout << "[Tokenizer] " << input_tokens.size() << " prompt tokens.\n";
            if (input_tokens.empty()) continue;
            if ((int)input_tokens.size() >= engine.maxContext()) {
                std::cerr << "[Hata] Prompt is longer than the context (" << engine.maxContext() << "); use --ctx.\n";
                continue;
            }

            try {
                for (size_t i = 0; i < input_tokens.size(); ++i) engine.step(input_tokens[i], (int)i, logits.data());
                int next_token_id = argmaxLogits(logits);

                const int max_new_tokens = 100;
                std::cout << "\nFRIDAY: " << std::flush;
                for (int step = 0; step < max_new_tokens && next_token_id >= 0; ++step) {
                    if (next_token_id == eos_token_id || next_token_id == 151643 || next_token_id == 151645) break;
                    std::cout << cleanTokenString(tokenizer.decode({next_token_id})) << std::flush;
                    input_tokens.push_back(next_token_id);
                    const int current_pos = (int)input_tokens.size() - 1;
                    if (current_pos >= engine.maxContext()) { std::cerr << "\n[context full]"; break; }
                    engine.step(next_token_id, current_pos, logits.data());
                    next_token_id = argmaxLogits(logits);
                }
            } catch (const std::exception& e) {
                std::cerr << "\n[Hata] " << e.what() << "\n";
            }
            std::cout << "\n" << std::flush;
        }

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