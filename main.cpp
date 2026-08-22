#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "tokenizer.h"
#include "transformer.h"
#include "cuda_math_kernels.h"

// Helper function to create a dummy tokenizer config file if it doesn't exist
// CPU Argmax Stub for Token Selection
int argmax_stub(const std::vector<half>& logits, int vocab_size, int step) {
    float max_val = -1e9f;
    int max_idx = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        float val = __half2float(logits[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = static_cast<int>(i);
        }
    }
    // Modulo by vocab size and add step to generate pseudo-random sequence for PoC
    return (max_idx * 313 + step * 17 + 10) % vocab_size;
}

void createDefaultTokenizerConfigIfNeeded(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cout << "[Setup] Creating default '" << filename << "' for demonstration...\n";
        std::ofstream out(filename);
        out << "{\n";
        out << "  \"unk_token_id\": 0,\n";
        out << "  \"id_to_token\": {\n";
        out << "    \"0\": \"<unk>\",\n";
        out << "    \"1\": \"<s>\",\n";
        out << "    \"2\": \"</s>\",\n";
        out << "    \"3\": \"the\",\n";
        out << "    \"4\": \"a\",\n";
        out << "    \"5\": \"this\",\n";
        out << "    \"6\": \" is\",\n";
        out << "    \"7\": \" test\",\n";
        out << "    \"8\": \"hello\",\n";
        out << "    \"9\": \"fridays\"\n";
        out << "  },\n";
        out << "  \"token_to_id\": {\n";
        out << "    \"<unk>\": 0,\n";
        out << "    \"<s>\": 1,\n";
        out << "    \"</s>\": 2,\n";
        out << "    \"the\": 3,\n";
        out << "    \"a\": 4,\n";
        out << "    \"this\": 5,\n";
        out << "    \" is\": 6,\n";
        out << "    \" test\": 7,\n";
        out << "    \"hello\": 8,\n";
        out << "    \"fridays\": 9\n";
        out << "  }\n";
        out << "}\n";
        out.close();
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  FRIDAY Inference Engine Orchestrator  \n";
    std::cout << "========================================\n";

    // 1. Setup Tokenizer config and initialize
    std::string config_path = "tokenizer_config.json";
    createDefaultTokenizerConfigIfNeeded(config_path);

    Tokenizer tokenizer(config_path);

    // 2. Initialize Async I/O Manager
    AsyncIOManager async_io_manager;

    // 3. Model Hyperparameters
    int hidden_dim = 512;
    int num_heads = 8;
    int num_experts = 4;
    int top_k_experts = 2;
    float rms_norm_epsilon = 1e-5f;

    std::cout << "[Init] Allocating GPU device memory for weights and buffers...\n";

    // Allocate GPU memory for model weights (FP16)
    half *d_Wq = nullptr, *d_Wk = nullptr, *d_Wv = nullptr, *d_router_weights = nullptr;
    size_t qkv_weight_size = hidden_dim * hidden_dim * sizeof(half);
    size_t router_weight_size = hidden_dim * num_experts * sizeof(half);

    CUDA_CHECK(cudaMalloc((void**)&d_Wq, qkv_weight_size));
    CUDA_CHECK(cudaMalloc((void**)&d_Wk, qkv_weight_size));
    CUDA_CHECK(cudaMalloc((void**)&d_Wv, qkv_weight_size));
    CUDA_CHECK(cudaMalloc((void**)&d_router_weights, router_weight_size));

    // Allocate activation buffers on GPU
    half *d_input_activations = nullptr;
    half *d_query_out = nullptr, *d_key_out = nullptr, *d_value_out = nullptr;
    half *d_attention_output = nullptr;
    half *d_expert_input = nullptr, *d_expert_output = nullptr;
    unsigned char *d_expert_weights_buffer = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_input_activations, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_query_out, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_key_out, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_value_out, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_attention_output, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_expert_input, (hidden_dim / 2) * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&d_expert_output, (hidden_dim / 2) * sizeof(half)));
    
    // Allocate space for packed ternary expert weights (e.g., 2048x1024 / 4 bytes per row roughly)
    size_t expert_packed_buffer_size = 2048 * 1024 / 4; 
    CUDA_CHECK(cudaMalloc((void**)&d_expert_weights_buffer, expert_packed_buffer_size));

    // 4. Initialize Transformer Engine
    Transformer transformer(
        hidden_dim,
        num_heads,
        num_experts,
        top_k_experts,
        rms_norm_epsilon,
        d_Wq, d_Wk, d_Wv,
        d_router_weights,
        &async_io_manager
    );

    // 5. Create Mock Expert Map (simulating expert_map.json output from Python script)
    // expert_id -> (offset, size, original_shape)
    std::map<int, std::tuple<size_t, size_t, std::vector<int>>> expert_file_map;
    for (int i = 0; i < num_experts; ++i) {
        expert_file_map[i] = {static_cast<size_t>(i) * 524288ULL, 524288ULL, {2048, 1024}};
    }

    std::cout << "[Init] FRIDAY Engine successfully initialized and ready.\n";
    std::cout << "Type your prompt below (type 'exit' or 'quit' to stop):\n";

    // 6. Interactive Auto-Regressive Generation Loop
    std::string prompt;
    while (true) {
        std::cout << "\nFRIDAY> ";
        if (!std::getline(std::cin, prompt)) break;
        if (prompt == "exit" || prompt == "quit") {
            std::cout << "[Shutdown] Exiting interactive loop...\n";
            break;
        }
        if (prompt.empty()) continue;

        std::vector<int> current_tokens = tokenizer.encode(prompt);
        std::cout << "[Tokenizer] Prompt encoded. Generating response...\n";
        std::cout << "FRIDAY: " << std::flush;

        const int max_new_tokens = 50;
        const int vocab_size = 32000;

        for (int step = 0; step < max_new_tokens; ++step) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            // Prepare dummy activation data
            std::vector<half> h_activations(hidden_dim, __float2half(0.15f + (step * 0.01f)));
            CUDA_CHECK(cudaMemcpyAsync(d_input_activations, h_activations.data(), hidden_dim * sizeof(half), cudaMemcpyHostToDevice, stream));

            // 1. RMSNorm
            transformer.applyRMSNorm(d_input_activations, d_Wq, d_input_activations, hidden_dim, stream);

            // 2. Self-Attention
            transformer.applySelfAttention(
                d_input_activations,
                d_query_out, d_key_out, d_value_out,
                d_attention_output,
                current_tokens.size(),
                step, // position offset
                stream
            );

            // 3. MoE Router & I/O
            transformer.routeMixtureOfExperts(
                d_attention_output,
                d_expert_input,
                d_expert_output,
                d_expert_weights_buffer,
                expert_file_map,
                stream
            );

            // Fetch logits (mocked from attention_output) for argmax
            std::vector<half> h_attention_output(hidden_dim);
            CUDA_CHECK(cudaMemcpyAsync(h_attention_output.data(), d_attention_output, hidden_dim * sizeof(half), cudaMemcpyDeviceToHost, stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamDestroy(stream));

            // Token Selection (Argmax)
            int next_token = argmax_stub(h_attention_output, vocab_size, step);
            
            // Auto-regressive append
            current_tokens.push_back(next_token);

            // Decode and stream output
            std::string text = tokenizer.decode({next_token});
            std::cout << text << std::flush;

            // Simple EOS check
            if (next_token == 2) break; // </s> for llama
        }
        std::cout << std::endl;
    }

    // 7. Safe Resource Cleanup (RAII / Manual Free to prevent memory leaks)
    std::cout << "[Cleanup] Freeing GPU and Host resources...\n";
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaFree(d_router_weights));
    CUDA_CHECK(cudaFree(d_input_activations));
    CUDA_CHECK(cudaFree(d_query_out));
    CUDA_CHECK(cudaFree(d_key_out));
    CUDA_CHECK(cudaFree(d_value_out));
    CUDA_CHECK(cudaFree(d_attention_output));
    CUDA_CHECK(cudaFree(d_expert_input));
    CUDA_CHECK(cudaFree(d_expert_output));
    CUDA_CHECK(cudaFree(d_expert_weights_buffer));

    std::cout << "[Shutdown] FRIDAY Engine successfully shut down.\n";
    return 0;
}
