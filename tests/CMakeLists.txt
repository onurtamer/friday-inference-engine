# Test labels:
#   unit  - CPU only, < 10 s. Run on every commit.
#   gpu   - needs an NVIDIA GPU, a few seconds. Run when kernels or memory code change.
#   model - needs FRIDAY_MODEL_DIR (model, tensor map, tokenizer), ~1 min, ~3 GB free RAM.
#           Run before merging any change that may affect numerics.
# Examples:  ctest -L unit     ctest -L gpu     ctest -L model

function(friday_test_target name)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    add_executable(${name} ${ARGN})
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PRIVATE friday_engine)
endfunction()

# --- unit -------------------------------------------------------------------
friday_test_target(test_tokenizer_bpe unit/test_tokenizer_bpe.cpp)
add_test(NAME tokenizer_bpe
         COMMAND test_tokenizer_bpe ${CMAKE_CURRENT_BINARY_DIR}/tokenizer_bpe_fixture.json)
set_tests_properties(tokenizer_bpe PROPERTIES LABELS "unit")

friday_test_target(test_tokenizer_pretokenize unit/test_tokenizer_pretokenize.cpp)
add_test(NAME tokenizer_pretokenize
         COMMAND test_tokenizer_pretokenize ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/pretokenize_reference.json)
set_tests_properties(tokenizer_pretokenize PROPERTIES LABELS "unit")

# --- gpu --------------------------------------------------------------------
friday_test_target(test_decode_attention kernels/test_decode_attention.cu)
add_test(NAME decode_attention COMMAND test_decode_attention)
set_tests_properties(decode_attention PROPERTIES LABELS "gpu")

find_package(Python3 COMPONENTS Interpreter)
set(FRIDAY_HAVE_GGUF_PY FALSE)
if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import gguf, numpy"
                    RESULT_VARIABLE _gguf_rc OUTPUT_QUIET ERROR_QUIET)
    if(_gguf_rc EQUAL 0)
        set(FRIDAY_HAVE_GGUF_PY TRUE)
    endif()
endif()

if(FRIDAY_HAVE_GGUF_PY)
    set(_fx ${CMAKE_CURRENT_BINARY_DIR}/kquant_fixtures)
    add_test(NAME kquant_fixtures
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/make_fixtures.py ${_fx})
    set_tests_properties(kquant_fixtures PROPERTIES LABELS "gpu" FIXTURES_SETUP kquant)

    friday_test_target(test_kquant_kernels kernels/test_kquant_kernels.cu)
    add_test(NAME kquant_kernels COMMAND test_kquant_kernels ${_fx})
    set_tests_properties(kquant_kernels PROPERTIES LABELS "gpu" FIXTURES_REQUIRED kquant)

    set(_gx ${CMAKE_CURRENT_BINARY_DIR}/gguf_fixtures)
    add_test(NAME gguf_fixtures
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/unit/make_gguf_fixtures.py ${_gx})
    set_tests_properties(gguf_fixtures PROPERTIES LABELS "unit" FIXTURES_SETUP gguf)
    friday_test_target(test_gguf_file unit/test_gguf_file.cpp)
    add_test(NAME gguf_file COMMAND test_gguf_file ${_gx})
    set_tests_properties(gguf_file PROPERTIES LABELS "unit" FIXTURES_REQUIRED gguf)
    friday_test_target(test_model_registry unit/test_model_registry.cpp)
    add_test(NAME model_registry COMMAND test_model_registry ${_gx})
    set_tests_properties(model_registry PROPERTIES LABELS "unit" FIXTURES_REQUIRED gguf)

    friday_test_target(test_cpu_kquants kernels/test_cpu_kquants.cpp)
    add_test(NAME cpu_kquants COMMAND test_cpu_kquants ${_fx})
    set_tests_properties(cpu_kquants PROPERTIES LABELS "gpu" FIXTURES_REQUIRED kquant)
else()
    message(STATUS "FRIDAY tests: Python 'gguf' + 'numpy' not found; GPU kernel tests disabled")
endif()

# --- model ------------------------------------------------------------------
set(FRIDAY_MODEL_DIR "" CACHE PATH
    "Directory with friday_aligned_int4.bin, tensor_map.json, tokenizer_config.json (enables model tests)")
friday_test_target(golden_check model/golden_check.cpp)
friday_test_target(test_tokenizer_qwen model/test_tokenizer_qwen.cpp)
if(FRIDAY_MODEL_DIR)
    set(_gguf ${FRIDAY_MODEL_DIR}/Qwen2.5-32B-Instruct-Q2_K.gguf)
    if(EXISTS ${_gguf})
        # CPU only: tokenizer from the GGUF metadata vs. reference Qwen2 tokenizer ids.
        add_test(NAME tokenizer_qwen_reference_gguf
                 COMMAND test_tokenizer_qwen ${_gguf}
                         ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/qwen2_encode_reference.json)
        set_tests_properties(tokenizer_qwen_reference_gguf PROPERTIES LABELS "model")
        # Golden test loading the GGUF directly (tensor map / tokenizer args unused: "-").
        add_test(NAME model_regression_qwen25_32b_q2k_gguf
                 COMMAND golden_check ${_gguf} -
                         ${CMAKE_CURRENT_SOURCE_DIR}/model/qwen25-32b-q2k - 12.0)
        set_tests_properties(model_regression_qwen25_32b_q2k_gguf PROPERTIES LABELS "model" TIMEOUT 600)
    endif()
    # Deprecated converter output (.bin + tensor_map.json + tokenizer_config.json), if present.
    if(EXISTS ${FRIDAY_MODEL_DIR}/friday_aligned_int4.bin AND EXISTS ${FRIDAY_MODEL_DIR}/tensor_map.json
       AND EXISTS ${FRIDAY_MODEL_DIR}/tokenizer_config.json)
        add_test(NAME tokenizer_qwen_reference
                 COMMAND test_tokenizer_qwen ${FRIDAY_MODEL_DIR}/tokenizer_config.json
                         ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/qwen2_encode_reference.json)
        set_tests_properties(tokenizer_qwen_reference PROPERTIES LABELS "model")
        add_test(NAME model_regression_qwen25_32b_q2k
                 COMMAND golden_check
                         ${FRIDAY_MODEL_DIR}/friday_aligned_int4.bin
                         ${FRIDAY_MODEL_DIR}/tensor_map.json
                         ${CMAKE_CURRENT_SOURCE_DIR}/model/qwen25-32b-q2k
                         ${FRIDAY_MODEL_DIR}/tokenizer_config.json
                         12.0)
        set_tests_properties(model_regression_qwen25_32b_q2k PROPERTIES LABELS "model" TIMEOUT 600)
    endif()
endif()
set(FRIDAY_QWEN3_GGUF "" CACHE FILEPATH "Qwen3-8B-Q4_K_M.gguf (Qwen/Qwen3-8B-GGUF) for the Qwen3 model tests")
if(FRIDAY_QWEN3_GGUF AND EXISTS ${FRIDAY_QWEN3_GGUF})
    add_test(NAME tokenizer_qwen3_reference
             COMMAND test_tokenizer_qwen ${FRIDAY_QWEN3_GGUF}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/qwen3_encode_reference.json)
    set_tests_properties(tokenizer_qwen3_reference PROPERTIES LABELS "model")
    add_test(NAME model_regression_qwen3_8b_q4km
             COMMAND golden_check ${FRIDAY_QWEN3_GGUF} - ${CMAKE_CURRENT_SOURCE_DIR}/model/qwen3-8b-q4km - 12.0)
    add_test(NAME model_regression_qwen3_8b_q4km_cpu
             COMMAND golden_check ${FRIDAY_QWEN3_GGUF} - ${CMAKE_CURRENT_SOURCE_DIR}/model/qwen3-8b-q4km - 3.0)
    set_tests_properties(model_regression_qwen3_8b_q4km model_regression_qwen3_8b_q4km_cpu
                         PROPERTIES LABELS "model" TIMEOUT 600)
endif()
set(FRIDAY_LLAMA31_GGUF "" CACHE FILEPATH
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (bartowski) for the Llama 3.1 model tests")
if(FRIDAY_LLAMA31_GGUF AND EXISTS ${FRIDAY_LLAMA31_GGUF})
    # Tokenizer vs. llama.cpp llama-tokenize ids (CPU only).
    add_test(NAME tokenizer_llama3_reference
             COMMAND test_tokenizer_qwen ${FRIDAY_LLAMA31_GGUF}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/llama3_encode_reference.json)
    set_tests_properties(tokenizer_llama3_reference PROPERTIES LABELS "model")
    # Golden vs. llama.cpp: all layers on the GPU, and a 3 GB VRAM cap that puts
    # most layers on the CPU path.
    add_test(NAME model_regression_llama31_8b_q4km
             COMMAND golden_check ${FRIDAY_LLAMA31_GGUF} - ${CMAKE_CURRENT_SOURCE_DIR}/model/llama31-8b-q4km - 12.0)
    add_test(NAME model_regression_llama31_8b_q4km_cpu
             COMMAND golden_check ${FRIDAY_LLAMA31_GGUF} - ${CMAKE_CURRENT_SOURCE_DIR}/model/llama31-8b-q4km - 3.0)
    set_tests_properties(model_regression_llama31_8b_q4km model_regression_llama31_8b_q4km_cpu
                         PROPERTIES LABELS "model" TIMEOUT 600)
endif()
