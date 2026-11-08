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
else()
    message(STATUS "FRIDAY tests: Python 'gguf' + 'numpy' not found; GPU kernel tests disabled")
endif()

# --- model ------------------------------------------------------------------
set(FRIDAY_MODEL_DIR "" CACHE PATH
    "Directory with friday_aligned_int4.bin, tensor_map.json, tokenizer_config.json (enables model tests)")
friday_test_target(golden_check model/golden_check.cpp)
friday_test_target(test_tokenizer_qwen model/test_tokenizer_qwen.cpp)
if(FRIDAY_MODEL_DIR)
    # CPU only: engine tokenizer vs. reference Qwen2 tokenizer ids.
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
