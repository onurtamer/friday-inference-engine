cmake_minimum_required(VERSION 3.18)

# Default: sm_86 (RTX 30-series). Must be set before project(), which otherwise
# initializes it to the compiler default (sm_52 with CUDA 12.x).
# Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()
project(FridayEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

option(FRIDAY_BUILD_TESTS "Build the test suite" ON)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
if(MSVC)
    # NVIDIA's static libraries (cudart_static, cudadevrt) request LIBCMT, while the
    # project uses the DLL runtime (/MD); drop the static CRT to avoid LNK4098.
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /NODEFAULTLIB:LIBCMT")
endif()

# Engine library: everything except the CLI entry point.
set(FRIDAY_ENGINE_SOURCES
    tokenizer.cpp
    transformer.cpp
    io_manager.cpp
    memory_manager.cpp
    cuda_math_kernels.cu
    int4_kernel.cu
    kernel_dispatcher.cpp
    hpc_k_quants.cu
    attention_kernels.cu
)
# Host-only CPU kernels, compiled by the C++ compiler (not nvcc).
set(FRIDAY_CPU_SOURCES
    gguf_file.cpp
    model_config.cpp
    cpu_kquants.cpp
    cpu_ops.cpp
    cpu_threadpool.cpp
)
set_source_files_properties(${FRIDAY_ENGINE_SOURCES} main.cpp PROPERTIES LANGUAGE CUDA)

add_library(friday_engine STATIC ${FRIDAY_ENGINE_SOURCES} ${FRIDAY_CPU_SOURCES})
# CPU K-quant kernels: AVX2/FMA/F16C code path (runtime-checked, scalar fallback).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|X86|x86")
    if(MSVC)
        set_source_files_properties(cpu_kquants.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(cpu_kquants.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    endif()
endif()
set_target_properties(friday_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(friday_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(UNIX AND NOT APPLE)
    # Linux builds are untested (see README).
    find_package(Threads REQUIRED)
    target_link_libraries(friday_engine PUBLIC Threads::Threads)
endif()

# CLI executable.
add_executable(FridayEngine main.cpp)
set_target_properties(FridayEngine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(FridayEngine PRIVATE friday_engine)

if(FRIDAY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
