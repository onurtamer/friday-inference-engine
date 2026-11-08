cmake_minimum_required(VERSION 3.18)

# FRIDAY_ENABLE_CUDA=OFF builds the CPU backend only (no CUDA toolkit needed).
option(FRIDAY_ENABLE_CUDA "Build the CUDA backend (needs the CUDA toolkit)" ON)

# Default: sm_86 (RTX 30-series). Must be set before project(), which otherwise
# initializes it to the compiler default (sm_52 with CUDA 12.x).
# Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(FRIDAY_ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()
if(FRIDAY_ENABLE_CUDA)
    project(FridayEngine LANGUAGES CXX CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Host code that sees CUDA headers is compiled by nvcc so executables get
    # the device link step for the static library's relocatable device code.
    set(FRIDAY_HOST_LANGUAGE CUDA)
else()
    project(FridayEngine LANGUAGES CXX)
    set(FRIDAY_HOST_LANGUAGE CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(FRIDAY_BUILD_TESTS "Build the test suite" ON)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
if(MSVC AND FRIDAY_ENABLE_CUDA)
    # NVIDIA's static libraries (cudart_static, cudadevrt) request LIBCMT, while the
    # project uses the DLL runtime (/MD); drop the static CRT to avoid LNK4098.
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /NODEFAULTLIB:LIBCMT")
endif()

# CUDA backend (GPU kernels, GPU transformer, kernel dispatcher).
set(FRIDAY_CUDA_SOURCES
    transformer.cpp
    memory_manager.cpp
    cuda_math_kernels.cu
    int4_kernel.cu
    kernel_dispatcher.cpp
    hpc_k_quants.cu
    attention_kernels.cu
)
# Host code shared by both builds; engine/tokenizer/IO see CUDA headers only
# in the CUDA build (engine.cpp selects the backend).
set(FRIDAY_HOST_SOURCES
    engine.cpp
    tokenizer.cpp
    io_manager.cpp
)
# Pure C++ (compiled by the C++ compiler in every build).
set(FRIDAY_CPU_SOURCES
    system_info.cpp
    model_layout.cpp
    cpu_backend.cpp
    gguf_file.cpp
    model_config.cpp
    model_registry.cpp
    sha256.cpp
    cpu_kquants.cpp
    cpu_ops.cpp
    cpu_threadpool.cpp
)
set_source_files_properties(${FRIDAY_HOST_SOURCES} main.cpp PROPERTIES LANGUAGE ${FRIDAY_HOST_LANGUAGE})

if(FRIDAY_ENABLE_CUDA)
    set_source_files_properties(${FRIDAY_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_library(friday_engine STATIC ${FRIDAY_CUDA_SOURCES} ${FRIDAY_HOST_SOURCES} ${FRIDAY_CPU_SOURCES})
    set_target_properties(friday_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(friday_engine PUBLIC FRIDAY_HAVE_CUDA=1)
else()
    add_library(friday_engine STATIC ${FRIDAY_HOST_SOURCES} ${FRIDAY_CPU_SOURCES})
    target_compile_definitions(friday_engine PUBLIC FRIDAY_HAVE_CUDA=0)
endif()

# CPU K-quant kernels: AVX2/FMA/F16C code path (runtime-checked, scalar fallback).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|X86|x86")
    if(MSVC)
        set_source_files_properties(cpu_kquants.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(cpu_kquants.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    endif()
endif()
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
if(FRIDAY_ENABLE_CUDA)
    set_target_properties(FridayEngine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
target_link_libraries(FridayEngine PRIVATE friday_engine)

if(FRIDAY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
