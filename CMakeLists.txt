cmake_minimum_required(VERSION 3.18)
project(FridayEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default: sm_86 (RTX 30-series). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

find_package(CUDAToolkit REQUIRED)

option(FRIDAY_BUILD_TESTS "Build the test suite" ON)

add_compile_options(-O3)

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
set_source_files_properties(${FRIDAY_ENGINE_SOURCES} main.cpp PROPERTIES LANGUAGE CUDA)

add_library(friday_engine STATIC ${FRIDAY_ENGINE_SOURCES})
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
