cmake_minimum_required(VERSION 3.18)
project(FridayEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

set(SOURCES
    main.cpp
    tokenizer.cpp
    transformer.cpp
    io_manager.cpp 
    memory_manager.cpp 
    cuda_math_kernels.cu
    int4_kernel.cu
    kernel_dispatcher.cpp
)

add_compile_options(-O3)

# sm_86 for RTX 3060 (user's hardware)
set(CMAKE_CUDA_ARCHITECTURES "86")

if(UNIX AND NOT APPLE)
    message(STATUS "Building on Linux: linking libaio and pthread.")
    target_link_libraries(FridayEngine PRIVATE aio pthread)
elseif(WIN32)
    message(STATUS "Building on Windows: standard libraries assumed.")
endif()

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
add_executable(FridayEngine ${SOURCES})

set_target_properties(FridayEngine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(FridayEngine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
