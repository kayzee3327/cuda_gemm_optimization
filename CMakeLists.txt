cmake_minimum_required(VERSION 3.20)
project(cuda_gemm_optimization LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_executable(cuda_gemm
    src/main.cu
    src/00_trivial.cu
    src/utils/utils.cpp
)

add_executable(cublas_gemm
    src/baseline_cublas.cu
    src/utils/utils.cpp
)

target_link_libraries(cublas_gemm
    PRIVATE
    CUDA::cublas
)