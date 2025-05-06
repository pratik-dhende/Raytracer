cmake_minimum_required(VERSION 3.28)

project(CudaProject VERSION 1.0.0 LANGUAGES CXX CUDA)

# Modern CMake features
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 4050

# Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Architecture
if(WIN32)
    set(CMAKE_GENERATOR_PLATFORM x64)
endif()

# CUDA setup
find_package(CUDAToolkit REQUIRED)

# Main target
add_executable(Raytracer src/Main.cu)
# target_include_directories(Raytracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compiler options
option(DEBUG_MODE "Debug mode" ON)
target_compile_options(Raytracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<BOOL:${DEBUG_MODE}>:-g;-G>
        $<$<NOT:$<BOOL:${DEBUG_MODE}>>:-O3;--use_fast_math>
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<BOOL:${DEBUG_MODE}>:-g;-O0>
        $<$<NOT:$<BOOL:${DEBUG_MODE}>>:-O3>
    >
)

# CUDA libraries
target_link_libraries(Raytracer PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)

# Output
add_custom_command(TARGET Raytracer POST_BUILD
    COMMAND $<TARGET_FILE:Raytracer> > out.ppm
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Profiling
add_custom_target(profile
    COMMAND nsys profile --stats=true --trace=cuda,nvtx,osrt -o raytracer_profile $<TARGET_FILE:Raytracer>
    DEPENDS Raytracer
)

