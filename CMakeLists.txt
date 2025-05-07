cmake_minimum_required(VERSION 3.28)

project(Raytracer VERSION 1.0.0 LANGUAGES CXX CUDA)

# Modern CMake features
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 4050

# Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA setup
find_package(CUDAToolkit 12.6 REQUIRED)

# Main target
add_executable(Raytracer src/Main.cu)

# CUDA libraries
target_link_libraries(Raytracer PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)

# Output
add_custom_command(TARGET Raytracer POST_BUILD
    COMMAND $<TARGET_FILE:Raytracer> > out.ppm
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

