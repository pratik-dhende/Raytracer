cmake_minimum_required(VERSION 3.10)

project(Raytracer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Raytracer src/Main.cpp src/Vec3.cpp)

target_include_directories(Raytracer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)