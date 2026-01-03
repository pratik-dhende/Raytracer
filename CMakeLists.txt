cmake_minimum_required(VERSION 3.10)

project(Raytracer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Raytracer src/Vec3.cpp src/Main.cpp)

target_include_directories(Raytracer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

add_custom_command(TARGET Raytracer POST_BUILD
    COMMAND $<TARGET_FILE:Raytracer> > out.ppm
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)