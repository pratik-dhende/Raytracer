#pragma once

#include <limits>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#define PI 3.1415926535897932385

constexpr double POSITIVE_INFINITY = std::numeric_limits<double>::infinity();

static inline double degreesToRadians(const double degree) {
    return degree * PI / 180.0;
}

static inline double random() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

static inline double random(double min, double max) {
    return min + (max - min) * random();
}

static void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " " << cudaGetErrorString(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
