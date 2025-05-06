#pragma once

#include "Utility.h"
#include <curand_kernel.h>

namespace Cuda{
    template<typename T>
    class SmartPointer {
    public:
        SmartPointer(const SmartPointer& other) = delete;
        SmartPointer& operator=(const SmartPointer& other) = delete;

        SmartPointer() : pointer(nullptr) {

        }

        explicit SmartPointer(int count, bool onHost) : pointer(nullptr) {
            alloc(count, onHost);
        }

        void alloc(int count, bool onHost) {
            if (pointer) {
                return;
            }

            if (onHost) {
                checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&pointer), sizeof(T) * count));
            }
            else {
                checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T) * count));
            }
        }

        ~SmartPointer() {
            if (pointer) {
                checkCudaErrors(cudaFree(pointer));
                pointer = nullptr;
            }
        }

        T* operator->() {
            return pointer;
        }

        T& operator[](int i) {
            return pointer[i];
        }

        T* get() {
            return pointer;
        }

    private:
        T* pointer;
    };

    __device__ static inline double random(curandState& randState) {;
        return curand_uniform(&randState);
    }
    
    __device__ static inline double random(double min, double max, curandState& randState) {
        return min + (max - min) * random(randState);
    }

    template<typename T>
    __host__ __device__ T max(const T& f, const T& s) {
        return f > s ? f : s;
    }

    template<typename T>
    __host__ __device__ T min(const T& f, const T& s) {
        return f < s ? f : s;
    }
}