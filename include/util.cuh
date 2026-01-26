#pragma once

#include <iostream>

void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " : "
                  << cudaGetErrorString(err) << '\n';
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(val) checkCudaErrors((val), __FILE__, __LINE__)

