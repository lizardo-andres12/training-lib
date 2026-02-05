#pragma once

#include <cuda_runtime.h>
#include <iostream>

// CUDA error check passes if there was no error and force quits the program if there was an error
void checkCudaErrors(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " : "
                  << cudaGetErrorString(err) << '\n';
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(val) checkCudaErrors((val), __FILE__, __LINE__)

// Custom assertion that verifies if the input poiner points to GPU memory. Exits program on failure via hardcoded CUDA_CHECK call
template <typename T>
void assert_device_pointer(T* ptr) {
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, d_inputs));
    
    if (attr.type != cudaMemoryTypeDevice) {
	CUDA_CHECK(cudaErrorInvalidDevice);
    }
}

