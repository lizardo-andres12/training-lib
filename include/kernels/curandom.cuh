#pragma once

#include <cstdint>
#include <curand_kernel.h>
#include <cuda_runtime.h>

namespace tl::kernels {

    // GPU random number generator initialization kernel
    __global__ void init_curand_state(curandState* state, unsigned long long seed);

    // Weight and bias random number initializer kernel
    __global__ void init_random_weights(curandState* state, float* buf, const uint32_t size);
}

