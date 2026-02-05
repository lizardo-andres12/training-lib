#include "kernels/curandom.cuh"

using namespace tl::kernels;

__global__ void init_curand_state(curandState *state, unsigned long long seed) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void init_random_weights(curandState* state, float *buf, const int size) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) {
	return;
    }

    buf[idx] = curand_uniform(&state[idx]);
}

