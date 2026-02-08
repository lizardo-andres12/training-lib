#include "kernels/curandom.cuh"

__global__ void tl::kernels::init_curand_state(curandState *state, unsigned long long seed) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void tl::kernels::init_random_weights(curandState* state, float* buf, const uint32_t size) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) {
	return;
    }

    buf[idx] = curand_uniform(&state[idx]);
}

