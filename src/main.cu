#include <cstdint>
#include <ctime>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include <iostream>

#include "include/kernels/curandom.cuh"
#include "include/kernels/relu_matmul.cuh"

template <uint32_t N, uint32_t TILE_WIDTH, uint32_t COARSEN_FACTOR>
__global__ void matmulsq_tiled_coarsened(const float* mat_a, const float* mat_b, float* out) {
    constexpr uint32_t COARSEN_OFFSET = cuda::ceil_div(TILE_WIDTH, COARSEN_FACTOR);

    const uint32_t bx = blockIdx.x * TILE_WIDTH;
    const uint32_t by = blockIdx.y * TILE_WIDTH;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;

    const uint32_t working_row = ty + by;
    const uint32_t working_col = tx + bx;

    if (working_row >= N || working_col >= N) {
	return;
    }

    __shared__ float smem_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float smem_b[TILE_WIDTH][TILE_WIDTH];

    float sums[COARSEN_FACTOR] = {0.f};

    for (uint32_t tile_idx{}; tile_idx < cuda::ceil_div(N, TILE_WIDTH); ++tile_idx) {
	for (uint32_t i{}; i < COARSEN_FACTOR; ++i) {
	    const uint32_t row_offset = COARSEN_OFFSET * i;

	    const uint32_t row_a = working_row + row_offset;
	    const uint32_t col_a = TILE_WIDTH * tile_idx + tx;
	    if (row_a < N && col_a < N) {
		smem_a[ty + row_offset][tx] = mat_a[row_a * N + col_a];
	    } else {
		smem_a[ty + row_offset][tx] = 0.f;
	    }

	    const uint32_t row_b = (TILE_WIDTH * tile_idx) + ty + row_offset;
	    const uint32_t col_b = working_col;
	    if (row_b < N && col_b < N) {
		smem_b[ty + row_offset][tx] = mat_b[row_b * N + col_b];
	    } else {
		smem_b[ty + row_offset][tx] = 0.f;
	    }
	}

	__syncthreads();

	for (uint32_t i{}; i < TILE_WIDTH; ++i) {
	    const float b_val = smem_b[i][tx];

	    for (uint32_t j{}; j < COARSEN_FACTOR; ++j) {
		const uint32_t row_offset = COARSEN_OFFSET * j;
		sums[j] += smem_a[ty + row_offset][i] * b_val;
	    }
	}

	__syncthreads();
    }

    for (uint32_t i{}; i < COARSEN_FACTOR; ++i) {
	const uint32_t row_offset = COARSEN_OFFSET * i;
	const uint32_t final_row = working_row + row_offset;

	if (final_row < N) {
	    out[final_row * N + working_col] = sums[i];
	}
    }
}

int main() {
    static constexpr uint32_t TILE_WIDTH = 32;
    static constexpr uint32_t COARSEN_FACTOR = 4;
    static constexpr uint32_t N = 32;

    static constexpr uint32_t AREA = N * N;
    static constexpr uint32_t REDUCED_HEIGHT = TILE_WIDTH / COARSEN_FACTOR;

    float* dev_a;
    float* dev_b;
    float* dev_out;
    curandState* dev_states;
    float* host_a, * host_b, * host_out;

    cudaMallocHost(&host_a, sizeof(float) * AREA);
    cudaMallocHost(&host_b, sizeof(float) * AREA);
    cudaMallocHost(&host_out, sizeof(float) * AREA);

    cudaMalloc(&dev_a, sizeof(float) * AREA);
    cudaMalloc(&dev_b, sizeof(float) * AREA);
    cudaMalloc(&dev_out, sizeof(float) * AREA);
    cudaMalloc(&dev_states, sizeof(curandState) * AREA);

    tl::kernels::init_curand_state<<<N, N>>>(dev_states, std::time(nullptr));
    tl::kernels::init_random_weights<<<N, N>>>(dev_states, dev_a, AREA);
    tl::kernels::init_random_weights<<<N, N>>>(dev_states, dev_b, AREA);

    cudaMemcpy(host_a, dev_a, sizeof(float) * AREA, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b, dev_b, sizeof(float) * AREA, cudaMemcpyDeviceToHost);

    dim3 block_dim(TILE_WIDTH, REDUCED_HEIGHT);
    dim3 grid_dim(cuda::ceil_div(N, TILE_WIDTH), cuda::ceil_div(N, REDUCED_HEIGHT));
    tl::kernels::relu::matmulsq_tiled_coarsened<N, TILE_WIDTH, COARSEN_FACTOR><<<grid_dim, block_dim>>>(dev_a, dev_b, dev_out);

    cudaMemcpy(host_out, dev_out, sizeof(float) * AREA, cudaMemcpyDeviceToHost);

    for (uint32_t i{}; i < N; ++i) {
	for (uint32_t j{}; j < N; ++j) {
	    std::cout << host_a[i * N + j] << ' ';
	}
	std::cout << '\n';
    }

    std::cout << "================================================\n";

    for (uint32_t i{}; i < N; ++i) {
	for (uint32_t j{}; j < N; ++j) {
	    std::cout << host_b[i * N + j] << ' ';
	}
	std::cout << '\n';
    }

    std::cout << "================================================\n";

    for (uint32_t i{}; i < N; ++i) {
	for (uint32_t j{}; j < N; ++j) {
	    std::cout << host_out[i * N + j] << ' ';
	}
	std::cout << '\n';
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_out);
    cudaFree(dev_states);

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_out);
}

