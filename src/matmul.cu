#include <ctime>

#include <cuda/cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <nvtx3/nvToolsExt.h>

#include "util.cuh"


template <size_t TILE_WIDTH>
__global__ void matmulsq_tiled(const float* A, const float* B, float* out, const size_t N) {
    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;

    const size_t row = blockIdx.y * TILE_WIDTH + ty;
    const size_t col = blockIdx.x * TILE_WIDTH + tx;

    float result = 0.f;

    __shared__ float smem_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float smem_B[TILE_WIDTH][TILE_WIDTH];

    if (row >= N || col >= N) [[unlikely]] {
	return;
    }

    const size_t tileBound = cuda::ceil_div(N, TILE_WIDTH);
    for (size_t tileIdx{}; tileIdx < tileBound; ++tileIdx) {
	const size_t tileOffset = tileIdx * TILE_WIDTH;

	size_t aColIdx = tileOffset + tx;
	if (row < N && aColIdx < N) [[likely]] {
	    smem_A[ty][tx] = A[(row * N) + aColIdx];
	} else {
	    smem_A[ty][tx] = 0.f;
	}

	size_t bRowIdx = tileOffset + ty;
	if (bRowIdx < N && col < N) [[likely]] {
	    smem_B[ty][tx] = B[(bRowIdx * N) + col];
	} else {
	    smem_B[ty][tx] = 0.f;
	}

	__syncthreads();

	for (size_t i{}; i < TILE_WIDTH; ++i) {
	    result += smem_A[ty][i] * smem_B[i][tx];
	}

	__syncthreads();
    }

    __syncthreads();
    out[(row * N) + col] = result;
}

__global__ void init_curand_state(curandState *state, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void initMat(curandState* state, float* mat, const size_t area, char a) {
    const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= area) [[unlikely]] {
	return;
    }

    mat[idx] = curand_uniform(&state[idx]);
}

int main() {
    static constexpr size_t N = 1 << 14;
    static constexpr size_t TOTAL_ELEMENTS = N * N;
    static constexpr size_t TILE_WIDTH = 16;

    static constexpr size_t DEFAULT_THREADS_PER_BLOCK = 512;
    static constexpr size_t DEFAULT_BLOCKS_PER_GRID = cuda::ceil_div(TOTAL_ELEMENTS, DEFAULT_THREADS_PER_BLOCK);

    float* out;
    float* devA;
    float* devB;
    float* devOut;
    curandState* devStates;

    CUDA_CHECK(cudaMallocHost(&out, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devA, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devOut, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devStates, TOTAL_ELEMENTS * sizeof(curandState)));

    init_curand_state<<<DEFAULT_BLOCKS_PER_GRID, DEFAULT_THREADS_PER_BLOCK>>>(devStates, std::time(nullptr));
    CUDA_CHECK(cudaGetLastError());

    initMat<<<DEFAULT_BLOCKS_PER_GRID, DEFAULT_THREADS_PER_BLOCK>>>(devStates, devA, TOTAL_ELEMENTS, 'a');
    CUDA_CHECK(cudaGetLastError());

    initMat<<<DEFAULT_BLOCKS_PER_GRID, DEFAULT_THREADS_PER_BLOCK>>>(devStates, devB, TOTAL_ELEMENTS, 'b');
    CUDA_CHECK(cudaGetLastError());

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(cuda::ceil_div(N, blockDim.x), cuda::ceil_div(N, blockDim.y));

    matmulsq_tiled<TILE_WIDTH><<<gridDim, blockDim>>>(devA, devB, devOut, N);
    CUDA_CHECK(cudaGetLastError());

    matmulsq_tiled<TILE_WIDTH><<<gridDim, blockDim>>>(devA, devB, devOut, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out, devOut, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devOut));
    CUDA_CHECK(cudaFree(devStates));
    CUDA_CHECK(cudaFreeHost(out));
}

