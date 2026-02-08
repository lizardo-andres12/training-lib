#pragma once

#include <cuda_runtime.h>
#include <cuda/cmath>
#include <numeric>

namespace tl::kernels {

    // NxN square matrix multiplication kernel that uses shared memory tiling to reduce global memory access.
    // 2D block and grid dimensions are required. The tile width is recommended to be either 16 or 32
    template <uint32_t N, uint32_t TILE_WIDTH>
    __global__ void matmulsq_tiled(const float* A, const float* B, float* out) {
	const uint32_t tx = threadIdx.x;
	const uint32_t ty = threadIdx.y;

	const uint32_t row = blockIdx.y * TILE_WIDTH + ty;
	const uint32_t col = blockIdx.x * TILE_WIDTH + tx;

	float result = 0.f;

	__shared__ float smem_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float smem_B[TILE_WIDTH][TILE_WIDTH];

	if (row >= N || col >= N) [[unlikely]] {
	    return;
	}

	const uint32_t tileBound = cuda::ceil_div(N, TILE_WIDTH);
	for (uint32_t tileIdx{}; tileIdx < tileBound; ++tileIdx) {
	    const uint32_t tileOffset = tileIdx * TILE_WIDTH;

	    uint32_t aColIdx = tileOffset + tx;
	    if (row < N && aColIdx < N) [[likely]] {
	        smem_A[ty][tx] = A[(row * N) + aColIdx];
	    } else {
	        smem_A[ty][tx] = 0.f;
	    }

	    uint32_t bRowIdx = tileOffset + ty;
	    if (bRowIdx < N && col < N) [[likely]] {
	        smem_B[ty][tx] = B[(bRowIdx * N) + col];
	    } else {
	        smem_B[ty][tx] = 0.f;
	    }

	    __syncthreads();

	    for (uint32_t i{}; i < TILE_WIDTH; ++i) {
	        result += smem_A[ty][i] * smem_B[i][tx];
	    }

	    __syncthreads();
	}

	out[(row * N) + col] = result;
    }

    // MxK times KxN matrix multiplication kernel that uses shared memory tiling to reduce global memory access.
    // 2D block and grid dimensions are required. The tile width is recommended to be either 16 or 32
    template <uint32_t M, uint32_t K, uint32_t N, uint32_t TILE_WIDTH>
    __global__ void matmul_tiled(const float* mat_a, const float* mat_b, float* out) {
	constexpr uint32_t TILE_BOUND = cuda::ceil_div(N, TILE_WIDTH);

	const uint32_t tx = threadIdx.x;
	const uint32_t ty = threadIdx.y;

	const uint32_t row = blockIdx.y * TILE_WIDTH + ty;
	const uint32_t col = blockIdx.x * TILE_WIDTH + tx;

	float result = 0.f;

	__shared__ float smem_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float smem_b[TILE_WIDTH][TILE_WIDTH];

	for (uint32_t tile_idx{}; tile_idx < TILE_BOUND; ++tile_idx) {
	    const uint32_t tile_offset = tile_idx * TILE_WIDTH;

	    if (row < M && tx + tile_offset < K) {
		smem_a[ty][tx] = mat_a[(row * K) + tx + tile_offset];
	    } else {
		smem_a[ty][tx] = 0.f;
	    }

	    if (ty + tile_offset < K && col < N) {
		smem_b[ty][tx] = mat_b[(ty + tile_offset) * N + col];
	    } else {
		smem_b[ty][tx] = 0.f;
	    }

	    __syncthreads();

	    for (uint32_t i{}; i < TILE_WIDTH; ++i) {
		result += smem_a[ty][i] * smem_b[i][tx];
	    }

	    __syncthreads();
	}

	if (row < M && col < N) {
	    out[(row * N) + col] = result;
	}
    }


    // NxN square matrix multiplication kernel that uses shared memory tiling and thread coarsening to reduce
    // global memory access and decrease threads spawned per kernel by 1/COARSEN_FACTOR. Launching this kernel
    // requires setting your thread block y dimension to 1/COARSEN_FACTOR what it was previously to guarantee
    // no out of bounds accesses that cause the kernel to crash.
    //
    // i.e. dim3 blockDim(TILE_WIDTH, cuda::ceil_div(TILE_WIDTH, COARSEN_FACTOR))
    template <uint32_t N, uint32_t TILE_WIDTH, uint32_t COARSEN_FACTOR>
    __global__ void matmulsq_tiled_coarsened(const float* mat_a, const float* mat_b, float* out) {
	constexpr uint32_t COARSEN_OFFSET = cuda::ceil_div(TILE_WIDTH, COARSEN_FACTOR);

	const uint32_t bx = blockIdx.x * TILE_WIDTH;
	const uint32_t by = blockIdx.y * TILE_WIDTH;

	const uint32_t tx = threadIdx.x;
	const uint32_t ty = threadIdx.y;

	const uint32_t working_row = ty + by;
	const uint32_t working_col = tx + bx;

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

	if (working_col < N) {
	    for (uint32_t i{}; i < COARSEN_FACTOR; ++i) {
		const uint32_t row_offset = COARSEN_OFFSET * i;
		const uint32_t final_row = working_row + row_offset;

		if (final_row < N) {
		    out[final_row * N + working_col] = sums[i];
		}
	    }
	}
    }
}

