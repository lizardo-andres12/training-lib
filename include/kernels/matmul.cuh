#pragma once

#include <cuda_runtime.h>
#include <cuda/cmath>

namespace tl::kernels {

    // NxN matrix multiplication kernel that uses shared memory tiling to reduce global memory access.
    // 2D block and grid dimensions are required. The tile width is recommended to be either 16 or 32
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

    // MxN matrix multiplication kernel that uses shared memory tiling to reduce global memory access.
    // 2D block and grid dimensions are required. The tile width is recommended to be either 16 or 32
    template <size_t TILE_WIDTH>
	__global__ void matmul_tiled(const float* A, const float* B, float* out, const size_t M, const size_t N) {
	    const size_t tx = threadIdx.x;
	    const size_t ty = threadIdx.y;

	    const size_t row = blockIdx.y * TILE_WIDTH + ty;
	    const size_t col = blockIdx.x * TILE_WIDTH + tx;

	    float result = 0.f;

	    __shared__ float smem_A[TILE_WIDTH][TILE_WIDTH];
	    __shared__ float smem_B[TILE_WIDTH][TILE_WIDTH];

	    if (row >= M || col >= N) [[unlikely]] {
		return;
	    }

	    const size_t tileBound = cuda::ceil_div(N, TILE_WIDTH);
	    for (size_t tileIdx{}; tileIdx < tileBound; ++tileIdx) {
		const size_t tileOffset = tileIdx * TILE_WIDTH;

		size_t aColIdx = tileOffset + tx;
		if (row < N && aColIdx < M) [[likely]] {
		    smem_A[ty][tx] = A[(row * N) + aColIdx];
		} else {
		    smem_A[ty][tx] = 0.f;
		}

		size_t bRowIdx = tileOffset + ty;
		if (bRowIdx < N && col < M) [[likely]] {
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
}

