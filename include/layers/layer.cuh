#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace tl {

    struct Layer {
	size_t in_features;
	size_t out_features;

	float* d_weights;
	float* d_biases;
	float* d_inputs;
	float* d_outputs;
	float* d_weight_grads;
	float* d_bias_grads;


	// Constructor that allocates all vector buffers to shape (1, in_features) and matrix buffers to shape (in_features, out_features)
	Layer(size_t in_features, size_t out_features);

	// Constructor that allocates all vector buffers to shape (1, in_features) and matrix buffers to shape (in_features, out_features). It will throw an exception meant to terminate the program if a non-device pointer is passed to the d_inputs parameter. Do not handle this exception 
	Layer(size_t in_features, size_t out_features, float* d_inputs);

	// Destructor uses streams to deallocate memory asynchronously and destorys the stream handle
	~Layer();

	Layer(const Layer& l) = delete;
	void operator=(const Layer& l) = delete;
	Layer(Layer&& l) = delete;
	void operator=(Layer&& l) = delete;

    private:
	// Stream allows concurrent construction/destruction of all layers in a network
	cudaStream_t _stream;
    };
}

