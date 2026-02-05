#include "layers/layer.cuh"
#include "util.cuh"

using namespace tl;

Layer::Layer(size_t in_features, size_t out_features)
    : in_features(in_features)
    , out_features(out_features)
{
    CUDA_CHECK(cudaStreamCreate(&_stream));

    cudaMallocAsync(&d_weights, sizeof(float) * in_features * out_features, _stream);
    cudaMallocAsync(&d_biases, sizeof(float) * in_features, _stream);
    cudaMallocAsync(&d_inputs, sizeof(float) * in_features, _stream);
    cudaMallocAsync(&d_outputs, sizeof(float) * in_features, _stream);
    cudaMallocAsync(&d_weight_grads, sizeof(float) * in_features * out_features, _stream);
    cudaMallocAsync(&d_bias_grads, sizeof(float) * in_features, _stream);
}

Layer::Layer(size_t in_features, size_t out_features, float* d_inputs)
    : in_features(in_features)
    , out_features(out_features)
    , d_inputs(d_inputs)
{
    assert_device_pointer(d_inputs);

    CUDA_CHECK(cudaStreamCreate(&_stream));

    cudaMallocAsync(&d_weights, sizeof(float) * in_features * out_features, _stream);
    cudaMallocAsync(&d_biases, sizeof(float) * in_features, _stream);
    cudaMallocAsync(&d_outputs, sizeof(float) * in_features, _stream);
    cudaMallocAsync(&d_weight_grads, sizeof(float) * in_features * out_features, _stream);
    cudaMallocAsync(&d_bias_grads, sizeof(float) * in_features, _stream);

    cudaStreamSynchronize(_stream);
}

Layer::~Layer() {
    cudaFreeAsync(d_weights, _stream);
    cudaFreeAsync(d_biases, _stream);
    cudaFreeAsync(d_inputs, _stream);
    cudaFreeAsync(d_outputs, _stream);
    cudaFreeAsync(d_weight_grads, _stream);
    cudaFreeAsync(d_bias_grads, _stream);

    cudaStreamSynchronize(_stream);
    cudaStreamDestroy(_stream);
}

void Layer::set_input_ptr(float* ptr) {
    assert_device_pointer(ptr);

    cudaFreeAsync(d_inputs, _stream);
    d_inputs = ptr;
}

