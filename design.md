# Dev Journal

## API

This library takes inspiration from PyTorch (that's what I'm most familiar with), but I don't really want to rip off the APIs and common patterns associated with PyTorch. I've been racking my brain on how I should design this API so that it's intuitive and let's the user not have to worry about any CUDA features.

### Layer Abstraction

Starting with a core part of deep learning, I'll talk about the design of layers. A layer can boil down to a matrix of weights (with their dimensions), a vector of biases, it's inputs, it's outputs, and the results of the gradients for the optimizer to update with. Since most of the computation should be done on the GPU, these should all be stored as device pointers in the GPU memory. To initialize the weights and biases, I will create a generic kernel that takes a layer and initializes allof the weights and biases to random values.

An initial problem I can spot with this implementation is the amount of memory taken by storing all of this information. A dense layer with 1 million neurons connected to another dense layer with 1 million neurons is too large for the GPU VRAM (my GPU VRAM is 16GiB i.e. 2^34, 1GiB by 1Gib is 2^30 * 2^30 = 2^60) with the weights matrix alone, but for the sake of not solving problems I don't have I'll ignore this scenario for now.

Another bottleneck will be the copying of output vectors to input vectors because they introduce a memory transfer overhead and a duplicated space overhead, but I can solve this by simply pointing to the previous layer's output vector. This would introduce the rule that I cannot modify the output of a layer until I've calculated the gradients for subsequent layers, but with good operation ordering it shouldn't be an issue.

I thought about handling the forward pass within a layer, but I thought that would make the API awkward since the computation would actually be handled on the GPU, and GPU kernels are procedural, not object-oriented. For that reason, I'm deciding to treat the Layer object as a POD object with extended lifetime handling (constructor/destructor and internal stream). The forward pass  will be a kernel (as well as backward pass/optmizer step). The forward pass kernel takes an input layer reference and an output layer reference and stores results in the objects themselves.

```cpp
struct Layer {
    // Dimensions, in_features is the number of rows and out_features is the number of columns
    uint32_t in_features;
    uint32_t out_features;

    // Buffer device pointers
    float* d_weights;
    float* d_biases;
    float* d_input;
    float* d_output;
    float* d_weight_grads;
    float* d_bias_grads;
}
```

It would be really slow if for every layer all of the device buffers were allocated sequentially, so I decided that the layer constructor should create a CUDA stream and use cudaMallocAsync to allocate the device buffers and cudaFreeAsync to deallocate the device buffers. This would allow the GPU to allocate and deallocate the memory concurrently and prevent network initialization and teardown from being a long wait for each layer to individually allocate or deallocate.

