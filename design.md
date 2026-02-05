# Dev Journal

## API

This library takes inspiration from PyTorch (that's what I'm most familiar with), but I don't really want to rip off the APIs and common patterns associated with PyTorch. I've been racking my brain on how I should design this API so that it's intuitive and let's the user not have to worry about any CUDA features.

### Layer Abstraction

Starting with a core part of deep learning, I'll talk about the design of layers. A layer can boil down to a matrix of weights (with their dimensions), a vector of biases, it's inputs, it's outputs, and the results of the gradients for the optimizer to update with. Since most of the computation should be done on the GPU, these should all be stored as device pointers in the GPU memory. To initialize the weights and biases, I will create a generic kernel that takes a layer and initializes allof the weights and biases to random values.

An initial problem I can spot with this implementation is the amount of memory taken by storing all of this information. A dense layer with 1 million neurons connected to another dense layer with 1 million neurons is too large for the GPU VRAM (my GPU VRAM is 16GiB i.e. 2^34, 1GiB by 1GiB is 2^30 * 2^30 = 2^60) with the weights matrix alone, but for the sake of not solving problems I don't have yet I'll ignore this scenario for now.

Another bottleneck will be the copying of output vectors to input vectors between layers because that would introduce a memory transfer overhead and a duplicated space overhead. My workaround is to simply set a layer's input buffer pointer to the preceding layer's output buffer. This would introduce the rule that I cannot modify the output of a layer until I've calculated all the gradients for connected layers and done the optimizer step, but with good operation ordering it shouldn't be an issue.

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

It would be really slow if for every layer all of the device buffers were allocated sequentially, so I decided that the layer constructor should create a CUDA stream and use `cudaMallocAsync` to allocate and `cudaFreeAsync` to deallocate the device buffers. This would allow the GPU to allocate and deallocate the memory concurrently and prevent network initialization and teardown from being a long wait for each layer to individually allocate or deallocate.

### Neural Network Abstraction

Now that the concept of layers has been introduced, the network abstraction can be introduced as well. Intuitively, a network can be thought of as a linear sequence of layers, and that is exactly how it is represented here. A network holds a vector of layers, with the input layer at index 0 and output layer at index n-1 (n being the number of layers), as well as an optimizer, a data loader, and a loss function. The optimizer, data loader, and loss function are all "strategy" parameters passed to the network meant to keep the network abstraction decoupled from other abstractions.

The network has public methods `forward`, `train`, `batched_train`, and `add_layer`. `forward` acts just as a forward pass does, generating predictions. The function call operator is also overloaded and it calls the `forward` method. `train` and `batched_train` take a data loader or batched data loader respectively as a parameter to load data to the input layer's device buffer. The network's `train` and `batched_train` are the intended ways to train a model, and should be preferred to custom training loops. `add_layer` is used to add a layer to the end of the network and will be validated against the previous layer's output dimension to prevent incompatible layer connections.

There are private methods `backward` and `step` used inside the training methods. `backward` uses a loss function that was passed to the class constructor to calculate the loss and the gradients, storing them in the respective layer buffers. `step` will use the optimizer object passed to the constructor and these calculated gradients to update the weights and biases.

### Data loader

Data loaders are the primary ways to generate input vectors for training. These can load batches or the entire dataset, though it is recommended to use batches as full datasets will typically overwhelm VRAM. Full dataset loaders will do a synchronous copy of the data into the device pointer, which is another reason to stay away from it if performance is most important. Batched loaders are more efficient as they asynchronously copy data from the CPU to the GPU in the background during the training loop.

Batched dataset loaders will load fixed-size batches in background threads into device pointers managed by the data loader class. The number of buffers is passed as an argument to the constructor and specifies how many batch buffers will be allocated on the GPU. You can optionally specify how many buffers are preloaded at a time. For example, you can allocate 4 buffers and 2 of those buffers should be full at any time. These values of 4 buffers and 2 preloaded are the default configurations for the batched data loaders. If the number of preloaded buffers that have not been used yet drops below the specified preload count, a thread will be spawned to load the data onto the cpu and execute an asynchronous copy.
