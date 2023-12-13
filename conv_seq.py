import numpy as np
import seq_operations as seq
# from pycuda.compiler import DynamicSourceModule
# import pycuda.driver as cuda
# import pycuda.gpuarray as gpuarray

class Conv2d:
    """
    A Convolution layer with configurable filter sizes.
    """
    def __init__(self, num_filters, kernel_size, learning_rate):
        # Warning: need to send the attributes to the GPU if you want to use CUDA
        self.learning_rate = learning_rate
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        kernel_height, kernel_width = kernel_size
        self.filters = np.random.randn(kernel_height, kernel_width, num_filters) 
        self.bias = np.random.randn(num_filters)
        # self.h = np.zeros(self.filters.shape)
        # self.hb = np.zeros(self.bias.shape)
        # self.rmsprop_rate = 0.85
        # print("self.filters: ", self.filters)

    def conv2d(self, input):
        """
        Apply a 2D convolution operation to the input array, which is a batch of images.
        """
        kernel_height, kernel_width = self.kernel_size
        n_filters = self.filters.shape[-1]
        batch_num, n_rows, n_cols = input.shape
        out_height = n_rows - kernel_height + 1
        out_width = n_cols - kernel_width + 1

        output = np.zeros((batch_num, out_height, out_width, n_filters))

        for i in range(out_height):
            for j in range(out_width):
                for f in range(n_filters):
                    region = input[:, i:i+kernel_height, j:j+kernel_width]
                    output[:, i, j, f] = np.sum(region * self.filters[:, :, f], axis=(1, 2)) + self.bias[f]
                    # assert(np.all(seq.sum(region * self.filters[:, :, f], axis=(1, 2))) == np.all(np.sum(region * self.filters[:, :, f], axis=(1, 2))))
        return output
    
    def conv2d_cuda(self, input):
        kernel_height, kernel_width = self.kernel_size
        n_filters = self.filters.shape[-1]
        n_rows, n_cols = input.shape
        out_height = n_rows - kernel_height + 1
        out_width = n_cols - kernel_width + 1

        output = gpuarray.zeros((out_height, out_width, n_filters), np.float32)
        mod = DynamicSourceModule(open("conv2d_kernel.so", "rb").read())
        conv2d_kernel = mod.get_function("conv2d_kernel")

        # Set up execution configuration
        block_size = (16, 16, 1)
        grid_size = (int(np.ceil(out_height / block_size[0])), int(np.ceil(out_width / block_size[1])), n_filters)

        # Launch the kernel
        conv2d_kernel(cuda.In(input), self.filters, self.bias, output, 
                      np.int32(kernel_height), np.int32(kernel_width), 
                      np.int32(out_height), np.int32(out_width), 
                      np.int32(n_filters), np.int32(n_rows), np.int32(n_cols), 
                      block=block_size, grid=grid_size)

        return output.get()

    def forward(self, input):
        """
        Performs a forward pass of the conv layer using the given input.
        """
        self.last_input = input
        return self.conv2d(input)
    
    def backprop_cuda(self, d_L_d_out):
        d_L_d_out_gpu = gpuarray.to_gpu(d_L_d_out)
        d_L_d_filters_gpu = gpuarray.zeros(self.filters.shape, np.float32)
        d_L_d_bias_gpu = gpuarray.zeros(self.bias.shape, np.float32)

        # Load the compiled CUDA kernel
        mod = DynamicSourceModule(open("backprop_kernel.so", "rb").read())
        backprop_kernel = mod.get_function("backprop_kernel")

        # Set up execution configuration
        block_size = (16, 16, 1)
        grid_size = (int(np.ceil((self.last_input.shape[0] - self.kernel_size[0] + 1) / block_size[0])),
                     int(np.ceil((self.last_input.shape[1] - self.kernel_size[1] + 1) / block_size[1])),
                     self.num_filters)

        # Launch the kernel
        backprop_kernel(d_L_d_out_gpu, self.last_input_gpu, d_L_d_filters_gpu, 
                        d_L_d_bias_gpu, np.int32(self.kernel_size[0]), np.int32(self.kernel_size[1]), 
                        np.int32(self.last_input.shape[0]), np.int32(self.last_input.shape[1]), 
                        np.int32(self.num_filters), block=block_size, grid=grid_size)

        # Update weights and biases
        self.filters -= self.learning_rate * d_L_d_filters_gpu.get()
        self.bias -= self.learning_rate * d_L_d_bias_gpu.get()

    def backprop(self, d_L_d_out, update_weights=True):
        """
        Performs a backward pass of the conv layer.
        """
        batch, n_rows, n_cols = self.last_input.shape
        d_L_d_filters = np.zeros((batch ,*self.filters.shape))
        d_L_d_bias = np.zeros((batch,*self.bias.shape))
        kernel_height, kernel_width = self.kernel_size
        for i in range(n_rows - kernel_height + 1):
            for j in range(n_cols - kernel_width + 1):
                for f in range(self.num_filters):
                    for b in range(self.last_input.shape[0]):
                        region = self.last_input[b, i:i+kernel_height, j:j+kernel_width]
                        d_L_d_filters[b, :, :, f] += region * d_L_d_out[b, i, j, f]
                        d_L_d_bias[b, f] += d_L_d_out[b, i, j, f]
        
        # get the average of the gradients
        d_L_d_filters_avg = seq.mean(d_L_d_filters, axis=0)
        assert(np.all(seq.mean(d_L_d_filters, axis=0)) == np.all(np.mean(d_L_d_filters, axis=0)))
        d_L_d_bias_avg = seq.mean(d_L_d_bias, axis=0)
        assert(np.all(seq.mean(d_L_d_bias, axis=0)) == np.all(np.mean(d_L_d_bias, axis=0)))
        # Update weights and biases
        if update_weights:
            self.filters -= self.learning_rate * d_L_d_filters_avg
            self.bias -= self.learning_rate * d_L_d_bias_avg
        
        return d_L_d_filters_avg, d_L_d_bias_avg
