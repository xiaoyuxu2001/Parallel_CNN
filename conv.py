import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.compiler import SourceModule
# from pycuda.compiler import DynamicSourceModule
# import pycuda.driver as cuda
# import pycuda.gpuarray as gpuarray

class Conv2d:
    """
    A Convolution layer with configurable filter sizes.
    """
    def __init__(self, num_filters, kernel_size, learning_rate, CUDA=False):
        # Warning: need to send the attributes to the GPU if you want to use CUDA
        self.learning_rate = learning_rate
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        kernel_height, kernel_width = kernel_size
        self.filters = np.random.randn(kernel_height, kernel_width, num_filters) 
        self.bias = np.random.randn(num_filters)
        if CUDA:
            self.filters_gpu = gpuarray.to_gpu(self.filters)
            self.bias_gpu = gpuarray.to_gpu(self.bias)

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

        return output
    
    def conv2d_gpu(self, input):
        kernel_height, kernel_width = 3, 3
        n_filters = self.filters.shape[0]
        batch_num, n_rows, n_cols = input.shape
        out_height = n_rows - kernel_height + 1
        out_width = n_cols - kernel_width + 1

        output = np.zeros((batch_num, out_height, out_width, n_filters), dtype=np.float32)

        input_gpu = gpuarray.to_gpu(input.astype(np.float32))
        output_gpu = gpuarray.zeros((batch_num, out_height, out_width, n_filters), np.float32)

        # CUDA kernel
        kernel_code = """
        __global__ void Conv2DKernel(const float *input, const float *filters, float *output, const float *bias, 
                                     int kernel_height, int kernel_width, int n_filters, 
                                     int n_rows, int n_cols, int out_height, int out_width) {
            int X = blockIdx.x * blockDim.x + threadIdx.x;
            int Y = blockIdx.y * blockDim.y + threadIdx.y;
            int F = blockIdx.z * blockDim.z + threadIdx.z;

            if (X < out_height && Y < out_width && F < n_filters) {
                float value = 0;
                for (int i = 0; i < kernel_height; ++i) {
                    for (int j = 0; j < kernel_width; ++j) {
                        value += input[(i + X) * n_cols + (j + Y)] * filters[F * kernel_height * kernel_width + i * kernel_width + j];
                    }
                }
                output[X * out_width * n_filters + Y * n_filters + F] = value + bias[F];
            }
        }
        """

        mod = SourceModule(kernel_code)
        conv2d_kernel = mod.get_function("Conv2DKernel")

        # Grid and block dimensions
        block_size = (16, 16, 1) 
        grid_size = (int(np.ceil(out_height / block_size[0])), int(np.ceil(out_width / block_size[1])), n_filters)

        # Launch kernel
        conv2d_kernel(input_gpu, self.filters_gpu, output_gpu, self.bias_gpu,
                      np.int32(kernel_height), np.int32(kernel_width), np.int32(n_filters),
                      np.int32(n_rows), np.int32(n_cols), np.int32(out_height), np.int32(out_width),
                      block=block_size, grid=grid_size)

        output_gpu.get(output)
        return output

    def forward(self, input):
        """
        Performs a forward pass of the conv layer using the given input.
        """
        self.last_input = input
        return self.conv2d(input)

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
        d_L_d_filters_avg = np.mean(d_L_d_filters, axis=0)
        d_L_d_bias_avg = np.mean(d_L_d_bias, axis=0)
        
        # Update weights and biases
        if update_weights:
            self.filters -= self.learning_rate * d_L_d_filters_avg
            self.bias -= self.learning_rate * d_L_d_bias_avg
        
        return d_L_d_filters_avg, d_L_d_bias_avg
    
    def backprop_gpu(self, d_L_d_out, update_weights=True):
        """
        Performs a backward pass of the conv layer using CUDA.
        """
        batch, n_rows, n_cols, _ = d_L_d_out.shape
        kernel_height, kernel_width = self.kernel_size

        # Transfer d_L_d_out to GPU
        d_L_d_out_gpu = gpuarray.to_gpu(d_L_d_out.astype(np.float32))

        # Allocate GPU memory for gradients
        d_L_d_filters_gpu = gpuarray.zeros((batch, *self.filters.shape), np.float32)
        d_L_d_bias_gpu = gpuarray.zeros((batch, *self.bias.shape), np.float32)

        # CUDA kernel
        kernel_code = """
        __global__ void BackpropKernel(const float *d_L_d_out, const float *last_input, 
                                    float *d_L_d_filters, float *d_L_d_bias, 
                                    int kernel_height, int kernel_width, int n_filters, 
                                    int n_rows, int n_cols) {
            int X = blockIdx.x * blockDim.x + threadIdx.x;
            int Y = blockIdx.y * blockDim.y + threadIdx.y;
            int F = blockIdx.z * blockDim.z + threadIdx.z;

            if (X < n_rows - kernel_height + 1 && Y < n_cols - kernel_width + 1 && F < n_filters) {
                for (int b = 0; b < batch; ++b) {
                    float region_product = 0;
                    for (int i = 0; i < kernel_height; ++i) {
                        for (int j = 0; j < kernel_width; ++j) {
                            region_product += last_input[b * n_rows * n_cols + (i + X) * n_cols + (j + Y)] * d_L_d_out[b * n_filters * n_rows + F * n_rows + X * n_cols + Y];
                        }
                    }
                    atomicAdd(&d_L_d_filters[b * n_filters * kernel_height * kernel_width + F * kernel_height * kernel_width + X * kernel_width + Y], region_product);
                    atomicAdd(&d_L_d_bias[b * n_filters + F], d_L_d_out[b * n_filters * n_rows + F * n_rows + X * n_cols + Y]);
                }
            }
        }
        """

        mod = SourceModule(kernel_code)
        backprop_kernel = mod.get_function("BackpropKernel")

        # Grid and block dimensions
        block_size = (16, 16, 1)  # Example values, adjust as needed
        grid_size = (int(np.ceil((n_rows - kernel_height + 1) / block_size[0])), int(np.ceil((n_cols - kernel_width + 1) / block_size[1])), self.num_filters)

        # Launch kernel
        backprop_kernel(d_L_d_out_gpu, self.last_input_gpu, d_L_d_filters_gpu, d_L_d_bias_gpu,
                        np.int32(kernel_height), np.int32(kernel_width), np.int32(self.num_filters),
                        np.int32(n_rows), np.int32(n_cols),
                        block=block_size, grid=grid_size)

        # Copy gradients back to CPU and compute averages
        d_L_d_filters = d_L_d_filters_gpu.get()
        d_L_d_bias = d_L_d_bias_gpu.get()
        d_L_d_filters_avg = np.mean(d_L_d_filters, axis=0)
        d_L_d_bias_avg = np.mean(d_L_d_bias, axis=0)

        # Update weights and biases
        if update_weights:
            self.filters -= self.learning_rate * d_L_d_filters_avg
            self.bias -= self.learning_rate * d_L_d_bias_avg

        return d_L_d_filters_avg, d_L_d_bias_avg

