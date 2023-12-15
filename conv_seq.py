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
        d_L_d_filters_avg = seq.mean(d_L_d_filters, axis=0)
        d_L_d_bias_avg = seq.mean(d_L_d_bias, axis=0)
        # Update weights and biases
        if update_weights:
            self.filters -= self.learning_rate * d_L_d_filters_avg
            self.bias -= self.learning_rate * d_L_d_bias_avg
        
        return d_L_d_filters_avg, d_L_d_bias_avg

