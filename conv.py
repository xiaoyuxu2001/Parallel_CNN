import numpy as np

class Conv2d:
    """
    A Convolution layer with configurable filter sizes.
    """
    def __init__(self, num_filters, kernel_size, learning_rate):
        self.learning_rate = learning_rate
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        kernel_height, kernel_width = kernel_size
        self.filters = np.random.randn(kernel_height, kernel_width, num_filters) 
        self.bias = np.random.randn(num_filters)
        self.h = np.zeros(self.filters.shape)
        self.hb = np.zeros(self.bias.shape)
        self.rmsprop_rate = 0.85
        # print("self.filters: ", self.filters)

    def conv2d(self, input):
        """
        Apply a 2D convolution operation to the input array.
        """
        kernel_height, kernel_width = self.kernel_size
        n_filters = self.filters.shape[-1]
        n_rows, n_cols = input.shape
        out_height = n_rows - kernel_height + 1
        out_width = n_cols - kernel_width + 1

        output = np.zeros((out_height, out_width, n_filters))

        for i in range(out_height):
            for j in range(out_width):
                for k in range(n_filters):
                    output[i, j, k] = np.sum(np.multiply(input[i:i+kernel_height, j:j+kernel_width], self.filters[:, :, k])) + self.bias[k]

        return output

    def forward(self, input, epoch):
        """
        Performs a forward pass of the conv layer using the given input.
        """
        # print("forwarding: conv")
        # if epoch != 0 and epoch != 1:
            # self.learning_rate = self.learning_rate * (epoch-1) / (epoch)
        self.last_input = input
        return self.conv2d(input)

    def backprop(self, d_L_d_out):
        """
        Performs a backward pass of the conv layer.
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_bias = np.zeros(self.bias.shape)
        kernel_height, kernel_width = self.kernel_size
        n_rows, n_cols = self.last_input.shape

        for i in range(n_rows - kernel_height + 1):
            for j in range(n_cols - kernel_width + 1):
                for f in range(self.num_filters):
                    region = self.last_input[i:i+kernel_height, j:j+kernel_width]
                    d_L_d_filters[:, :, f] += d_L_d_out[i, j, f] * region
                    d_L_d_bias[f] += d_L_d_out[i, j, f]
        
        rms_f = np.sqrt(self.h + 1e-8)
        rms_b = np.sqrt(self.hb + 1e-8)

        self.filters -= self.learning_rate * np.multiply(1/rms_f, d_L_d_filters)
        self.bias -= self.learning_rate * np.multiply(1/rms_b, d_L_d_bias)

        self.h = (1-self.rmsprop_rate) * np.square(d_L_d_filters) + self.rmsprop_rate * self.h
        self.hb = (1-self.rmsprop_rate) * np.square(d_L_d_bias) + self.rmsprop_rate * self.hb
