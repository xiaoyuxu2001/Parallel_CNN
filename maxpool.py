import numpy as np

class MaxPool2:
    """
    A Max Pooling layer using a configurable pool size and stride.
    """
    def __init__(self, pool_size=2, stride=1):
        self.pool_size = pool_size
        self.stride = stride
        
    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h = h // self.pool_size
        new_w = w // self.pool_size

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * self.pool_size):(i * self.pool_size+ self.pool_size),\
                                (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                yield im_region, i, j
                
    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        '''
        self.last_input = input
        n_rows, n_cols, n_channels = input.shape
        out_rows = (n_rows // self.pool_size)
        out_cols = (n_cols // self.pool_size)

        output = np.zeros((out_rows, out_cols, n_channels))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output
    
    def backprop(self, d_L_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        '''
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * self.pool_size + i2, j * self.pool_size  + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input
