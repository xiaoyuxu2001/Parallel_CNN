import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

class MaxPool2:
    """
    A Max Pooling layer using a configurable pool size and stride.
    """
    def __init__(self, pool_size=2, stride=1):
        self.pool_size = pool_size
        self.stride = stride
        
    def iterate_regions(self, image):
        '''
        Generates non-overlapping (???) 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h = h-self.stride
        new_w = w-self.stride

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[i:i+ self.pool_size, j:j+ self.pool_size]
                yield im_region, i, j
                
    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        '''
        self.last_input = input
        batch, width, height, n_channels = input.shape
        pooled_height = height - 1  # since 2x2 pooling, reduce dimension by 1
        pooled_width = width - 1

        pooled_array = np.zeros((batch, pooled_width, pooled_height, n_channels))
        for channel in range(n_channels):
            for i in range(pooled_height):
                for j in range(pooled_width):
                    window = input[:, i:i+2, j:j+2, channel]
                    pooled_array[:, i, j, channel] = np.max(window, axis=(1, 2))

        return pooled_array

        # maxpooling
    
    def backprop(self, d_L_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        '''
        d_L_d_input = np.zeros(self.last_input.shape)
        
        for b in range(len(self.last_input)):
            for im_region, i, j in self.iterate_regions(self.last_input[b]):
                h, w, f = im_region.shape
                amax = np.max(im_region, axis=(0,1))

                for i2 in range(h):
                    for j2 in range(w):
                        for f2 in range(f):
                            # If this pixel was the max value, copy the gradient to it.
                            if im_region[i2, j2, f2] == amax[f2]:
                                d_L_d_input[b, i + i2, j+ j2, f2] += d_L_d_out[b, i, j, f2]

        return d_L_d_input
