from mpi4py import MPI
import numpy as np
from typing import Callable, Tuple
import seq_operations as seq
import parallel_operations as par
import time

INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]

class Parallel_Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float, layer_index: int, parallel = True):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Initialize learning rate for SGD
        self.lr = learning_rate
        self.parallel = parallel
        self.layer_index = layer_index
        # Initialize weights for the local tensor
        self.biased = False
        if self.layer_index == 0:
            self.w = weight_init_fn((output_size, input_size + 1))
            self.w[:, 0] = 0
            self.biased = True
        elif self.layer_index == 1 and self.rank == 0:
            self.w = weight_init_fn((output_size, input_size + 1))
            self.w[:, 0] = 0
            self.biased = True
        else:
            self.w = weight_init_fn((output_size, input_size))
        # Initialize gradients
        self.dw = np.zeros_like(self.w)
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        assume that the data is already splited
        '''
        if self.biased:
            bias = np.ones((x.shape[0], 1))
            x = np.hstack((bias, x))
        else:
            x = x

        self.input = x

        if self.parallel:
            res = par.dot(x, self.w.T)
        else:
            res = seq.dot(x, self.w.T)
            
        return res
    

    def backprop(self, dz: np.ndarray) -> np.ndarray:
        if self.parallel:
            self.dw = par.dot(dz.T, self.input) / dz.shape[0]
        else:
            self.dw = seq.dot(dz.T, self.input) / dz.shape[0]  # Averaging over the batch
        # Compute gradient w.r.t. inputs (dx)
        dx = None
        if self.biased:
            if self.parallel:
                dx = par.dot(dz, self.w[:, 1:])
            else:
                dx = seq.dot(dz, self.w[:, 1:])
        else:
            if self.parallel:
                dx = par.dot(dz, self.w)
            else:
                dx = seq.dot(dz, self.w)
        
        # update weight
        self.w = self.w - self.lr * self.dw
        return dx

    