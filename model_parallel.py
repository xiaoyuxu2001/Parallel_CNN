from mpi4py import MPI
import numpy as np
from typing import Callable, Tuple
from fast_matrix_mult import matmul_parallel_divideA_horizontal, matmul_parallel_divideB_vertical

INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]

class Parallel_Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float, layer_index: int):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Initialize learning rate for SGD
        self.lr = learning_rate
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
        # Insert bias term
        if self.biased:
            bias = np.ones((x.shape[0], 1))
            x = np.hstack((bias, x))
        else:
            x = x
        print(x.shape)
        self.input = x
        print(self.w.T.shape)

        return np.dot(x, self.w.T)
  
        # x_row, x_col = x.shape
        # w_row, w_col = self.w.T.shape
        # gathered_y = None
        # assert(x_col == w_row)
        # if self.layer_index == 0:
        #     gathered_y = matmul_parallel_divideA_horizontal(x, self.w.T, x_row, x_col, w_col)
        # elif self.layer_index == 1:
        #     gathered_y = matmul_parallel_divideB_vertical(x, self.w.T, x_row, x_col, w_col)
        # return gathered_y

    def backprop(self, dz: np.ndarray) -> np.ndarray:
        self.dw = np.dot(dz.T, self.input) / dz.shape[0]  # Averaging over the batch
        self.w = self.w - self.lr * self.dw
        # Compute gradient w.r.t. inputs (dx)
        
        if self.biased:
            dx = np.dot(dz, self.w[:, 1:])
        else:
            dx = np.dot(dz, self.w)
        return dx
    
    # def step(self) -> None:
    #     # Update local weights with local gradients
    #     self.local_w -= self.lr * self.local_dw


    