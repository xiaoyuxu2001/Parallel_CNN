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
        self.w = weight_init_fn((output_size, input_size + 1))
        self.w[:, 0] = 0  # Set bias weights to 0
        # Initialize gradients
        self.dw = np.zeros_like(self.w)
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Insert bias term
        bias = np.ones((x.shape[0], 1))
        x = np.hstack((x, bias))
        print(x.shape)
        self.input = x
        print(self.w.T.shape)
  
        x_row, x_col = x.shape
        w_row, w_col = self.w.T.shape
        gathered_y = None
        assert(x_col == w_row)
        if self.layer_index == 0:
            gathered_y = matmul_parallel_divideA_horizontal(x, self.w.T, x_row, x_col, w_col)
        elif self.layer_index == 1:
            gathered_y = matmul_parallel_divideB_vertical(x, self.w.T, x_row, x_col, w_col)
        return gathered_y

    def backward(self, dz: np.ndarray) -> np.ndarray:
        
        # Compute local gradients for weights
        dz_row, dz_col = dz.T.shape
        in_row, in_col = self.input[:, 1:].shape
        w_row, w_col = self.w[:, 1:].shape
        print(dz_col, in_row)
        assert(dz_col == in_row)
        print(dz_col, dz_row, w_row, w_col)
        assert(dz_row == w_row)
        
        if self.layer_index == 1:
            print("Entering layer1 backward")
            print(dz_row, dz_col, in_row, in_col)
            print(dz.T.shape)
            self.dw = matmul_parallel_divideB_vertical(dz.T, self.input[:, 1:], dz_row, dz_col, in_col)
            print("Step layer1 backward")
            self.w = self.w - self.lr * self.dw
            dx =  matmul_parallel_divideB_vertical(dz, self.w[:, 1:].T, dz_col, dz_row, w_col)
        elif self.layer_index == 0:
            print("Entering layer0 backward")
            self.dw = matmul_parallel_divideA_horizontal(dz.T, self.input[:, 1:], dz_row, dz_col, in_col)
            print("Step layer0 backward")
            self.w = self.w - self.lr * self.dw
            dx =  matmul_parallel_divideA_horizontal(dz, self.w[:, 1:].T, dz_col,dz_row, w_col)
        # self.dw = np.dot(dz.T, self.input) / dz.shape[0]  # Averaging over the batch
        # self.w = self.w - self.lr * self.dw
        # dx = np.dot(dz, self.w)
        return dx
    
    # def step(self) -> None:
    #     # Update local weights with local gradients
    #     self.local_w -= self.lr * self.local_dw


    