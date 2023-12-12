from mpi4py import MPI
import numpy as np
from typing import Callable, Tuple

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
        # Determine the local output size based on the number of devices
        local_output_size = output_size // self.size
        if self.rank == self.size - 1:
            local_output_size += output_size % self.size
        print(" output_size, local_output_size", (output_size, local_output_size))
        # Initialize weights for the local tensor
        self.local_w = weight_init_fn((local_output_size, input_size + 1))
        print("(local_output_size, input_size + 1)", (local_output_size, input_size + 1))
        self.local_w[:, 0] = 0  # Set bias weights to 0
        # Initialize gradients
        self.local_dw = np.zeros_like(self.local_w)
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:


    def backward(self, dz: np.ndarray) -> np.ndarray:

    
    def step(self) -> None:
        # Update local weights with local gradients
        self.local_w -= self.lr * self.local_dw


    