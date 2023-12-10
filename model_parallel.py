from mpi4py import MPI
import numpy as np
from typing import Callable, Tuple

INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]

class Parallel_Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Initialize learning rate for SGD
        self.lr = learning_rate
        
        # Determine the local output size based on the number of devices
        local_output_size = output_size // self.size
        if self.rank == self.size - 1:
            local_output_size += output_size % self.size
        
        # Initialize weights for the local tensor
        self.local_w = weight_init_fn((local_output_size, input_size + 1))
        self.local_w[:, 0] = 0  # Set bias weights to 0
        
        # Initialize gradients
        self.local_dw = np.zeros_like(self.local_w)
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Insert bias term
        x = np.insert(x, 0, 1, axis=1)
        self.input = x
        
        # Perform the local part of the forward pass
        local_y = np.dot(x, self.local_w.T)
        
        # Gather the outputs from all workers to form the full output
        gathered_y = None
        if self.rank == 0:
            gathered_y = np.empty((x.shape[0], self.local_w.shape[0] * self.size), dtype=np.float64)
        
        self.comm.Gather(local_y, gathered_y, root=0)
        
        # Only the root process will have the complete output
        return gathered_y

    def backward(self, dz: np.ndarray) -> np.ndarray:
        # Scatter the gradient among all workers
        local_dz = np.empty((dz.shape[0], self.local_w.shape[0]), dtype=np.float64)
        self.comm.Scatter(dz, local_dz, root=0)
        
        # Compute local gradients for weights
        self.local_dw = np.dot(local_dz.T, self.input)
        
        # Compute gradient for input
        grad_input = np.dot(local_dz, self.local_w[:, 1:])
        return grad_input
    
    def step(self) -> None:
        # Update local weights with local gradients
        self.local_w -= self.lr * self.local_dw


    