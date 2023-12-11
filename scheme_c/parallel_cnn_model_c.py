from conv import Conv2d
from maxpool import MaxPool2
from dense import  Relu, Flatten, Dense, random_init, SoftMaxCrossEntropy, shuffle
import numpy as np
from typing import Callable, List, Tuple
import logging
import data_parallel_c
from model_parallel import Parallel_Linear
from mpi4py import MPI

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class ParallelCNN:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        # Initialize parallel layers
        self.conv_layers = data_parallel_c.Parallel_Convolution(learning_rate)
        self.dense_layers = [Parallel_Linear(20000, 128, random_init, learning_rate),
                        #   Relu(), 
                          Parallel_Linear(128, 2, random_init, learning_rate)]
        self.softmax = SoftMaxCrossEntropy()
        print('Parallel MNIST CNN initialized!')

    def forward(self, image, label, epoch):
        recvbuf = data_parallel_forward(image, comm, self)
        out = recvbuf.reshape((1, -1))
        print(out.shape) #(20000,)
        for layer in self.dense_layers:
            out = layer.forward(out)
            print(out.shape)

        # Gather outputs at root for softmax and loss
        gathered_output = None
        if rank == 0:
            gathered_output = np.empty([size, *out.shape], dtype=out.dtype)
        comm.Gather(out, gathered_output, root=0)
        print(out.shape)
        print(gathered_output.shape)

        # Compute loss and softmax only on the root
        if rank == 0:
            y_hat, loss = self.softmax.forward(gathered_output, label)
            return y_hat, loss
        else:
            return None, None

    def backward(self, label, label_hat):
        # Model parallelism for backward pass of fully connected layers
        # The gradient is scattered across workers
        if rank == 0:
            # Only the root has the complete label_hat and label
            gradient = self.softmax.backprop(label, label_hat)
        # Perform backward pass on the local gradients
        for layer in reversed(self.dense_layers):
            local_grad = layer.backward(gradient)

        # Data parallelism for backward pass of convolutional layers
        # Gather the gradients from all workers to root
        gathered_grads = None
        if rank == 0:
            gathered_grads = np.empty([size, *local_grad.shape], dtype=local_grad.dtype)
        
        comm.Gather(local_grad, gathered_grads, root=0)

        # Continue the backward pass on the root worker
        if rank == 0:
            # Continue with the convolutional layers
            gradient = gathered_grads
            for layer in reversed(self.conv_layers):
                gradient = layer.backward(gradient)
        return gradient

    def step(self):
        # Each worker updates its part of the model
        for layer in self.dense_layers:
            if hasattr(layer, 'step'):
                layer.step()

        # For the convolutional layers, each worker updates independently
        for layer in self.conv_layers:
            if hasattr(layer, 'step'):
                layer.step()
        
        # Synchronize the weights of the model parallel layers?

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int, batch_num: int) -> Tuple[List[float], List[float]]:
        train_loss_list = []
        test_loss_list = []
        # init the mpi
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        assert(size == 4)

        for e in range (n_epochs):
            print('--- Epoch %d ---' % (e))
            index = np.random.choice(len(X_tr), batch_num, replace=False) 
            X_s, y_s = X_tr[index], y_tr[index]

            # divide the data
            split_x = data_parallel_c.divide_data(X_s, 4)
            split_y = data_parallel_c.divide_data(y_s, 4)
            # spread the data across the processes
            batch_split_x = comm.scatter(split_x, root=0)
            batch_split_y = comm.scatter(split_y, root=0)

            # apply the forward pass on each process only on convolutional layers
            partial_conv = self.conv_layers.forward(batch_split_x)
            
            # there are 4 rounds in total, in each round, do forward and backward at the same time
            cur_round = 0
            while cur_round < 4:
                # assume 4 workers!!
                
                
                data_stored = data_parallel_c.gather_batch(partial_conv, cur_round, comm)


                # continue the forward pass using model parallelism
                # model.forward
                # 









        return train_loss_list, test_loss_list
    

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        y_predict_list = np.zeros(len(X))
        error = 0
        for i in range(len(X)):
            y_hat, loss = self.forward(X[i], y[i], 0)
            y_predict = np.argmax(y_hat)
            y_predict_list[i] = y_predict
            if y_predict != y[i]:
                error += 1
        return y_predict_list, error / len(X)
    


