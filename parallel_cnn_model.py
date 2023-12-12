from conv import Conv2d
from maxpool import MaxPool2
from dense import  Relu, Flatten, Dense, random_init, SoftMaxCrossEntropy, shuffle
import numpy as np
from typing import Callable, List, Tuple
import logging
from data_parallel import data_parallel_forward, divide_data
from model_parallel import Parallel_Linear

from mpi4py import MPI
import numpy as np



class ParallelCNN:
    def __init__(self, learning_rate, comm, rank, size):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.learning_rate = learning_rate
        # Initialize parallel layers
        self.conv_layers = [Conv2d(num_filters=32, kernel_size=(3, 3), learning_rate=learning_rate),
                            Relu(),
                            MaxPool2(),
                            Flatten()]
        self.dense_layers = [Parallel_Linear(20000, 128, random_init, learning_rate),
                        #   Relu(), 
                          Parallel_Linear(128, 2, random_init, learning_rate)]
        self.softmax = SoftMaxCrossEntropy()
        logging.info("model initialized")

    def forward(self, image, label, comm, rank, size):
        #---------------------------perform the forward pass on the conv layers---------------------------
        out_split = None
        partition_X = None
        if rank == 0:
            # Split the data into subsets for each worker
            partition_X = divide_data(image, size)
        # Send the data to each worker
        comm.scatter(partition_X, out_split, root=0)
        # Apply the forward pass on each worker
        for layer in self.conv_layers:
            if isinstance(layer, Conv2d):
                out_split = layer.forward(out_split)
            else:
                out_split = layer.forward(out_split)

        # allgatherv to collect the data from each process
        # calculate the size of the data to be received
        recv_size = np.zeros(comm.Get_size(), dtype=int)
        recv_size[comm.Get_rank()] = out_split.shape[0]
        recv_size = comm.allgather(recv_size)
        # calculate the displacement of the data to be received
        displacements = np.zeros(comm.Get_size(), dtype=int)
        for i in range(1, comm.Get_size()):
            displacements[i] = displacements[i - 1] + recv_size[i - 1]
        # calculate the total size of the data to be received
        total_recv_size = np.sum(recv_size)
        # allocate a buffer to hold the received data
        recvbuf = np.zeros(total_recv_size, dtype=np.float64)
        # perform the allgatherv
        comm.Allgatherv([out_split, MPI.FLOAT], [recvbuf, (recv_size, displacements), MPI.FLOAT])

        #---------------------------perform the forward pass on the dense layers---------------------------
        out = recvbuf.reshape((1, -1))
        print(out.shape) #(20000,)
        for layer in self.dense_layers:
            out = layer.forward(out)
            print(out.shape)

        # Gather outputs at root for softmax and loss
        gathered_output = None
        if rank == 0:
            gathered_output = np.empty([size, *out_split.shape], dtype=out_split.dtype)
        comm.Gather(out_split, gathered_output, root=0)
        print(out_split.shape)
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
        if self.rank == 0:
            # Only the root has the complete label_hat and label
            gradient = self.softmax.backprop(label, label_hat)
        # Perform backward pass on the local gradients
        for layer in reversed(self.dense_layers):
            local_grad = layer.backward(gradient)

        # Data parallelism for backward pass of convolutional layers
        # Gather the gradients from all workers to root
        gathered_grads = None
        if self.rank == 0:
            gathered_grads = np.empty([self.size, *local_grad.shape], dtype=local_grad.dtype)
        
        self.comm.Gather(local_grad, gathered_grads, root=0)

        # Continue the backward pass on the root worker
        if self.rank == 0:
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
        for e in range (n_epochs):
            print('--- Epoch %d ---' % (e))
            X_s, y_s = None, None
            if self.rank == 0:
                index = np.random.choice(len(X_tr), batch_num, replace=False)
                X_s, y_s = X_tr[index], y_tr[index]
            y_hat, loss = self.forward(X_s, y_s)
            print("loss: ", np.average(loss))
            if self.backprop(y_s, y_hat):
                break

        return train_loss_list, test_loss_list
    

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        error = 0
        y_hat, loss = self.forward(X, y)
        y_predict = np.argmax(y_hat, axis = 1)
        error = np.count_nonzero(y_predict - y)
        return y_predict, error / len(X)