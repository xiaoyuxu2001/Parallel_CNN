from data_parallel import data_parallel_forward, divide_data, ring_all_reduce
from model_parallel import Parallel_Linear
import seq_operations as seq
from conv_seq import Conv2d
from maxpool_seq import MaxPool2
from dense_seq import  Relu, Flatten, Dense, random_init, SoftMaxCrossEntropy, shuffle
import numpy as np
from typing import Callable, List, Tuple
import logging
import time

from mpi4py import MPI



class ParallelCNN:
    def __init__(self, learning_rate, comm, rank, size, batch_num):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.batch_num = batch_num
        self.learning_rate = learning_rate
        # Initialize parallel layers
        self.conv_layers = [Conv2d(num_filters=32, kernel_size=(3, 3), learning_rate=learning_rate),
                            Relu(),
                            MaxPool2(),
                            Flatten()]
        self.dense_layers = [Parallel_Linear(20000, 128 // self.size, random_init, learning_rate, layer_index = 0),
                        #   Relu(), 
                          Parallel_Linear(128 // self.size, 2, random_init, learning_rate, layer_index = 1)]
        self.softmax = SoftMaxCrossEntropy()
        logging.info("model initialized")

    def forward(self, image, label):
        # 1. perform the forward pass on the conv layers!
        out_split = None
        partition_X = None
        if self.rank == 0:
            # If source node, split the data into subsets for each worker
            partition_X = divide_data(image, self.size)
        out_split = self.comm.scatter(partition_X, root=0) # data for current worker
        # Apply the forward pass on each worker
        for layer in self.conv_layers:
            out_split = layer.forward(out_split)

        # 2. allgatherv to collect the data from each process!
        # calculate the size of the data to be received
        recv_num : int = np.prod(out_split.shape)
        recv_size = np.full(self.size, recv_num, dtype=int) # array representing amount of data from each worker
        # calculate the displacement of the data to be received
        displacements = recv_num * np.arange(self.size)
        # calculate the total size of the data to be received
        total_recv_size = self.size * recv_num
        # allocate a buffer to hold the received data
        recvbuf = np.zeros(total_recv_size, dtype=np.float64)
        self.comm.Allgatherv([out_split, MPI.DOUBLE], [recvbuf, (recv_size, displacements), MPI.DOUBLE])

        # 3. perform the forward pass on the dense layers!
        out = recvbuf.reshape((total_recv_size//20000, 20000))

        for layer in self.dense_layers:
            out = layer.forward(out)
        
        # 4. gather the output from each process!
        gathered_output = None
        if self.rank == 0:
            # The receiving buffer size is the size of 'out' times the number of processes
            gathered_output = np.empty((self.size, *out.shape), dtype=out.dtype)

        # Gather the output from each process to the root
        self.comm.Gather(out, gathered_output, root=0)
        # sum the output from each process
        gathered_output = np.sum(gathered_output, axis=0)
        
        if self.rank == 0:
            y_hat, loss = self.softmax.forward_batch(gathered_output, label)
            return y_hat, loss
        else:
            return None, None

    def backprop(self, label, label_hat):
        # 1. perform the backward pass on the dense layers
        gradient_sent, gradient = None, None
        if self.rank == 0:
            gradient = self.softmax.backprop_batch(label, label_hat)
            # spread the gradient across the processes
            gradient_sent = np.full((self.size, *gradient.shape), gradient, dtype=np.float16)
        # scatter the gradient to each process
        gradient = self.comm.scatter(gradient_sent, root=0)
        for layer in reversed(self.dense_layers):
            gradient = layer.backprop(gradient)   
            self.comm.Barrier()   

        self.comm.Barrier()

        # 2. perform the backward pass on the conv layers
        local_grad_bias = None
        local_grad_filter = None
        gradient = ring_all_reduce(gradient, self.comm, self.rank, self.size)
        local_grad = gradient[self.rank * self.batch_num // self.size : (self.rank + 1) * self.batch_num // self.size]
        for i, layer in enumerate(reversed(self.conv_layers)):
            if isinstance(layer, Conv2d):
                local_grad_filter,  local_grad_bias= layer.backprop(local_grad, False)
            else:
                local_grad = layer.backprop(local_grad)        
        # perform the allreduce to get the global gradient
        # append the bias to the filter
        global_grad_bias = ring_all_reduce(local_grad_bias, self.comm, self.rank, self.size)
        global_grad_filter = ring_all_reduce(local_grad_filter, self.comm, self.rank, self.size)

        # apply to the corresponding conv layer
        self.conv_layers[0].filters -= self.learning_rate * (global_grad_filter / self.batch_num)
        self.conv_layers[0].bias -= self.learning_rate * (global_grad_bias / self.batch_num)

        return

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
            time_start = time.time()
            X_s, y_s = None, None
            if self.rank == 0:
                index = np.random.choice(len(X_tr), batch_num, replace=False)
                X_s, y_s = X_tr[index], y_tr[index]
            time_forward_start = time.time()
            y_hat, loss = self.forward(X_s, y_s)
            # if e >= 10:
                # print(y_hat)
            if self.rank == 0:
                print("loss:", np.mean(loss))
            time_forward_end = time.time()
            # print("Time for forward: ", format(time_forward_end - time_forward_start, '.2f'), "s")
            if self.backprop(y_s, y_hat):
                break
            time_end = time.time()
            # print("time for backprop: ", format(time_end - time_forward_end, '.2f'), "s")


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
        y_hat, loss = self.forward(X, y)
        if self.rank == 0:
            y_predict = np.argmax(y_hat, axis = 1)
            error = np.count_nonzero(y_predict - y)
            return y_predict, error / len(X)
        else:
            return None, None