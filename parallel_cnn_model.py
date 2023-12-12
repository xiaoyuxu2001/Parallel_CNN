from conv import Conv2d
from maxpool import MaxPool2
from dense import  Relu, Flatten, Dense, random_init, SoftMaxCrossEntropy, shuffle
import numpy as np
from typing import Callable, List, Tuple
import logging
from data_parallel import data_parallel_forward, divide_data, ring_all_reduce
from model_parallel import Parallel_Linear

from mpi4py import MPI
import numpy as np



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
        self.dense_layers = [Parallel_Linear(20000, 128, random_init, learning_rate),
                        #   Relu(), 
                          Parallel_Linear(128, 2, random_init, learning_rate)]
        self.softmax = SoftMaxCrossEntropy()
        logging.info("model initialized")

    def forward(self, image, label):
        # perform the forward pass on the conv layers
        out_split = None
        partition_X = None
        if self.rank == 0:
            # Split the data into subsets for each worker
            partition_X = divide_data(image, self.size)
        out_split = self.comm.scatter(partition_X, root=0)
        # Apply the forward pass on each worker
        for layer in self.conv_layers:
            out_split = layer.forward(out_split)

        # allgatherv to collect the data from each process
        # calculate the size of the data to be received
        recv_num : int = (self.batch_num // self.size) * out_split.shape[1] 
        recv_size = np.full(self.size, recv_num, dtype=int)
        # calculate the displacement of the data to be received
        displacements = np.zeros(self.size, dtype=int)
        for i in range(1, self.size):
            displacements[i] = displacements[i - 1] + recv_size[i - 1]
        # calculate the total size of the data to be received
        total_recv_size = np.sum(recv_size)
        # allocate a buffer to hold the received data
        recvbuf = np.zeros(total_recv_size, dtype=np.float64)
        self.comm.Allgatherv([out_split, MPI.DOUBLE], [recvbuf, (recv_size, displacements), MPI.DOUBLE])

        #---------------------------perform the forward pass on the dense layers---------------------------
        out = recvbuf.reshape((self.batch_num, 20000))
        # print("out shape: ", out.shape)
        
        
        for i, layer in enumerate(self.dense_layers):
            try:
                out = layer.forward(out)
            except Exception as e:
                print(f"An error occurred on rank {self.rank} during forward pass: {e}")
                raise
            
            # After the first dense layer, broadcast 'out' from the root process to all other processes
            if i == 0:
                if self.rank == 0:
                    # The root process has the correct 'out', which will be broadcasted
                    out = np.where(out > 0, out, 0)
                else:
                    out = np.empty(self.batch_num * 128, dtype=np.float64)
                self.comm.Bcast(out, root=0)
                out = out.reshape((self.batch_num, 128))
                # print(out)
        gathered_output = out
        print(out)
        # print("Finished forward, shape is ",gathered_output.shape)

        # Compute loss and softmax only on the root
        if self.rank == 0:
            y_hat, loss = self.softmax.forward_batch(gathered_output, label)
            print("finish forward")
            return y_hat, loss
        else:
            print("finish forward non root")
            return None, None

    def backprop(self, label, label_hat):
        #---------------------------perform the backward pass on the dense layers---------------------------
        # Model parallelism for backward pass of fully connected layers
        # The gradient is scattered across workers
        print("start backprop")
        gradient = None
        if self.rank == 0:
            # Only the root has the complete label_hat and label
            gradient = self.softmax.backprop_batch(label, label_hat)
        # Perform backward pass on the local gradients
        local_grad = None
        for layer in reversed(self.dense_layers):
            local_grad = layer.backward(gradient)

        # Data parallelism for backward pass of convolutional layers
        # Gather the gradients from all workers to root
        # gathered_grads = None
        # if self.rank == 0:
        #     gathered_grads = np.empty([self.size, *local_grad.shape], dtype=local_grad.dtype)
        
        # self.comm.Gather(local_grad, gathered_grads, root=0)

        #---------------------------perform the backward pass on the conv layers----------------------------
        local_grad_bias = None
        local_grad_filter = None
        for layer in reversed(self.conv_layers):
            if isinstance(layer, "Conv2d"):
                print("Conv2d")
                local_grad_filter,  local_grad_bias= layer.backward(local_grad, False)
            else:
                local_grad = layer.backward(local_grad)
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
            X_s, y_s = None, None
            if self.rank == 0:
                index = np.random.choice(len(X_tr), batch_num, replace=True)
                X_s, y_s = X_tr[index], y_tr[index]
                # print("X_s's shape", X_s.shape) ## X_s's shape (128, 28, 28)
            y_hat, loss = self.forward(X_s, y_s)
            if self.rank == 0:
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
        if self.rank == 0:
            error = 0
            y_hat, loss = self.forward(X, y)
            y_predict = np.argmax(y_hat, axis = 1)
            error = np.count_nonzero(y_predict - y)
            return y_predict, error / len(X)
        else:
            return None, None