from conv import Conv2d
from maxpool import MaxPool2
from dense import  Relu, Flatten, Dense, random_init, SoftMaxCrossEntropy, shuffle
import numpy as np
from typing import Callable, List, Tuple
import logging
from data_parallel import data_parallel_forward
from model_parallel import Parallel_Linear

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class ParallelCNN:
    def __init__(self, learning_rate):
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
        print('Parallel MNIST CNN initialized!')

    def forward(self, image, label, epoch):
        recvbuf = data_parallel_forward(image, comm, self)
        out = recvbuf
        print(out.shape) #(20000,)
        for layer in self.dense_layers:
            out = layer.forward(out)

        # Gather outputs at root for softmax and loss
        gathered_output = None
        if rank == 0:
            gathered_output = np.empty([size, *out.shape], dtype=out.dtype)
        
        comm.Gather(out, gathered_output, root=0)

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
              n_epochs: int) -> Tuple[List[float], List[float]]:
        train_loss_list = []
        test_loss_list = []
        for e in range (n_epochs):
            print('--- Epoch %d ---' % (e))
            # X_s, y_s = shuffle(X_tr, y_tr, e)
            index = np.random.choice(len(X_tr)) 
            X_s, y_s = X_tr[index], y_tr[index]
            y_hat, loss = self.forward(X_s, y_s, e)
            if self.backprop(y_s, y_hat):
                break
            self.step()

            # for i, (im, label) in enumerate(zip(X_s, y_s)):
                # y_hat, loss = self.forward(X_s[i], y_s[i])
                # self.backprop(y_s[i], y_hat)
                # self.step()
            if e % 500 == 0 and e!= 0:
                train_loss = self.compute_loss(X_tr, y_tr)
                print("train loss: ", train_loss)
                train_loss_list.append(train_loss)
                # test_loss = self.compute_loss(X_test, y_test)
                # print("test loss: ", test_loss)
                # test_loss_list.append(test_loss)
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