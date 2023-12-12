from conv import Conv2d
from maxpool import MaxPool2
from dense import  Relu, Flatten, Dense, random_init, SoftMaxCrossEntropy, shuffle
import numpy as np
from typing import Callable, List, Tuple
import logging

# todo: improve the model, right now it is very shabby
class CNN:
    def __init__(self, learning_rate) -> None:
        # Initialize layers
        self.layers = [
            Conv2d(num_filters=32, kernel_size=(3, 3), learning_rate=learning_rate),
            Relu(),
            MaxPool2(),
            Flatten(),
            Linear(20000, 128, random_init, learning_rate),
            Relu(),
            Linear(128, 2, random_init, learning_rate),
            SoftMaxCrossEntropy()
        ]
        print('MNIST CNN initialized!')
    
    def forward(self, image, label):
        '''
        Completes a forward pass of the CNN and calculates the loss and prediction
        - image is a 2d numpy array
        - label is a digit
        '''
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        # Forward pass through each layer
        out = image / 255 - 0.5  # Normalize input
        for layer in self.layers[:-1]:  # Exclude last layer (SoftMaxCrossEntropy)
            out = layer.forward(out)
            print("layer: ", out.shape)
        y_hat, loss = self.layers[-1].forward_batch(out, label)  # Last layer, softmax
        return y_hat, loss

    def backprop(self, label, label_hat):
        '''
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        '''
        # Backpropagation through each layer in reverse order
        gradient = self.layers[-1].backprop_batch(label, label_hat)  # Start with last layer
        print("gradient: ", np.linalg.norm(gradient))
        if np.linalg.norm(gradient) < 1e-10:
            return True
        for layer in reversed(self.layers[:-1]):
            gradient = layer.backprop(gradient) 
        return False
    
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int, batch_num: int) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        train_loss_list = []
        test_loss_list = []
        for e in range (n_epochs):
            print('--- Epoch %d ---' % (e))
            index = np.random.choice(len(X_tr), batch_num, replace=False) 
            X_s, y_s = X_tr[index], y_tr[index]
            y_hat, loss = self.forward(X_s, y_s)
            print("y_hat: ", np.argmax(y_hat, axis = 1), y_hat)
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
    
INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]
class Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        :param input_size: number of units in the input of the layer 
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate
        self.w = weight_init_fn((output_size, input_size + 1))
        self.w[:, 0] = 0
        self.dw = np.zeros((output_size, input_size + 1))
        self.input = np.zeros(input_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (batch_num, input_size)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column 
                  and you will need to add it in yourself in this method.
                  Since we train on 1 example at a time, batch_size should be 1
                  at training.
        :return: output z of linear layer with shape (batch_num, output_size)
        """
        # Insert bias term
        x = np.insert(x, 0, 1, axis=1)
        self.input = x
        return np.dot(x, self.w.T)
       
    def backprop(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z of linear
                shape: (batch_size, num_outputs)
        :return: dx, partial derivative of loss with respect to input x of linear
                shape: (batch_size, num_inputs)

        Note that this function should set self.dw (gradient of weights with respect to loss)
        but not directly modify self.w; NN.step() is responsible for updating the weights.
        """
        # Assuming self.input shape: (batch_size, num_inputs)
        # Compute gradient w.r.t. weights (self.dw)
        self.dw = np.dot(dz.T, self.input) / dz.shape[0]  # Averaging over the batch
        self.w = self.w - self.lr * self.dw
        # Compute gradient w.r.t. inputs (dx)
        dx = np.dot(dz, self.w[:, 1:])

        return dx