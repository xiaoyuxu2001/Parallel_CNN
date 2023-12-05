from conv import Conv2d
from maxpool import MaxPool2
from dense import  Relu, Flatten, Dense, random_init, SoftMaxCrossEntropy, shuffle
import numpy as np
from typing import Callable, List, Tuple
import logging

# todo: improve the model, right now it is very shabby
class CNN:
    def __init__(self, learning_rate) -> None:
        # config
        self.before_flat = (25, 25, 32)

        # Initialize layers
        self.conv = Conv2d(num_filters=32, kernel_size=(3, 3), learning_rate=learning_rate)
        self.relu1 = Relu()
        self.pool = MaxPool2()
        self.flat = Flatten()
        self.dense1 = Linear(20000, 128, random_init, learning_rate)
        self.relu2 = Relu()
        self.dense2 = Linear(128, 10, random_init, learning_rate)
        self.softmax = SoftMaxCrossEntropy()
        print('MNIST CNN initialized!')
    
    def forward(self, image, label):
        '''
        Completes a forward pass of the CNN and calculates the loss and prediction
        - image is a 2d numpy array
        - label is a digit
        '''
        
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        out = self.conv.forward((image / 255))
        logging.debug("after conv", out.shape)
        out = self.relu1.forward(out)
        logging.debug("after relu", out.shape)
        out = self.pool.forward(out)
        logging.debug("after pool", out.shape)
        self.before_flat = out.shape
        out = self.flat.forward(out)
        logging.debug("after flat", out.shape)
        out = self.dense1.forward(out)
        out = self.relu2.forward(out)
        out = self.dense2.forward(out)
        y_hat, loss = self.softmax.forward(out, label)

        return y_hat, loss
    
    def backprop(self, label, label_hat):
        '''
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        '''

        # Backprop
        gradient = self.softmax.backprop(label, label_hat)  
        gradient = self.dense2.backprop(gradient)
        gradient = self.relu2.backprop(gradient)
        gradient = self.dense1.backprop(gradient)
        gradient = self.flat.backprop(gradient, self.before_flat)
        gradient = self.pool.backprop(gradient)
        gradient = self.relu1.backprop(gradient)
        gradient = self.conv.backprop(gradient)
    
    def step(self):
        """
        Apply GD update to weights.
        """
        self.dense1.step()
        self.dense2.step()

        # conv layer has been udpated in the process of backprop

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        loss = 0
        for i in range(len(X)):
            y_hat_i, loss_i = self.forward(X[i], y[i])
            loss += loss_i
        return loss / len(X)
    
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int) -> Tuple[List[float], List[float]]:
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
            print('--- Epoch %d ---' % (e + 1))
            X_s, y_s = shuffle(X_tr, y_tr, e)
            for i, (im, label) in enumerate(zip(X_s, y_s)):
                y_hat, loss = self.forward(X_s[i], y_s[i])
                self.backprop(y_s[i], y_hat)
                self.step()
            train_loss = self.compute_loss(X_tr, y_tr)
            train_loss_list.append(train_loss)
            test_loss = self.compute_loss(X_test, y_test)
            test_loss_list.append(test_loss)
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
        y_predict_list = np.zeros(len(X))
        error = 0
        for i in range(len(X)):
            y_hat, loss = self.forward(X[i], y[i])
            y_predict = np.argmax(y_hat)
            y_predict_list[i] = y_predict
            if y_predict != y[i]:
                error += 1
        return y_predict_list, error / len(X)
    
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
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column 
                  and you will need to add it in yourself in this method.
                  Since we train on 1 example at a time, batch_size should be 1
                  at training.
        :return: output z of linear layer with shape (output_size,)
        """
        x = np.insert(x, 0, 1)
        self.input = x
        return np.dot(self.w, x)

    def backprop(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
        :return: dx, partial derivative of loss with respect to input x
            of linear
        
        Note that this function should set self.dw
            (gradient of weights with respect to loss)
            but not directly modify self.w; NN.step() is responsible for
            updating the weights.

        HINT: You may want to use some of the values you previously cached in 
        your forward() method.
        """
        a : int = len(dz)
        c : int = len(self.input)
        self.dw = np.dot(dz.reshape(a,1), (self.input.transpose()).reshape(1,c))
        return self.w[:, 1:].transpose() @ dz
        
    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been 
        set in NN.backward().
        """
        self.w = self.w - self.lr * self.dw
