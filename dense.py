import numpy as np
from typing import Tuple
def shuffle(X: np.ndarray, y: np.ndarray, 
            epoch: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    DO NOT modify this function.

    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]
def flatten(input_array):
    """
    Reshapes the input array into a one-dimensional array.
    """
    return np.array(input_array.reshape(-1))

def zero_init(shape : Tuple[int, int]) -> np.ndarray:
    """
    ZERO Initialization: All weights are initialized to 0.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape=shape)


def random_init(shape : Tuple[int, int]) -> np.ndarray:
    """

    RANDOM Initialization: The weights are initialized randomly from a uniform
        distribution from -0.1 to 0.1.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    M, D = shape
    np.random.seed(M * D)  # Don't change this line!

    W = np.random.uniform(-0.1, 0.1, shape)
    return W

def shuffle(X: np.ndarray, y: np.ndarray, 
            epoch: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    DO NOT modify this function.

    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        # input is a batch so flatten each image in the batch
        self.input_shape = input.shape
        flattened = np.empty((input.shape[0], np.prod(input.shape[1:])))
        for i in range(len(input)):
            flattened[i] = flatten(input[i])
        return flattened

    def backprop(self, d_L_d_out):
        # input_shape is the shape before we flattened it
        input_shape = self.input_shape
        ret = np.empty(input_shape)
        for i in range(len(d_L_d_out)):
            ret[i] = np.reshape(d_L_d_out[i], input_shape[1:])
        return ret

class Dense:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) * 0.01
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        self.last_input = input
        self.last_output = np.dot(input, self.weights) + self.biases
        return self.last_output

    def backprop(self, d_L_d_out, lr):
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)

        # Update weights and biases
        d_L_d_weights = np.dot(self.last_input.T, d_L_d_out)
        d_L_d_biases = d_L_d_out.mean(axis=0) * self.last_input.shape[0]

        self.weights -= lr * d_L_d_weights
        self.biases -= lr * d_L_d_biases

        return d_L_d_input

      
class Relu:
    def forward(self, input):
        """
        Performs a forward pass of the ReLU activation on the input where input is a batch
        """
        self.last_input = input
        return np.maximum(0, input)

    def backprop(self, d_L_d_out):
        """
        Performs a backward pass of the ReLU activation, where input is a batch.
        """
        d_L_d_input = d_L_d_out.copy()
        d_L_d_input = np.where(d_L_d_input > 0, d_L_d_input, 0)
        return d_L_d_input

    
class SoftMaxCrossEntropy:

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Implement softmax function.
        :param z: input logits of shape (num_classes,)
        :return: softmax output of shape (num_classes,)
        """
        # TODO: implement
        sum_exp = np.sum(np.exp(z))
        return np.exp(z) / sum_exp
    
    def _softmax_batch(self, z: np.ndarray) -> np.ndarray:
        """
        Implement softmax function.
        :param z: input logits of shape (num_batch, num_classes)
        :return: softmax output of shape (num_classes,)
        """

        sum_exp = np.sum(np.exp(z), axis=1)
        return np.divide(np.exp(z), np.array([sum_exp]).T)

    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        # TODO: implement
        # This is because we have 10 classes in total, if failed, that means my assumption is wrong
        # assert(len(y_hat) == 10)
        # assert(y >= 0 and y < 10)
        
        loss = -np.log(y_hat[y])
        
        return loss
    
    def _cross_entropy_batch(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        one_hot_y = np.zeros_like(y_hat)
        one_hot_y[np.arange(len(y_hat)), y] = 1
        loss = -np.sum(np.multiply(np.log(y_hat), one_hot_y), axis = 1)
        
        return loss

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarrayt
            loss: cross entropy loss
        """
        y_hat = self._softmax(z)
        cross_entropy = self._cross_entropy(y, y_hat)
        return y_hat, cross_entropy
    
    def forward_batch(self, z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_batch, num_classes)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarrayt
            loss: cross entropy loss, arrray of shape (num_batch,)
        """    
        y_hat_batch = self._softmax_batch(z)
        cross_entropy_batch = self._cross_entropy_batch(y, y_hat_batch)
        return y_hat_batch, cross_entropy_batch

    def backprop(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.
        Note that here instead of calculating the gradient w.r.t. the softmax
        output, we are directly computing gradient w.r.t. the softmax input.

        Try deriving the gradient yourself (Question 2.2(b) on the study guide),
        and you'll see why we want to calculate this in a single step.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        # TODO: implement using the formula you derived in the written
        res = y_hat.copy()
        print("y_hat: ", y_hat, y)
        res[y] -= 1
        return res
    
    def backprop_batch(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.
        Note that here instead of calculating the gradient w.r.t. the softmax
        output, we are directly computing gradient w.r.t. the softmax input.

        Try deriving the gradient yourself (Question 2.2(b) on the study guide),
        and you'll see why we want to calculate this in a single step.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """

        res = y_hat.copy()
        for i in range(len(y_hat)):
            res[i][y[i]] -= 1
        return res
