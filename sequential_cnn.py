import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

plt.rcParams["figure.figsize"] = (10,6)

sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set it None to display all rows in the dataframe
# pd.set_option('display.max_rows', None)

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print("There are ", len(X_train), "images in the training dataset")     
print("There are ", len(X_test), "images in the test dataset")  

## Normalize the X train and X test using max value of the image arrays.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Reshape the X into 4 dimension
X_train = X_train.reshape(X_train.shape[0],28, 28, 1) 
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
Y_train = to_categorical(y_train, 10) 
Y_test = to_categorical(y_test, 10)


# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))

def conv2d(input, filters, kernel_size):
    """
    Apply a 2D convolution operation to the input array.
    """
    kernel_height, kernel_width = kernel_size
    if (filter_height != kernel_height) or (filter_width != kernel_width):
        raise ValueError("Kernel size must match the dimensions of the filters")

    n_filters, filter_height, filter_width, n_channels = filters.shape
    n_rows, n_cols, _ = input.shape
    out_height = n_rows - filter_height + 1
    out_width = n_cols - filter_width + 1

    output = np.zeros((out_height, out_width, n_filters))

    for i in range(out_height):
        for j in range(out_width):
            for k in range(n_filters):
                output[i, j, k] = np.sum(input[i:i+filter_height, j:j+filter_width, :] * filters[k, :, :, :])

    return output

def maxpool2d(input, pool_size=2, stride=1):
    """
    Apply a 2D max pooling operation to the input array.
    """
    # Dimensions of the input feature map
    n_rows, n_cols, n_channels = input.shape
    # Output dimensions
    out_rows = ((n_rows - pool_size) // stride) + 1
    out_cols = ((n_cols - pool_size) // stride) + 1
    # Initialize the output
    pooled = np.zeros((out_rows, out_cols, n_channels))
    # Apply max pooling
    for y in range(0, n_rows, stride):
        for x in range(0, n_cols, stride):
            for channel in range(n_channels):
                window = input[y:y+pool_size, x:x+pool_size, channel]
                pooled[y // stride, x // stride, channel] = np.max(window)

    return pooled



def flatten(input_array):
    """
    Reshapes the input array into a one-dimensional array.
    """
    return input_array.reshape(-1)

class DenseLayer:
    def __init__(self, input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, size = (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(x):
        return np.exp(x)/sum(np.exp(x))
    

    def forward_pass(self, x, activation):
        self.z = np.matmul(x, self.weights) + self.biases
        if activation == "relu":
            self.a = self.relu(self.z)
        if activation == "softmax":
            self.a = self.softmax(self.z)
        return self.a
    
class SequentialModel:
    def __init__(self):
        self.layers = []
        self.activations = []

    def add(self, layer, activation=None):
        self.layers.append(layer)
        self.activations.append(activation)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            if isinstance(layer, DenseLayer):
                x = layer.forward_pass(x, activation)
            elif callable(layer):
                x = layer(x)
        return x


def categorical_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


## From model.compile in Keras
def rmsprop(lr, rho, epsilon):
    """
    Define a basic RMSprop optimizer function generator in numpy.

    Parameters:
    lr (float): Learning rate.
    rho (float): Decay rate.
    epsilon (float): Numerical stability term.

    Returns:
    function: A function to update the weights.
    """
    def update_weights(weights, gradients):
        """
        Update weights using RMSprop algorithm.

        Parameters:
        weights (list of np.array): Current weights of the model.
        gradients (list of np.array): Gradients of the loss w.r.t weights.

        Returns:
        list of np.array: Updated weights.
        """
        if not hasattr(update_weights, 's'):
            update_weights.s = [np.zeros_like(w) for w in weights]

        updated_weights = []
        for w, g, s in zip(weights, gradients, update_weights.s):
            s = rho * s + (1 - rho) * g**2
            update_weights.s.append(s)
            updated_weights.append(w - lr * g / (np.sqrt(s) + epsilon))

        return updated_weights

    return update_weights