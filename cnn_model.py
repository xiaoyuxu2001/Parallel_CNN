from conv import Conv2d
from maxpool import MaxPool2
from dense import Softmax, Relu, Flatten, Dense
import numpy as np

# todo: improve the model, right now it is very shabby
class CNN:
    def __init__(self) -> None:
        self.conv = Conv2d(num_filters=32, kernel_size=(3, 3))
        self.pool = MaxPool2()
        self.flat = Flatten()
        self.dense = Dense(128, 13 * 13 * 32)
        self.relu = Relu()
        self.softmax = Softmax(13 * 13 * 32, 10)
        print('MNIST CNN initialized!')
    
    def forward(self, image, label):
        '''
        Completes a forward pass of the CNN and calculates the accuracy and
        cross-entropy loss.
        - image is a 2d numpy array
        - label is a digit
        '''
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        out = self.conv.forward((image / 255))
        out = self.pool.forward(out) # (13, 13, 32)
        out_shape = out.shape
        out = self.softmax.forward(out) #(10,)

        # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc, out_shape
    
    def backprop(self, im, label, out, loss, acc, lr=.005):
        '''
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        '''
        # Forward
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        # Backprop
        gradient = self.softmax.backprop(gradient, lr)  #(13, 13, 32)
        # gradient = relu.backprop(gradient)
        # gradient = dense.backprop(gradient, lr)
        # gradient = flat.backprop(gradient, out_shape)
        gradient = self.pool.backprop(gradient) #(26, 26, 32)
        gradient = self.conv.backprop(gradient, lr)
        

        return loss, acc