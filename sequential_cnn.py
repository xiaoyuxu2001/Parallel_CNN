import mnist
import numpy as np
from conv import Conv2d
from maxpool import MaxPool2
from dense import Softmax, Relu, Flatten, Dense
import sequential_cnn
import argparse




def main():
  # We only use the first 1k examples of each set in the interest of time.
  # Feel free to change this if you want.
  train_images = mnist.train_images()[:1000]
  train_labels = mnist.train_labels()[:1000]
  test_images = mnist.test_images()[:1000]
  test_labels = mnist.test_labels()[:1000]

  cnn_model = sequential_cnn.CNN()

  # Train the CNN for 3 epochs
  for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
      if i % 100 == 99:
        print(
          '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
          (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

      l, acc = train(im, label, cnn_model)
      loss += l
      num_correct += acc

  # Test the CNN
  print('\n--- Testing the CNN ---')
  loss = 0
  num_correct = 0
  for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

  num_tests = len(test_images)
  print('Test Loss:', loss / num_tests)
  print('Test Accuracy:', num_correct / num_tests)

def forward(image, label, model):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  out, loss, acc, out_shape= model.forward((image / 255), label)

  return out, loss, acc, out_shape

def train(im, label, model, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc, out_shape= forward(im, label)

  # Calculate initial gradient
  model.backprop(im, label, out, loss, acc, lr)

  return loss, acc




if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')