from cnn_model import CNN 
import mnist
import argparse
import time
import numpy as np

def main(args):
      # We only use the first 1k examples of each set in the interest of time.
    # Feel free to change this if you want.
  image_num = args.num_image
  

  train_0s = np.where(mnist.train_labels() == 0)[0] 
  test_0s = np.where(mnist.test_labels() == 0)[0]
  train_1s = np.where(mnist.train_labels() == 1)[0]
  test_1s = np.where(mnist.test_labels() == 1)[0]
  train_idxs = np.empty(image_num, dtype=int)
  train_idxs[:image_num // 2] = np.random.choice(train_0s, image_num // 2)
  train_idxs[image_num//2:] = np.random.choice(train_1s, image_num - image_num // 2)
  test_idxs = np.empty(image_num, dtype=int)
  test_idxs[:image_num // 2] = np.random.choice(test_0s, image_num // 2)
  test_idxs[image_num//2:] = np.random.choice(test_1s, image_num - image_num // 2)

  
  
#   train_idx = np.random.choice(len(mnist.train_images()), image_num)
  train_images = mnist.train_images()[train_idxs]
  train_labels = mnist.train_labels()[train_idxs]
  print(train_labels)
#   test_idx = np.random.choice(len(mnist.test_images()), image_num)
  test_images = mnist.test_images()[test_idxs]
  test_labels = mnist.test_labels()[test_idxs]


  labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
    
  cnn = CNN(
      learning_rate=args.learning_rate,
  )
#   calculate time for training
  start_time = time.time()
  train_losses, test_losses = cnn.train(train_images, train_labels, test_images, test_labels, n_epochs=args.num_epoch)
  end_time = time.time()
#   print("filter after training: ", cnn.layers[0].filters)
  print("Time for training: ", format(end_time - start_time, '.2f'), "s")
  train_labels, train_error_rate = cnn.test(train_images, train_labels)
  test_labels, test_error_rate = cnn.test(test_images, test_labels)
  with open(args.train_out, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
  with open(args.test_out, "w") as f:
      for label in test_labels:
          f.write(str(label) + "\n")
  with open(args.metrics_out, "w") as f:
      for i in range(len(train_losses)):
          cur_epoch = i + 1
          cur_tr_loss = train_losses[i]
          f.write("epoch={} crossentropy(train): {}\n".format(
              cur_epoch, cur_tr_loss))
      f.write("error(train): {}\n".format(train_error_rate))
      f.write("error(test): {}\n".format(test_error_rate))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_input', type=str, default='data/mnist_train.csv',
                      help='path to training input .csv file')
  parser.add_argument('--test_input', type=str, default='data/mnist_test.csv',
                      help='path to validation input .csv file')
  parser.add_argument('--train_out', type=str, default='train_out.txt',
                      help='path to store prediction on training data')
  parser.add_argument('--test_out', type=str, default='test_out.txt',
                      help='path to store prediction on validation data')
  parser.add_argument('--metrics_out', type=str, default='metrics_out.txt',
                      help='path to store training and testing metrics')
  parser.add_argument('--num_epoch', type=int, default=10,
                      help='number of training epochs')
  parser.add_argument('--num_image', type=int, default=50, help='number of images used')
  # parser.add_argument('hidden_units', type=int,
  #                     help='number of hidden units')
  # parser.add_argument('init_flag', type=int, choices=[1, 2],
  #                     help='weight initialization functions, 1: random')
  parser.add_argument('--learning_rate', type=float, default=0.05,
                      help='learning rate')
  parser.add_argument('--batch_num', type=int, default=128,
                      help='batch_num')
  args = parser.parse_args()
  main(args)
