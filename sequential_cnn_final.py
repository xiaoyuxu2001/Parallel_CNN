from cnn_model import CNN 
import mnist
import argparse

def main(args):
      # We only use the first 1k examples of each set in the interest of time.
  # Feel free to change this if you want.
  train_images = mnist.train_images()[:1000]
  train_labels = mnist.train_labels()[:1000]
  test_images = mnist.test_images()[:1000]
  test_labels = mnist.test_labels()[:1000]
  
  labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
    
  cnn = CNN(
      learning_rate=args.learning_rate,
  )
  train_losses, test_losses = cnn.train(train_images, train_labels, test_images, test_labels, epochs=3)
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
          cur_te_loss = test_losses[i]
          f.write("epoch={} crossentropy(train): {}\n".format(
              cur_epoch, cur_tr_loss))
          f.write("epoch={} crossentropy(validation): {}\n".format(
              cur_epoch, cur_te_loss))
      f.write("error(train): {}\n".format(train_error_rate))
      f.write("error(validation): {}\n".format(test_error_rate))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('train_input', type=str, default='data/mnist_train.csv',
                      help='path to training input .csv file')
  parser.add_argument('validation_input', type=str, default='data/mnist_test.csv',
                      help='path to validation input .csv file')
  parser.add_argument('train_out', type=str, default='train_out.txt',
                      help='path to store prediction on training data')
  parser.add_argument('validation_out', type=str, default='test_out.txt',
                      help='path to store prediction on validation data')
  parser.add_argument('metrics_out', type=str, default='metrics_out.txt',
                      help='path to store training and testing metrics')
  parser.add_argument('num_epoch', type=int, default=10,
                      help='number of training epochs')
  # parser.add_argument('hidden_units', type=int,
  #                     help='number of hidden units')
  # parser.add_argument('init_flag', type=int, choices=[1, 2],
  #                     help='weight initialization functions, 1: random')
  parser.add_argument('learning_rate', type=float, default=0.005,
                      help='learning rate')
  args = parser.parse_args()
  main(args)
