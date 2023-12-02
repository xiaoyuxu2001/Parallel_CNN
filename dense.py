import numpy as np

def flatten(input_array):
    """
    Reshapes the input array into a one-dimensional array.
    """
    return np.array(input_array.reshape(-1))

class Flatten:
    def forward(self, input):
        return input.flatten()

    def backprop(self, d_L_d_out, input_shape):
        return d_L_d_out.reshape(input_shape)

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
        Performs a forward pass of the ReLU activation on the input.
        """
        self.last_input = input
        return np.maximum(0, input)
    def d_relu(self, x):
          return 1 if x > 0 else 0

    def backprop(self, d_L_d_out):
        """
        Performs a backward pass of the ReLU activation.
        """
        d_L_d_input = d_L_d_out.copy()
        d_L_d_input = np.where(d_L_d_input > 0, 1, 0)
        return d_L_d_input

class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    limit = np.sqrt(6 / (input_len + nodes))
    self.weights = np.random.uniform(-limit, limit, size = (input_len, nodes))
    self.biases = np.zeros(nodes)
    
  def softmax(self, x):
    return np.exp(x)/sum(np.exp(x))

  def forward(self, input):
    '''
    Performs a forward pass of the softmax
    '''
    self.last_input_shape = input.shape

    input = flatten(input)
    self.last_input = input

    totals = np.dot(input, self.weights) + self.biases
    self.last_totals = totals
    return self.softmax(totals)

  def backprop(self, d_L_d_out, lr):
    '''
    Performs a backward pass of the softmax
    '''
    t_exp = np.exp(self.last_totals)
    S = np.sum(t_exp)
    d_t_d_w = self.last_input
    d_t_d_b = 1
    d_t_d_inputs = self.weights
    
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
      d_L_d_t = gradient * d_out_d_t

      d_L_d_w =np.matmul(d_t_d_w[np.newaxis].T, d_L_d_t[np.newaxis])
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = np.matmul(d_t_d_inputs, d_L_d_t)
      self.weights -= lr * d_L_d_w
      self.biases -= lr * d_L_d_b
      return d_L_d_inputs.reshape(self.last_input_shape)