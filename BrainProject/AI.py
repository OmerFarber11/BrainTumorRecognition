import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py as h5

"""# ***ITrainable***"""

class ITrainable():
  """
  Interface for trainable layers. Defines methods for saving, loading, updating parameters,
  and performing forward and backward propagation.
  """

  def save_parameters():
    """
    Save the parameters of the layer to a file.
    """
    raise NotImplementedError("save_parameters has not been created yet.")

  def load_parameters():
    """
    Load the parameters of the layer from a file.
    """
    raise NotImplementedError("load_parameters has not been created yet.")

  def update_parameters(self):
    """
    Update the parameters of the layer using optimization techniques.
    """
    raise NotImplementedError("update_parameters has not been created yet.")

  def forward_propagation(self, X):
    """
    Perform forward propagation through the layer.

    Args:
        X (np.ndarray): Input data.

    Returns:
        np.ndarray: Output after forward propagation.
    """
    raise NotImplementedError("forward_propagation has not been created yet.")

  def backward_propagation(self, dY_hat):
    """
    Perform backward propagation through the layer.

    Args:
        dY_hat (np.ndarray): Gradient of the loss with respect to the layer's output.

    Returns:
        np.ndarray: Gradient of the loss with respect to the layer's input.
    """
    raise NotImplementedError("backward_propagation has not been created yet.")

"""# ***DL Network***"""

class DLNetwork(ITrainable):
    def __init__(self, layers=[]):
        """
        Initializes the DLNetwork with layers consisting of linear and activation layers.

        Args:
            layers (list): List of layer objects, each providing linear and activation layers.
        """
        self.__layers = []
        for neurons_layer in layers:
            self.__layers.append(neurons_layer.get_linear_layer())
            self.__layers.append(neurons_layer.get_activation_layer())

    def __str__(self):
      s = f"Network: \n"
      for idx, layer in enumerate(self.layers, start=1):
          s += f"{idx}. {layer}\n"
      return s
    
    def print_weights(self):
      for i in self.__layers:
        i.linear_print_weights()
      
    def save_parameters(self, dir_path):
        """
        Saves the parameters of each layer to a specified directory.

        Args:
            dir_path (str): Directory path where parameters will be saved.
        """
        path = os.path.join(dir_path, "Parameters")
        os.makedirs(path, exist_ok=True) ## changed
        for layer in self.__layers:
            layer.save_parameters(path)

    def load_parameters(self, dir_path):
        """
        Loads parameters for each layer from a specified directory.

        Args:
            dir_path (str): Directory path from where parameters will be loaded.
        """
        path = os.path.join(dir_path, "Parameters")
        for layer in self.__layers:
            layer.load_parameters(path)
        self.print_weights()
    def update_parameters(self, optimization):
        """
        Updates parameters of each layer using a specified optimization method.

        Args:
            optimization (object): Optimization method for updating parameters.
        """
        for layer in self.__layers:
            layer.update_parameters(optimization)

    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.

        Args:
            X (np.ndarray): Input data or activations from the previous layer.

        Returns:
            np.ndarray: Output activations after forward propagation.
        """
        output = X
        for layer in self.__layers:
            output = layer.forward_propagation(output)
        return output

    def backward_propagation(self, dY_hat):
        """
        Performs backward propagation to compute gradients.

        Args:
            dY_hat (np.ndarray): Gradient of the loss with respect to the predicted output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input to the network.
        """
        grad = dY_hat
        for layer in reversed(self.__layers):
            grad = layer.backward_propagation(grad)
        return grad

    def add_layer(self, layer):
        """
        Adds a layer to the network.

        Args:
            layer (object): Layer object to be added.
        """
        self.__layers.append(layer)

"""# ***DL Neurons Layer***"""

class DLNeuronsLayer(DLNetwork):
  """
  Layer of neurons, consisting of a linear layer followed by an activation layer.
  """

  def __init__(self, input_size, output_size, activation, alpha, number, optimization=None):
    """
    Initialize the neuron layer with given parameters.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        alpha (float): Learning rate.
        layer_num (int): Layer number.
        activation (str): Activation function to use.
        optimization (str): Optimization technique to use.
    """

    self.__linear = DLLinearLayer(input_size, output_size, alpha, number, optimization)
    self.__activation = DLActivation(activation)

  def get_linear_layer(self):
    """
    Get the linear layer of the neuron layer.

    Returns:
        DLLinearLayer: Linear layer.
    """
    return self.__linear

  def get_activation_layer(self):
    """
    Get the activation layer of the neuron layer.

    Returns:
        DLActivationLayer: Activation layer.
    """
    return self.__activation

"""# ***DL Linear Layer***"""

class DLLinearLayer(ITrainable):
    """
    Linear layer for a deep learning network.
    Implements forward and backward propagation and parameter updates.
    """
    def __init__(self, input_size, output_size, alpha, number, optimization=None):
        """
        Initialize the linear layer with given parameters.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            alpha (float): Learning rate.
            number (int): Layer number.
            optimization (str): Optimization technique to use.
        """
        self.__input_size = input_size
        self.__output_size = output_size
        self.__alpha = alpha
        self.__number = number
        self.__W = self.__initialize_weights(self.__output_size, self.__input_size)
        self.__b = self.__initialize_bias(self.__output_size)
        self.__adaptive_W = np.full_like(self.__W, self.__alpha)
        self.__adaptive_b = np.full_like(self.__b, self.__alpha)
        self.__optimization = optimization
        self.adaptive_cont = 1.1
        self.adaptive_switch = 0.5

        self.__dW = None
        self.__dZ = None
        self.__prev_A = None

    def __str__(self):
        s = f"Dense Layer:\n"
        s += f"\tlearning_rate (alpha): {self.alpha}\n"
        s += f"\tnum inputs: {self.__input_size}\n"
        s += f"\tnum units: {self.__output_size}\n"
        if self.__optimization is not None:
            s += f"\tOptimization: {self.__optimization}\n"
        s += "\tParameters shape:\n"
        s += f"\t\tW shape: {self.__W.shape}\n"
        s += f"\t\tb shape: {self.__b.shape}\n"
        return s

    def __initialize_weights(self, rows, cols):
        """
        Initialize weights using He initialization.

        Args:
            rows (int): Number of rows (output features).
            cols (int): Number of columns (input features).

        Returns:
            np.ndarray: Initialized weights.
        """
        return np.random.randn(rows, cols) * np.sqrt(2 / cols)

    def linear_print_weights(self):
      print(self.__W)
      
    def __initialize_bias(self, size):
        """
        Initialize bias.

        Args:
            size (int): Size of the bias vector.

        Returns:
            np.ndarray: Initialized bias.
        """
        return np.random.randn(size, 1)

    def __linear_combination(self, W, A, b):
        """
        Perform linear combination of weights, input activations, and bias.

        Args:
            W (np.ndarray): Weights.
            A (np.ndarray): Input activations.
            b (np.ndarray): Bias.

        Returns:
            np.ndarray: Linear combination result.
        """
        return np.dot(W, A) + b

    def __adaptive_update(self):
        """
        Update parameters using adaptive learning rate.
        """
        # Update adaptive learning rates for weights
        self.adaptive_W *= np.where(self.adaptive_W * self.dW > 0, self.adaptive_cont, -self.adaptive_switch)
        self.W -= self.adaptive_W

        # Update adaptive learning rates for biases
        self.adaptive_b *= np.where(self.adaptive_b * self.db > 0, self.adaptive_cont, -self.adaptive_switch)
        self.b -= self.adaptive_b

    def __standard_update(self):
        """
        Update parameters using standard learning rate.
        """
        self.__W -= self.__alpha * self.__dW
        self.__b -= self.__alpha * self.__dZ

    def forward_propagation(self, prev_A):
        """
        Perform forward propagation through the linear layer.

        Args:
            prev_A (np.ndarray): Input activations from the previous layer.

        Returns:
            np.ndarray: Output activations after linear transformation.
        """
        self.__prev_A = prev_A
        return self.__linear_combination(self.__W, self.__prev_A, self.__b)

    def backward_propagation(self, dZ):
        """
        Perform backward propagation through the linear layer.

        Args:
            dZ (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input.
        """
        self.__dW = np.dot(dZ, self.__prev_A.T)
        input_gradient = np.dot(self.__W.T, dZ)
        self.__dZ = dZ
        return input_gradient

    def update_parameters(self, optimization=None):
        """
        Update the parameters of the linear layer using the specified optimization technique.

        Args:
            optimization (str): Optimization technique to use.
        """
        self.__optimization = optimization
        if self.__optimization == 'adaptive':
            self.__adaptive_update()
        else:
            self.__standard_update()

    def save_parameters(self, file_path):
        """
        Save the parameters of the layer to a file.

        Args:
            dir_path (str): Directory path to save the parameters.
        """

        file_name = os.path.join(file_path, f"TrainedParameters{self.__number}.h5")
        with h5.File(file_name, 'w') as hf:
            hf.create_dataset("W", data=self.__W)
            hf.create_dataset("b", data=self.__b)

    def load_parameters(self, file_path):
        """
        Load the parameters of the layer from a file.

        Args:
            dir_path (str): Directory path to load the parameters from.
        """
        file_name = os.path.join(file_path, f"TrainedParameters{self.__number}.h5")
        with h5.File(file_name, 'r') as hf:
            self.__W = np.array(hf['W'])
            self.__b = np.array(hf['b'])

"""# ***DL Activation Layer***"""

class DLActivation(ITrainable):
  """
  Activation layer for a deep learning network using ReLU activation function.
  Implements forward and backward propagation.
  """

  def __init__(self, activation):
    """
    Initialize the activation layer with the given activation function.

    Args:
        activation (str): Activation function to use.
    """

    self.leaky_relu_d = 0.01
    self.forward_propagation, self.backward_propagation = self.__select_activation_function(activation)

  def __select_activation_function(self, func_name):
    if func_name == 'sigmoid':
      return self.__sigmoid, self.__sigmoid_dZ
    elif func_name == 'tanh':
      return self.__tanh, self.__tanh_dZ
    elif func_name == 'relu':
      return self.__relu, self.__relu_dZ
    elif func_name == 'leaky_relu':
      return self.__leaky_relu, self.__leaky_relu_dZ
    elif func_name == 'softmax':
      return self.__softmax, self.__softmax_dZ
    else:
      raise ValueError(f"{func_name} is not a valid activation function\n")

  def __sigmoid(self,input):
    """
    Forward propagation using sigmoid activation.

    Args:
        input (np.ndarray): Input data.

    Returns:
        np.ndarray: Output after applying sigmoid activation.
    """

    self.input = input
    Sig = 1/(1+np.exp(-input))
    return Sig

  def linear_print_weights(self):
    print("activation")

  def __sigmoid_dZ(self,dA):
    """
    Backward propagation using sigmoid activation.

    Args:
        dA (np.ndarray): Gradient of the loss with respect to the layer's output.

    Returns:
        np.ndarray: Gradient of the loss with respect to the layer's input.
    """

    sig = self.__sigmoid(self.input)
    return np.multiply(dA,sig * (1 - sig))

  def __tanh(self, Z):
    """
    Forward propagation using tanh activation.

    Args:
        input (np.ndarray): Input data.

    Returns:
        np.ndarray: Output after applying tanh activation.
    """

    self.res = np.tanh(Z)
    return self.res

  def __tanh_dZ(self, dA):
    """
    Backward propagation using tanh activation.

    Args:
        dA (np.ndarray): Gradient of the loss with respect to the layer's output.

    Returns:
        np.ndarray: Gradient of the loss with respect to the layer's input.
    """
    return dA * (1 - self.res ** 2)

  def __relu(self, Z):
    """
    Forward propagation using ReLU activation.

    Args:
        Z (np.ndarray): Input data.

    Returns:
        np.ndarray: Output after applying ReLU activation.
    """

    self.Z = Z
    return np.maximum(0, Z)

  def __relu_dZ(self, dA):
    """
    Backward propagation using ReLU activation.

    Args:
        dA (np.ndarray): Gradient of the loss with respect to the layer's output.

    Returns:
        np.ndarray: Gradient of the loss with respect to the layer's input.
    """

    return np.where(self.Z <= 0, 0, 1) * dA

  def __leaky_relu(self, Z):
    """
    Forward propagation using leaky ReLU activation.

    Args:
        Z (np.ndarray): Input data.

    Returns:
        np.ndarray: Output after applying leaky ReLU activation.
    """

    self.Z = Z
    return np.where(self.Z <= 0, self.leaky_relu_d * self.Z, self.Z)

  def __leaky_relu_dZ(self, dA):
    """
    Backward propagation using leaky ReLU activation.

    Args:
        dA (np.ndarray): Gradient of the loss with respect to the layer's output.

    Returns:
        np.ndarray: Gradient of the loss with respect to the layer's input.
    """

    return np.where(self.Z <= 0, self.leaky_relu_d, 1) * dA

  def __softmax(self, Z):
    """
    Forward propagation using softmax activation.

    Args:
        Z (np.ndarray): Input data.

    Returns:
        np.ndarray: Output after applying softmax activation.
    """

    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

  def __softmax_dZ(self, dZ):
    """
    Backward propagation using softmax activation.

    Args:
        dA (np.ndarray): Gradient of the loss with respect to the layer's output.

    Returns:
        np.ndarray: Gradient of the loss with respect to the layer's input.
    """

    return dZ

  # not relevant in activatio layer
  def update_parameters(self,optimization):
    pass

  def save_parameters(self,file_path):
    pass

  def load_parameters(self,file_path):
    pass

"""# ***Loss Function***"""

class Loss:
  def __init__(self, loss):
    self.loss = loss


  def get_loss_functions(self):
    return self.__select_loss_functions()

  def __select_loss_functions(self):
    """
    Selects the appropriate loss functions based on the provided loss type.

    Args:
        loss (str): Type of loss function to select.

    Returns:
        tuple: Tuple of two functions, loss_forward and loss_backward.

    Raises:
        ValueError: If the provided loss type is not supported.
    """
    if self.loss == "square_distance":
        return self.square_dist, self.d_square_dist
    elif self.loss == "categorical_cross_entropy":
        return self.categorical_cross_entropy, self.d_categorical_cross_entropy
    elif self.loss == "cross_entropy":
        return self.cross_entropy, self.d_cross_entropy
    else:
        raise ValueError("The loss function you chose is not in the list. Please choose 'square_distance', 'cross_entropy', or 'categorical_cross_entropy'")


  def square_dist(self, Y_hat, Y):
    """
    Calculates the squared distance loss between predicted and true values.

    Args:
        Y_hat (np.ndarray): Predicted output from the model.
        Y (np.ndarray): True labels or target values.

    Returns:
        np.ndarray: Squared distance loss.
    """
    return (Y_hat - Y) ** 2


  def d_square_dist(self, Y_hat, Y):
    """
    Computes the gradient of squared distance loss.

    Args:
        Y_hat (np.ndarray): Predicted output from the model.
        Y (np.ndarray): True labels or target values.

    Returns:
        np.ndarray: Gradient of squared distance loss.
    """
    return 2 * (Y_hat - Y)


  def cross_entropy(self, Y_hat, Y):
    """
    Calculates the cross-entropy loss.

    Args:
        Y_hat (np.ndarray): Predicted output from the model.
        Y (np.ndarray): True labels or target values.

    Returns:
        np.ndarray: Cross-entropy loss.
    """
    Y_hat = np.clip(Y_hat, 1e-7, 1 - 1e-7)
    loss = - (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return np.mean(loss)


  def d_cross_entropy(self, Y_hat, Y):
    """
    Computes the gradient of cross-entropy loss.

    Args:
        Y_hat (np.ndarray): Predicted output from the model.
        Y (np.ndarray): True labels or target values.

    Returns:
        np.ndarray: Gradient of cross-entropy loss.
    """
    Y_hat = np.clip(Y_hat, 1e-7, 1 - 1e-7)
    return Y_hat - Y


  def categorical_cross_entropy(self, Y_hat, Y):
    """
    Calculates the categorical cross-entropy loss.

    Args:
        Y_hat (np.ndarray): Predicted output from the model.
        Y (np.ndarray): True labels or target values.

    Returns:
        np.ndarray: Categorical cross-entropy loss.
    """
    Y_hat = np.clip(Y_hat, 1e-7, 1 - 1e-7)
    loss = -np.sum(Y * np.log(Y_hat), axis=1)
    return np.mean(loss)


  def d_categorical_cross_entropy(self, Y_hat, Y):
    """
    Computes the gradient of categorical cross-entropy loss.

    Args:
        Y_hat (np.ndarray): Predicted output from the model.
        Y (np.ndarray): True labels or target values.

    Returns:
        np.ndarray: Gradient of categorical cross-entropy loss.
    """
    Y_hat = np.clip(Y_hat, 1e-7, 1 - 1e-7)
    return Y_hat - Y

"""# ***DL Model***"""

class DLModel:
    def __init__(self, network, loss, optimization=None):
      """
      Initialize the DLModel with a neural network, loss function, and optional optimization method.

      Args:
          network (object): Instance of a neural network class with forward and backward propagation methods.
          loss (str): Type of loss function to use. Supported: 'square_distence', 'categorical_cross_entropy', 'cross_entropy'.
          optimization (object, optional): Optimization method for updating network parameters. Default is None.
              (Note: Should be compatible with the network's update_parameters method.)

      Raises:
          ValueError: If an unsupported loss function is provided.
      """

      self.__network = network
      self.__optimization = optimization
      self.__loss_forward, self.__loss_backward = Loss(loss).get_loss_functions()

    def forward_propagation(self, prev_A):
      """
      Performs forward propagation through the neural network.

      Args:
          prev_A (np.ndarray): Input data or activations from the previous layer.

      Returns:
          np.ndarray: Output activations after forward propagation.
      """
      return self.__network.forward_propagation(prev_A)

    def backward_propagation(self, Y_hat, Y):
      """
      Performs backward propagation to compute gradients.

      Args:
          Y_hat (np.ndarray): Predicted output from the model.
          Y (np.ndarray): True labels or target values.

      Returns:
          np.ndarray: Gradients of the loss with respect to the predicted output.
      """
      dY_hat = self.__loss_backward(Y_hat, Y)
      return self.__network.backward_propagation(dY_hat)


    def train(self, x_train, y_train, num_iterations, verbose=True):
      """
      Trains the model using mini-batch gradient descent.

      Args:
          x_train (list): List of input data samples.
          y_train (list): List of true labels corresponding to the input data samples.
          num_iterations (int): Number of training iterations (epochs).
          verbose (bool, optional): If True, print training progress. Default is True.

      Returns:
          list: List of errors (losses) over the training iterations.
      """
      errors = []
      m = len(x_train)

      for i in range(num_iterations):
          total_cost = 0
          #looping on each image
          for j in range(m):
              X = x_train[j]
              Y = y_train[j]

              # Forward propagation
              Y_hat = self.forward_propagation(X)

              # Compute cost
              total_cost += self.__loss_forward(Y_hat, Y)

              # Backward propagation
              self.backward_propagation(Y_hat, Y)
              self.__network.update_parameters(self.__optimization)

          # Average cost over the training samples
          average_cost = total_cost / m
          errors.append(average_cost)

          if verbose:
              print(f"Error after {i+1} updates ({((i+1)*100)//num_iterations}%): {average_cost}") ##TODO: Change

      return errors