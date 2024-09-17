"""
Multi-Layer Perceptron for binary classification from
Machin Learning with PyTorch and Scikit-learn by Sebastian Raschka et al.

Example:

model = NeuralNetworkMLP(num_features=28*28, num_hidden=50, num_classes=10)


"""
import numpy as np

##########################
### MODEL
##########################

def sigmoid(z):
    """
    Compute the sigmoid activation function.
    
    Args:
    z (numpy.ndarray): Input array
    
    Returns:
    numpy.ndarray: Sigmoid of the input
    """
    return 1.0 / (1.0 + np.exp(-z))

def int_to_onehot(y, num_labels):
    """
    Convert integer labels to one-hot encoded format.
    
    Args:
    y (numpy.ndarray): Array of integer labels
    num_labels (int): Number of unique labels
    
    Returns:
    numpy.ndarray: One-hot encoded labels
    """
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

class NeuralNetMLP:
    """
    Multi-layer Perceptron Neural Network implementation.
    """

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        """
        Initialize the Neural Network.
        
        Args:
        num_features (int): Number of input features
        num_hidden (int): Number of hidden units
        num_classes (int): Number of output classes
        random_seed (int): Seed for random number generation
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Initialize random number generator
        rng = np.random.RandomState(random_seed)
        
        # Initialize weights for hidden layer
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # Initialize weights for output layer
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
        
    def forward(self, x):
        """
        Perform forward pass through the network.
        
        Args:
        x (numpy.ndarray): Input data, shape (n_examples, n_features)
        
        Returns:
        tuple: (hidden layer activations, output layer activations)
        """
        # Hidden layer
        # Compute weighted sum: z_h = x * w_h.T + b_h
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        # Apply activation function
        a_h = sigmoid(z_h)

        # Output layer
        # Compute weighted sum: z_out = a_h * w_out.T + b_out
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        # Apply activation function
        a_out = sigmoid(z_out)
        
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        """
        Perform backward pass to compute gradients.
        
        Args:
        x (numpy.ndarray): Input data
        a_h (numpy.ndarray): Hidden layer activations
        a_out (numpy.ndarray): Output layer activations
        y (numpy.ndarray): True labels
        
        Returns:
        tuple: Gradients for weights and biases
        """
        # Convert labels to one-hot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # Compute gradients for output layer
        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]  # Derivative of MSE loss
        d_a_out__d_z_out = a_out * (1.0 - a_out)  # Derivative of sigmoid
        delta_out = d_loss__d_a_out * d_a_out__d_z_out # Combine derivatives using Chain Rule

        # Compute gradients for output weights and biases
        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # Compute gradients for hidden layer
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1.0 - a_h)  # Derivative of sigmoid
        d_z_h__d_w_h = x

        # Compute gradients for hidden weights and biases
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)
