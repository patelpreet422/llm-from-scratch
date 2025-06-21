import numpy as np
from .activations import relu, relu_derivative, softmax # Import from our activations.py

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation=None, random_seed=42):
        """
        Initializes a dense (fully connected) layer.

        Args:
            input_dim (int): Number of input features/neurons from the previous layer.
            output_dim (int): Number of neurons in this layer.
            activation (callable, optional): The activation function to use (e.g., relu, softmax).
                                            Defaults to None (linear activation).
            random_seed (int): Seed for weight initialization for reproducibility.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.randGen = np.random.default_rng(seed=random_seed)

        # Initialize weights and biases
        # Weights are initialized to small random values around zero
        self.weights = self.randGen.uniform(-1, 1, size=(input_dim, output_dim)) * 0.01
        self.biases = np.zeros((1, output_dim))

        # Caches for backward pass
        self.input_cache = None     # Input to this layer (X from previous, or original input)
        self.Z_cache = None         # Pre-activation values (X @ W + b)
        self.A_cache = None         # Post-activation values (output of this layer)

    def forward(self, input_data):
        """
        Performs the forward pass for this layer.
        
        Args:
            input_data (np.array): Input from the previous layer or original features.

        Returns:
            np.array: Output of this layer after activation.
        """
        self.input_cache = input_data
        
        self.Z_cache = np.dot(input_data, self.weights) + self.biases
        
        if self.activation:
            self.A_cache = self.activation(self.Z_cache)
        else:
            self.A_cache = self.Z_cache # Linear activation
            
        return self.A_cache

    def backward(self, d_A_next):
        """
        Performs the backward pass for this layer, computing gradients for weights,
        biases, and passing the gradient back to the previous layer.

        Args:
            d_A_next (np.array): Gradient of the loss with respect to the
                                 activation of this layer, propagated from the next layer.
                                 (dL/dA of this layer's output).

        Returns:
            np.array: Gradient of the loss with respect to the input of this layer (dL/dX_this_layer).
            tuple: Gradients for this layer's weights (dW) and biases (db).
        """
        N = self.input_cache.shape[0] # Number of samples in the batch

        # Calculate dZ (gradient of loss w.r.t. pre-activation Z)
        # This depends on the activation function
        if self.activation == relu:
            dZ = d_A_next * relu_derivative(self.Z_cache)
        elif self.activation == softmax:
            # Special case: softmax derivative is typically handled by the overall
            # (softmax + cross-entropy) simplification in the network's backward method,
            # so this layer should not typically have softmax activation itself.
            # However, if it did, the dZ calculation would be more complex.
            # For our model, softmax is only in the *last* layer of the model,
            # and its dZ is calculated by the model's backward method (predictions - true_labels).
            # So, this 'else' branch should ideally not be hit for the output layer.
            # For a general Layer class, one might pass dZ directly from the model's backward.
            dZ = d_A_next # If no activation or if d_A_next is already dZ
        else: # Linear activation
            dZ = d_A_next

        # Gradients for weights and biases of this layer
        dW = np.dot(self.input_cache.T, dZ) / N
        db = np.sum(dZ, axis=0, keepdims=True) / N

        # Gradient for the input of this layer (to pass to the previous layer)
        d_input = np.dot(dZ, self.weights.T)

        return d_input, (dW, db)