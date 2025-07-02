from typing import List
import numpy as np
from .layers import DenseLayer
from .activations import softmax # Only softmax is needed for final output layer
from .losses import categorical_cross_entropy_loss

class NeuralNetwork:
    def __init__(self):
        """Initializes an empty list of layers."""
        self.layers: List[DenseLayer] = []
        self.randGen = np.random.default_rng(seed=42)

    def __str__(self):
        """Returns a string representation of the entire neural network."""
        # Start with a header for the network
        s = "Neural Network Architecture:\n"
        s += "="*30 + "\n"
        
        # If no layers, say so
        if not self.layers:
            s += "  No layers in this network.\n"
            return s
            
        # Iterate through each layer and add its string representation
        for i, layer in enumerate(self.layers):
            s += f"Layer {i+1}:\n{layer}"
            if i < len(self.layers) - 1:
                s += "\n\n"
        
        s += "\n" + "="*30
        return s

    def add_layer(self, layer: DenseLayer):
        """Adds a layer to the network."""
        self.layers.append(layer)

    def forward(self, X):
        """
        Performs a forward pass through all layers in the network.
        
        Args:
            X (np.array): Input data for the network.

        Returns:
            np.array: Final output probabilities of the network.
        """
        current_output = X
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, y_true, final_predictions):
        """
        Performs the backward pass through all layers, computing and returning gradients.
        
        Args:
            X (np.array): Original input to the network (needed for first layer's gradient).
            y_true (np.array): True one-hot encoded labels.
            final_predictions (np.array): Output probabilities from the forward pass.

        Returns:
            dict: A dictionary containing gradients for all weights and biases in the network.
                  Keys are tuples like ('W', layer_index) or ('b', layer_index).
        """
        gradients = {}
        
        # Start backpropagation from the output layer
        # The derivative of Cross-Entropy loss w.r.t. the pre-activation of the softmax output layer (dZ_o)
        # simplifies to (predictions - true_labels).
        # This is the 'd_A_next' for the last layer's backward pass, but effectively dZ directly.
        # This is dL/dZ_o
        # Note: final_predictions are the output probabilities after softmax activation.
        # y_true is the true one-hot encoded labels.
        # d_output is the gradient of the loss with respect to the output layer's pre-activation (Z_o) 
        # Z_o is 
        d_output = final_predictions - y_true # This is dL/dZ_o

        # Iterate backward through layers
        # The last layer is at index len(self.layers) - 1
        for i in reversed(range(len(self.layers))):
            current_layer = self.layers[i]
            
            d_input_prev_layer, (dW, db) = current_layer.backward(d_output)
            
            # Store gradients
            gradients[('W', i)] = dW
            gradients[('b', i)] = db
            
            # Update d_output for the next (earlier in forward pass) layer's backward pass
            d_output = d_input_prev_layer # This is dL/dA_prev_layer

        return gradients

    def update_parameters(self, gradients, learning_rate):
        """
        Updates the weights and biases of all layers using the computed gradients.
        """
        for i, layer in enumerate(self.layers):
            layer.weights -= learning_rate * gradients[('W', i)]
            layer.biases -= learning_rate * gradients[('b', i)]

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Trains the neural network using gradient descent.
        """
        print("--- Starting Training ---")
        print(f"  Epochs: {epochs}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Training Samples: {X_train.shape[0]}\n")

        # Basic check for at least one layer
        if not self.layers:
            raise ValueError("No layers added to the neural network. Please add layers using .add_layer().")

        for epoch in range(1, epochs + 1):
            # 1. Forward Pass
            predictions = self.forward(X_train)

            # 2. Compute Loss
            loss = categorical_cross_entropy_loss(predictions, y_train)

            # 3. Backward Pass (Compute Gradients)
            gradients = self.backward(y_train, predictions)

            # 4. Update Weights and Biases (Gradient Descent)
            self.update_parameters(gradients, learning_rate)

            # Print loss periodically
            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
        
        print("\n--- Training Complete ---")
