import numpy as np

# --- 1. Define Activation Functions ---
def relu(z):
    """
    Iterate over all the elements in the input array and apply ReLU activation.
    ReLU activation function: max(0, i) for each element i in z.
    """
    return np.maximum(0, z)  # ReLU activation: max(0, z) for each element in z

def softmax(z):
    """
    Softmax activation for output layer in multi-class classification.
    Converts raw scores (logits) into probabilities that sum to 1.
    """
    exp_z = np.exp(z)

    # ss is the sum of exponentials across the last axis (axis=-1)
    ss = np.sum(exp_z, axis=-1, keepdims=True)
    print(f"Softmax sum (should be 1 for each sample): {exp_z, ss}")

    return exp_z / ss

# --- 2. Neural Network Class Definition ---
class SimpleNeuralNetwork:
    
    NETWORK_STRUCTURE = """
Network Structure:
    Input Layer: Batch of input samples (aka batch of features for which prediction is to be done).
    Shape = (batch_size, feature_size)
        [
            [f1, f2, f3, ...],             features for one sample (1D array with shape (input_size,)).
            [b11, b12, b13, ...],          features for another sample (1D array with shape (input_size,)).
            ...
        ]

    Feature weight(W_h): Conceptually a single neuron.
    Shape = (feature_size, 1)
        [
            [f1_weight],
            [f2_weight],
            ...
        ]
    
    Bias:
    Node bias: Single bias value for the neuron.
    Shape = (1, 1)
        [
            [b_h]                           (1D array with shape (1, 1))
        ]

    Output bias:
    Shape = (1, output_size)
        [
            [b_o1, b_o2, b_o3, ...]         (1D array with shape (output_size,))
        ]


    Output weight(W_o): Conceptually a single neuron.
    Shape = (1, output_size)
        [
            [o1_weight],
            [o2_weight],
            ...
        ]

    Output Layer: Produces probabilities for each class. 
    Shape is (batch_size, output_size)
        [
            [p1, p2, p3, ...],              (1D array with shape (output_size,)) is one sample's output/prediction.
            [p11, p12, p13, ...],           (1D array with shape (output_size,)) is another sample's output/prediction.
            ...
        ]

"""

    def __init__(self, feature_size, output_size):
        """
        Initializes the weights and biases of the neural network.
        The hidden layer is simplified to a single conceptual neuron (hidden_size=1).
        Weights are initialized randomly, biases to zeros.
        """

        print(self.NETWORK_STRUCTURE)

        self.feature_size = feature_size
        self.output_size = output_size
        self.randGen = np.random.default_rng()

        # Weights for "Single Neuron" Layer (conceptually the hidden layer): (input_size x 1)
        # For small networks, small random values are fine
        # Feature weights: (feature_size x 1)
        self.W_h = self.randGen.uniform(0, 1, size=(feature_size, 1))  # Initialize weights with small random values

        # Bias for "Single Neuron" Layer: (1 x 1)
        self.b_h = np.array([[0.001]]) # Bias for "Single Neuron" layer (1 row, 1 column)

        # Weights for Output Layer: (1 x output_size)
        self.W_o = self.randGen.uniform(size=(1, output_size))
        self.b_o = np.zeros((1, output_size)) # Bias for output layer

        print("Network initialized (hidden layer simplified to 1 neuron):")
        print(f"  Feature size: {feature_size}")
        print(f"  Output size: {output_size}")
        print(f"  Shape of W_h (Input to Single Neuron): {self.W_h.shape}")
        print(f"  Shape of b_h (Single Neuron Bias): {self.b_h.shape}")
        print(f"  Shape of W_o (Single Neuron to Output): {self.W_o.shape}")
        print(f"  Shape of b_o (Output Bias): {self.b_o.shape}\n")

    def forward(self, batch):
        """
        Performs the forward pass through the neural network for a single input sample.
        
        Args:
            batch (np.array): Expected shape: (batch_size, feature_size).

        Returns:
            np.array: The output of the network (e.g., probabilities for classes).
                      Shape: (batch_size, output_size)
        """
        
        if batch.ndim != 2:
            raise ValueError(f"Input must be a 2D array. Got shape: {batch.shape}, if you are passing a single sample, please reshape it to (1, {batch.size}).")
        
        print(f"Forward Pass with input shape: {batch.shape}")
        print(f"Inputs (X):\n{batch}\n")

        # --- Single Neuron Layer Calculation (Simplified Hidden Layer) ---
        print(f"--- Single Neuron Layer ---")
        print(f"Weights W_h (Input to Single Neuron) shape: {self.W_h.shape}:\n{self.W_h}\n")
        print(f"Bias b_h (Single Neuron Bias) shape: {self.b_h.shape}:\n{self.b_h}\n")

        # Z_h = Inputs @ W_h + b_h
        # (batch_size, feature_size) @ (feature_size, 1) -> (batch_size, 1)
        Z_h = np.dot(batch, self.W_h) + self.b_h
        print(f"Z_h (Inputs @ W_h + b_h) shape: {Z_h.shape} (pre-activation single neuron):\n{Z_h}\n")

        A_h = relu(Z_h) # Apply ReLU activation to single neuron layer
        print(f"A_h (relu(Z_h)) shape: {A_h.shape} (activated single neuron):\n{A_h}\n")

        # --- Output Layer Calculation ---
        print(f"--- Output Layer ---")
        print(f"Weights W_o (Single Neuron to Output) shape: {self.W_o.shape}:\n{self.W_o}\n")
        print(f"Bias b_o (Output Bias) shape: {self.b_o.shape}:\n{self.b_o}\n")

        # Z_o = A_h @ W_o + b_o
        # (batch_size, 1) @ (1, output_size) -> (batch_size, output_size)
        # Broadcasting adds b_o (1, output_size)
        Z_o = np.dot(A_h, self.W_o) + self.b_o
        print(f"Z_o (A_h @ W_o + b_o) shape: {Z_o.shape} (activation output):\n{Z_o}\n")

        A_o = softmax(Z_o) # Apply Softmax activation to output layer (for probabilities)
        print(f"A_o (softmax(Z_o)) shape: {A_o.shape} (activated output - probabilities):\n{A_o}\n")

        # Output should be 1D (output_size,)
        return A_o.squeeze() # Removes dimensions of size 1

# --- 3. Example Usage of the Neural Network ---

# Define network architecture
input_dim = 4    # e.g., 4 features for a data point
output_dim = 3   # e.g., 3 possible output classes (like "cat", "dog", "bird")

# Create the network
my_nn = SimpleNeuralNetwork(input_dim, output_dim) # hidden_dim removed

# --- Test with a single input sample ---
print("\n--- Testing with a single input sample ---")
# Example input representing one data point with 4 features
single_input = np.array([[0.1, 0.5, 0.2, 0.8]])
output_single = my_nn.forward(single_input)
print(f"Output for single input: {output_single}")
print(f"Sum of probabilities (should be close to 1): {np.sum(output_single):.4f}\n")

# Batch input testing has been removed as the network now only handles single samples.
# --- Test with a batch of input samples ---
print("\n--- Testing with a batch of input samples ---")
# Example input representing 2 data points, each with 4 features
batch_inputs = np.array([
    [0.1, 0.5, 0.2, 0.8], # Sample 1
    [0.9, 0.3, 0.7, 0.1]  # Sample 2
]) # Shape: (2, 4)

output_batch = my_nn.forward(batch_inputs)
print(f"Output for batch input:\n{output_batch}")
print(f"Sum of probabilities for each sample in batch:\n{np.sum(output_batch, axis=1).round(4)}")
