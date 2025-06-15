import numpy as np

# --- 1. Define the Activation Functions ---
"""
Role of Activation Functions:
Activation functions are used in `single_neuron_forward_pass` for the following reasons:

1.  **Introduce Non-linearity**: The primary role of an activation function is to introduce non-linearity into the neuron's output. If neurons only performed a weighted sum of inputs (a linear operation), then a multi-layer network would still behave like a single-layer linear model. Non-linearity allows neural networks to learn complex patterns and relationships in data that linear models cannot capture. In the script, `z = np.dot(inputs, weights) + bias` calculates a linear combination. The subsequent application of `sigmoid(z)`, `relu(z)`, or `tanh(z)` transforms this linear output non-linearly.

2.  **Control Output Range**: Activation functions often map the input (the weighted sum `z`) to a specific output range.
    *   **Sigmoid**: As seen in `def sigmoid(z): return 1 / (1 + np.exp(-z))`, it squashes the output to a range between 0 and 1. This is useful for binary classification tasks where the output can be interpreted as a probability.
    *   **Tanh**: The `tanh(z)` function (hyperbolic tangent) squashes the output to a range between -1 and 1.
    *   **ReLU**: The `relu(z): return np.maximum(0, z)` function outputs the input directly if it is positive, and 0 otherwise. This means its output range is \[0, ∞). ReLU is widely used because it helps mitigate the vanishing gradient problem and is computationally efficient.

3.  **Mimic Biological Neurons (Historically)**: The concept of activation functions was partly inspired by how biological neurons "fire" or activate only when a certain threshold of input stimulus is reached. While modern artificial neurons and activation functions have evolved beyond direct biological mimicry, this initial inspiration helped shape their development.

In the `single_neuron_forward` function, after calculating the weighted sum `z`, the chosen activation function (`sigmoid`, `relu`, or `tanh`) is applied to `z` to produce the final `activated_output`. This `activated_output` is then what would be passed to subsequent neurons in a larger network or used as the final output if it's an output layer neuron.
"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

# --- 2. Define the Neuron's Forward Pass Function ---

def single_neuron_forward(inputs, weights, bias, activation_function_name="relu"):
    """
    Simulates the forward pass of a single artificial neuron.

    Args:
        inputs (np.array): A 1D NumPy array representing the input features.
        weights (np.array): A 1D NumPy array representing the weights for each input.
        bias (float): The bias term.
        activation_function_name (str): Name of the activation function to use ('sigmoid', 'relu', 'tanh').

    Returns:
        float: The activated output of the neuron.
    """
    if inputs.shape != weights.shape:
        raise ValueError("Input and weights arrays must have the same shape for dot product.")

    # Step 1: Calculate the weighted sum (linear combination)
    # This is the dot product of inputs and weights, plus the bias.
    z = np.dot(inputs, weights) + bias
    print(f"  Weighted sum (z): {z:.4f}")

    # Step 2: Apply the activation function
    if activation_function_name == "sigmoid":
        activated_output = sigmoid(z)
    elif activation_function_name == "relu":
        activated_output = relu(z)
    elif activation_function_name == "tanh":
        activated_output = tanh(z)
    else:
        raise ValueError("Unsupported activation function. Choose 'sigmoid', 'relu', or 'tanh'.")

    return activated_output

# --- 3. Example Usage ---

print("--- Neuron Example 1: Basic ReLU Activation ---")
inputs_1 = np.array([0.5, 1.2, -0.3]) # Example inputs
weights_1 = np.array([0.8, -0.5, 0.2]) # Corresponding weights
bias_1 = 0.1 # Bias term

output_1 = single_neuron_forward(inputs_1, weights_1, bias_1, activation_function_name="relu")
print(f"Output with ReLU: {output_1:.4f}\n")


print("--- Neuron Example 2: Sigmoid Activation (often used for probabilities) ---")
inputs_2 = np.array([2.0, -1.0])
weights_2 = np.array([0.6, 0.4])
bias_2 = -0.5

output_2 = single_neuron_forward(inputs_2, weights_2, bias_2, activation_function_name="sigmoid")
print(f"Output with Sigmoid: {output_2:.4f}\n")


print("--- Neuron Example 3: Tanh Activation ---")
inputs_3 = np.array([0.1, 0.9, -0.7, 0.4])
weights_3 = np.array([1.0, -0.2, 0.5, -0.3])
bias_3 = 0.0 # No bias

output_3 = single_neuron_forward(inputs_3, weights_3, bias_3, activation_function_name="tanh")
print(f"Output with Tanh: {output_3:.4f}\n")


print("--- Neuron Example 4: When ReLU 'fires' (positive z) ---")
inputs_4 = np.array([1.0, 1.0])
weights_4 = np.array([1.0, 1.0])
bias_4 = -1.0 # Try to make z positive
output_4 = single_neuron_forward(inputs_4, weights_4, bias_4, activation_function_name="relu")
print(f"Output with ReLU (positive z): {output_4:.4f}\n")


print("--- Neuron Example 5: When ReLU 'is off' (negative z) ---")
inputs_5 = np.array([1.0, 1.0])
weights_5 = np.array([1.0, 1.0])
bias_5 = -3.0 # Try to make z negative
output_5 = single_neuron_forward(inputs_5, weights_5, bias_5, activation_function_name="relu")
print(f"Output with ReLU (negative z): {output_5:.4f}\n")