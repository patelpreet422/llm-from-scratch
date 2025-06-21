import numpy as np

def relu(z):
    """ReLU activation function: max(0, z)"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, 0 otherwise"""
    return (z > 0).astype(float)

def softmax(z):
    """
    Softmax activation for output layer in multi-class classification.
    Converts raw scores (logits) into probabilities that sum to 1.
    
    NOTE: For simplicity and clarity during learning, we're using a direct
    implementation. In production-grade code, a numerical stability trick
    (e.g., subtracting max(z) before exp) is typically used to prevent
    overflow/underflow with very large or very small input values.
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# Softmax derivative is typically handled in combination with cross-entropy loss
# which simplifies to (predictions - true_labels) as seen in backward pass.