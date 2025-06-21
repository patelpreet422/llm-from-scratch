import numpy as np

def categorical_cross_entropy_loss(predictions, targets):
    """
    Calculates the Categorical Cross-Entropy Loss.
    L = - (1/N) * sum(sum(y_true * log(y_pred)))
    
    Args:
        predictions (np.array): Predicted probabilities from the network (output of softmax).
        targets (np.array): True one-hot encoded labels.

    Returns:
        float: The calculated loss value.
    """
    # Clip predictions to avoid log(0) which results in -infinity
    predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
    
    N = predictions.shape[0] # Number of samples in the batch
    loss = -np.sum(targets * np.log(predictions)) / N
    return loss

# Note: The derivative of Cross-Entropy w.r.t. Softmax output (dL/dZ_o)
# simplifies to (predictions - targets), which will be handled directly in
# the model's backward pass.