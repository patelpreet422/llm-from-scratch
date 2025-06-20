import numpy as np

# --- 1. Define Activation Functions ---
def relu(z):
    """ReLU activation function: max(0, z)"""
    return np.maximum(0, z)

def softmax(z):
    """
    Softmax activation for output layer in multi-class classification.
    Converts raw scores (logits) into probabilities that sum to 1.
    Includes numerical stability trick to prevent overflow/underflow.
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# --- 2. Neural Network Class Definition ---
class LocationPredictorNN:
    def __init__(self, feature_size, output_size):
        """
        Initializes the weights and biases of the neural network.
        Uses a simplified hidden layer with a single neuron.
        Weights are initialized randomly, biases to zeros.
        """
        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_size = 1 # Explicitly set to 1 for the simplified hidden layer
        self.randGen = np.random.default_rng()

        # Weights for Hidden Layer (W_h): (feature_size x 1)
        self.W_h = self.randGen.uniform(0, 1, size=(self.feature_size, self.hidden_size))

        # Bias for Hidden Layer (b_h): (1 x 1)
        self.b_h = np.zeros((1, self.hidden_size))

        # Weights for Output Layer (W_o): (1 x output_size)
        self.W_o = self.randGen.uniform(0, 1, size=(self.hidden_size, self.output_size))
        
        # Bias for Output Layer (b_o): (1 x output_size)
        self.b_o = np.zeros((1, self.output_size))

        print("--- Network Architecture Initialized ---")
        print(f"  Input Features: {self.feature_size}")
        print(f"  Hidden Layer Neurons: {self.hidden_size}")
        print(f"  Output Classes: {self.output_size}")
        print(f"  W_h (Input->Hidden) shape: {self.W_h.shape}")
        print(f"  b_h (Hidden Bias) shape: {self.b_h.shape}")
        print(f"  W_o (Hidden->Output) shape: {self.W_o.shape}")
        print(f"  b_o (Output Bias) shape: {self.b_o.shape}\n")

    def forward(self, batch):
        """
        Performs the forward pass through the neural network for a batch of input samples.
        
        Args:
            batch (np.array): Input data, expected shape: (batch_size, feature_size).
                              Will raise ValueError if input is not 2D.

        Returns:
            np.array: Probabilities for each class, shape: (batch_size, output_size).
        """
        if batch.ndim != 2:
            raise ValueError(f"Input must be a 2D array. Got shape: {batch.shape}. "
                             f"If you are passing a single sample, reshape it to (1, {batch.shape[-1]}).")
        
        # Hidden Layer Calculation: Z_h = Inputs @ W_h + b_h
        Z_h = np.dot(batch, self.W_h) + self.b_h
        A_h = relu(Z_h) # Apply ReLU activation

        # Output Layer Calculation: Z_o = A_h @ W_o + b_o
        Z_o = np.dot(A_h, self.W_o) + self.b_o
        A_o = softmax(Z_o) # Apply Softmax activation for probabilities

        return A_o

# --- 3. Prepare Our Training Data ---

# Define our labels and their corresponding one-hot encodings
class_names = ["Paris", "India", "USA"]

# Latitude, Longitude pairs for training (approximate central points/cities)
paris_coords = np.array([
    [48.8566, 2.3522], [48.8600, 2.3000], [48.8400, 2.3800],
    [48.8700, 2.3400], [48.8500, 2.3600]
])
india_coords = np.array([
    [20.5937, 78.9629], [22.0000, 78.0000], [18.0000, 80.0000],
    [24.0000, 77.0000], [19.0000, 79.0000]
])
usa_coords = np.array([
    [39.8283, -98.5795], [38.0000, -100.0000], [40.0000, -96.0000],
    [37.0000, -99.0000], [41.0000, -97.0000]
])

# Combine all coordinates
X_train_raw = np.vstack((paris_coords, india_coords, usa_coords)).astype(np.float32)

# Create corresponding one-hot encoded labels
y_paris = np.tile([1, 0, 0], (len(paris_coords), 1))
y_india = np.tile([0, 1, 0], (len(india_coords), 1))
y_usa = np.tile([0, 0, 1], (len(usa_coords), 1))

y_train = np.vstack((y_paris, y_india, y_usa)).astype(np.float32)

print("--- Training Data Prepared ---")
print(f"Total training samples (X_train_raw) shape: {X_train_raw.shape}")
print(f"Total training labels (y_train) shape: {y_train.shape}\n")

print(f"X_train_raw (Raw Coordinates): {X_train_raw}")
print(f"y_train (One-Hot Encoded Labels): {y_train}\n")

# --- IMPORTANT: Data Normalization (Min-Max Scaling to [0, 1]) ---
# Store min/max for later use (e.g., when predicting new, unseen coordinates)
min_lat, max_lat = X_train_raw[:, 0].min(), X_train_raw[:, 0].max()
min_long, max_long = X_train_raw[:, 1].min(), X_train_raw[:, 1].max()

# --- 4. Instantiate Our Neural Network ---
input_size = 2 # Latitude, Longitude
output_size = 3 # Paris, India, USA

model = LocationPredictorNN(input_size, output_size)

# --- 5. Perform a "Pre-Training" Forward Pass and Observe Random Predictions ---
print("--- Making Predictions with Untrained Network on Normalized Data ---")
print("These predictions will be random because the network hasn't learned yet!\n")

# Predict for the entire normalized training batch
batch_predictions = model.forward(X_train_raw)

# Display some predictions alongside actual labels for observation
print("--- Sample Predictions (Untrained) ---")
print(f"{'Input (Raw Lat, Long)':<25} | {'Actual Label':<12} | {'Predicted Probs':<25} | {'Predicted Label'}")
print("-" * 88)
for i in range(len(X_train_raw)): # Show all samples
    input_coords_raw = X_train_raw[i] # Display raw for readability
    actual_label_idx = np.argmax(y_train[i])
    actual_label_name = class_names[actual_label_idx]
    
    predicted_probs = batch_predictions[i]
    predicted_label_idx = np.argmax(predicted_probs)
    predicted_label_name = class_names[predicted_label_idx]

    print(f"{str(input_coords_raw):<25} | {actual_label_name:<12} | {str(predicted_probs.round(3)):<25} | {predicted_label_name}")
print("-" * 88)
print("\nRemember: The network's weights are random, so its predictions are too!\n")
