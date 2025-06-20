import numpy as np

# --- 1. Define Activation Functions ---
def relu(z):
    """ReLU activation function: max(0, z)"""
    return np.maximum(0, z)

# Derivative of ReLU (used in backpropagation)
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

# --- 2. Neural Network Class Definition ---
class LocationPredictorNN:
    def __init__(self, feature_size, output_size):
        """
        Initializes the weights and biases of the neural network.
        Uses a simplified hidden layer with a single neuron.
        Weights are initialized to small random values around zero, biases to zeros.
        """
        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_size = 1 # Explicitly set to 1 for the simplified hidden layer
        self.randGen = np.random.default_rng(seed=42) # Added a seed for reproducibility!

        # Weights for Hidden Layer (W_h): (feature_size x 1)
        self.W_h = self.randGen.uniform(-1, 1, size=(self.feature_size, self.hidden_size)) * 0.01

        # Bias for Hidden Layer (b_h): (1 x 1)
        self.b_h = np.zeros((1, self.hidden_size))

        # Weights for Output Layer (W_o): (1 x output_size)
        self.W_o = self.randGen.uniform(-1, 1, size=(self.hidden_size, self.output_size)) * 0.01
        
        # Bias for Output Layer (b_o): (1 x output_size)
        self.b_o = np.zeros((1, self.output_size))

        # Caching variables for backward pass
        self.A_h_cache = None
        self.Z_h_cache = None # Needed for ReLU derivative
        self.Z_o_cache = None

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
        Stores intermediate values (cache) needed for the backward pass.
        """
        if batch.ndim != 2:
            raise ValueError(f"Input must be a 2D array. Got shape: {batch.shape}. "
                             f"If you are passing a single sample, reshape it to (1, {batch.shape[-1]}).")
        
        # Hidden Layer Calculation
        self.Z_h_cache = np.dot(batch, self.W_h) + self.b_h
        self.A_h_cache = relu(self.Z_h_cache)

        # Output Layer Calculation
        self.Z_o_cache = np.dot(self.A_h_cache, self.W_o) + self.b_o
        A_o = softmax(self.Z_o_cache)

        return A_o

    def compute_loss(self, predictions, targets):
        """
        Calculates the Categorical Cross-Entropy Loss.
        L = - (1/N) * sum(sum(y_true * log(y_pred)))
        """
        # Clip predictions to avoid log(0) which results in -infinity
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        
        # Sum only for the true class (since y_true is one-hot, others are 0)
        N = predictions.shape[0]
        loss = -np.sum(targets * np.log(predictions)) / N
        return loss

    def backward(self, X, y_true, A_o):
        """
        Performs backpropagation to calculate gradients for weights and biases.
        X: Input batch (same as used in forward pass for this prediction)
        y_true: True labels (one-hot encoded)
        A_o: Predicted probabilities (output of softmax)
        """
        N = X.shape[0] # Number of samples in the batch

        # --- Gradients for Output Layer ---
        # dL/dZ_o (gradient of loss w.r.t. Z_o)
        # For Softmax + Cross-Entropy, this simplifies to A_o - y_true
        dZ_o = A_o - y_true # Shape: (N, output_size)

        # dL/dW_o (gradient of loss w.r.t. W_o)
        dW_o = np.dot(self.A_h_cache.T, dZ_o) / N 

        # dL/db_o (gradient of loss w.r.t. b_o)
        db_o = np.sum(dZ_o, axis=0, keepdims=True) / N

        # --- Gradients for Hidden Layer ---
        # dL/dA_h (gradient of loss w.r.t. A_h)
        d_A_h = np.dot(dZ_o, self.W_o.T)

        # dL/dZ_h (gradient of loss w.r.t. Z_h)
        # Multiply by derivative of ReLU for element-wise gradient
        dZ_h = d_A_h * relu_derivative(self.Z_h_cache)

        # dL/dW_h (gradient of loss w.r.t. W_h)
        dW_h = np.dot(X.T, dZ_h) / N

        # dL/db_h (gradient of loss w.r.t. b_h)
        db_h = np.sum(dZ_h, axis=0, keepdims=True) / N

        return dW_h, db_h, dW_o, db_o

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Trains the neural network using gradient descent.
        """
        print("--- Starting Training ---")
        print(f"  Epochs: {epochs}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Training Samples: {X_train.shape[0]}\n")

        for epoch in range(1, epochs + 1):
            # 1. Forward Pass
            predictions = self.forward(X_train)

            # 2. Compute Loss
            loss = self.compute_loss(predictions, y_train)

            # 3. Backward Pass (Compute Gradients)
            dW_h, db_h, dW_o, db_o = self.backward(X_train, y_train, predictions)

            # 4. Update Weights and Biases (Gradient Descent)
            self.W_h -= learning_rate * dW_h
            self.b_h -= learning_rate * db_h
            self.W_o -= learning_rate * dW_o
            self.b_o -= learning_rate * db_o

            # Print loss periodically
            if epoch % 1000 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
        
        print("\n--- Training Complete ---")

# --- 3. Prepare Our Training Data ---
class_names = ["France", "India", "USA"]

def evaluate_model(model, X, y, stage=""):
    """Helper function to evaluate the model at a given stage."""
    print(f"\n--- Evaluating Model {stage} ---")
    predictions = model.forward(X)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)

    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    print(f"{'Input (Raw Lat, Long)':<25} | {'Actual Label':<12} | {'Predicted Probs':<25} | {'Predicted Label'}")
    print("-" * 88)
    for i in range(len(X)):
        input_coords_raw = X[i]
        actual_label_name = class_names[true_classes[i]]
        predicted_probs = predictions[i]
        predicted_label_name = class_names[predicted_classes[i]]

        status = "CORRECT" if predicted_classes[i] == true_classes[i] else "INCORRECT"
        print(f"{str(input_coords_raw):<25} | {actual_label_name:<12} | {str(predicted_probs.round(3)):<25} | {predicted_label_name} ({status})")
    print("-" * 88)

france_coords = np.array([[48.8566, 2.3522], [48.8600, 2.3000], [48.8400, 2.3800], [48.8700, 2.3400], [48.8500, 2.3600]])
india_coords = np.array([[20.5937, 78.9629], [22.0000, 78.0000], [18.0000, 80.0000], [24.0000, 77.0000], [19.0000, 79.0000]])
usa_coords = np.array([[39.8283, -98.5795], [38.0000, -100.0000], [40.0000, -96.0000], [37.0000, -99.0000], [41.0000, -97.0000]])

X_train_raw = np.vstack((france_coords, india_coords, usa_coords)).astype(np.float32)

y_france = np.tile([1, 0, 0], (len(france_coords), 1))
y_india = np.tile([0, 1, 0], (len(india_coords), 1))
y_usa = np.tile([0, 0, 1], (len(usa_coords), 1))
y_train = np.vstack((y_france, y_india, y_usa)).astype(np.float32)

print("--- Training Data Prepared ---")
print(f"Total training samples (X_train_raw) shape: {X_train_raw.shape}")
print(f"Total training labels (y_train) shape: {y_train.shape}\n")

# --- 4. Instantiate Our Neural Network ---
input_size = 2
output_size = 3
model = LocationPredictorNN(input_size, output_size)

# --- 5. Evaluate Before Training ---
evaluate_model(model, X_train_raw, y_train, stage="Before Training")

# --- 6. Train the model ---
epochs = 100000 # Number of times to iterate over the entire dataset
learning_rate = 0.001 # How big of a step to take during each weight update
model.train(X_train_raw, y_train, epochs, learning_rate)

# --- 7. Evaluate After Training ---
evaluate_model(model, X_train_raw, y_train, stage="After Training")

# --- 8. Human Evaluation Loop ---
print("\n--- Human Evaluation Loop ---")
print("Enter latitude and longitude to predict the country.")
print("Type 'quit' or 'exit' to stop.")

while True:
    user_input = input("\nEnter lat,long (e.g., 48.8,2.3): ")
    if user_input.lower() in ['quit', 'exit']:
        break

    try:
        # Parse and prepare the input
        lat_str, lon_str = user_input.split(',')
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        
        # Reshape for a single prediction
        input_coords = np.array([[lat, lon]], dtype=np.float32)

        # Make a prediction
        prediction = model.forward(input_coords)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_country = class_names[predicted_class_index]

        print(f"-> Prediction: {predicted_country}")
        print(f"   (Probabilities: {prediction.round(3)})")

    except (ValueError, IndexError):
        print("Invalid input. Please use the format 'latitude,longitude'.")

print("\n[INFO]: For more stable and efficient training in real-world scenarios,")
print("        data normalization (scaling features to a common range) is generally highly recommended.")
print("        However, for this small example, we chose to skip it for now to focus on core concepts.")
