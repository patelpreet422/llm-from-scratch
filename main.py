# my_neural_net/main.py
import numpy as np
from tinynn.components.model import NeuralNetwork
from tinynn.components.layers import DenseLayer
from tinynn.components.activations import relu, softmax # Import specific activations for layer creation

# --- 1. Prepare Our Training Data ---
class_names = ["France", "India", "USA"]

france_coords = np.array([
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

X_train_raw = np.vstack((france_coords, india_coords, usa_coords)).astype(np.float32)

y_france = np.tile([1, 0, 0], (len(france_coords), 1))
y_india = np.tile([0, 1, 0], (len(india_coords), 1))
y_usa = np.tile([0, 0, 1], (len(usa_coords), 1))
y_train = np.vstack((y_france, y_india, y_usa)).astype(np.float32)

print("--- Training Data Prepared ---")
print(f"Total training samples (X_train_raw) shape: {X_train_raw.shape}")
print(f"Total training labels (y_train) shape: {y_train.shape}\n")
print(f"X_train_raw (Raw Coordinates):\n{X_train_raw}\n")
print(f"y_train (One-Hot Encoded Labels):\n{y_train}\n")


# --- 2. Instantiate Our Neural Network with two hidden layers ---
input_size = 2
hidden_1_size = 16 
hidden_2_size = 32
output_size = 3

model = NeuralNetwork()
# Add layers to the model
model.add_layer(DenseLayer(input_dim=input_size, output_dim=hidden_1_size, activation=relu))
model.add_layer(DenseLayer(input_dim=hidden_1_size, output_dim=hidden_2_size, activation=relu))
model.add_layer(DenseLayer(input_dim=hidden_2_size, output_dim=output_size, activation=softmax))

print("--- Network Architecture Initialized ---")
print(f"  Network contains {len(model.layers)} layers.")
for i, layer in enumerate(model.layers):
    print(f"  Layer {i+1}: Input_dim={layer.input_dim}, Output_dim={layer.output_dim}, Activation={layer.activation.__name__ if layer.activation else 'None'}")
print("\n")

# --- 3. Train the model ---
epochs = 10000 
learning_rate = 0.001 

model.train(X_train_raw, y_train, epochs, learning_rate)


# --- 4. Evaluate the Trained Model ---
print("\n--- Evaluating Trained Model ---")
final_predictions = model.forward(X_train_raw)
predicted_classes = np.argmax(final_predictions, axis=1)
true_classes = np.argmax(y_train, axis=1)

accuracy = np.mean(predicted_classes == true_classes)
print(f"Training Accuracy: {accuracy * 100:.2f}%\n")

print(f"{'Input (Raw Lat, Long)':<25} | {'Actual Label':<12} | {'Predicted Probs':<25} | {'Predicted Label'}")
print("-" * 88)
for i in range(len(X_train_raw)):
    input_coords_raw = X_train_raw[i]
    actual_label_name = class_names[true_classes[i]]
    predicted_probs = final_predictions[i]
    predicted_label_name = class_names[predicted_classes[i]]

    status = "CORRECT" if predicted_classes[i] == true_classes[i] else "INCORRECT"
    print(f"{str(input_coords_raw):<25} | {actual_label_name:<12} | {str(predicted_probs.round(3)):<25} | {predicted_label_name} ({status})")
print("-" * 88)

print("\n[INFO]: For more stable and efficient training in real-world scenarios,")
print("        data normalization (scaling features to a common range) is generally highly recommended.")
print("        However, for this small example, we chose to skip it for now to focus on core concepts.")

# --- 5. Human Evaluation Loop ---
print("\n--- Human Evaluation ---")
print("Enter latitude and longitude to get a prediction. Type 'exit' to quit.")
while True:
    user_input = input("Enter lat,long: ")
    if user_input.lower() == 'exit':
        break
    try:
        lat_str, lon_str = user_input.split(',')
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        
        # Prepare input for the model
        input_coords = np.array([[lat, lon]]).astype(np.float32)
        
        # Get prediction
        prediction_probs = model.forward(input_coords)
        predicted_class_index = np.argmax(prediction_probs, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        
        print(f"  -> Prediction: {predicted_class_name} (Probabilities: {prediction_probs.round(3)})")
        
    except (ValueError, IndexError):
        print("  Invalid input. Please enter coordinates in the format 'lat,long'.")
