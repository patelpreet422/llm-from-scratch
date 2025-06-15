import numpy as np
import matplotlib.pyplot as plt # For visualization, great for Mac!

# 1. Define our simple function (our "loss function")
def f(x):
    return x**2

# 2. Define its derivative (our "gradient")
def df_dx(x):
    return 2 * x # This is the analytical gradient for x^2

# --- Gradient Descent Parameters ---
learning_rate = 0.1 # How big of a step we take in the opposite direction of the gradient
num_iterations = 50 # How many steps we'll take

# 3. Starting point (our initial "weight" or "parameter")
x = 10.0 # Let's start far from the minimum at x=0

# Keep track of our progress for plotting
x_history = [x]
f_x_history = [f(x)]

print(f"Starting at x = {x}, f(x) = {f(x)}")

# --- 4. The Gradient Descent Loop ---
for i in range(num_iterations):
    # Calculate the gradient at the current x
    gradient = df_dx(x)

    # Update x: Move in the opposite direction of the gradient
    # The 'learning_rate' controls the step size
    x = x - learning_rate * gradient

    # Store history for visualization
    x_history.append(x)
    f_x_history.append(f(x))

    if (i + 1) % 10 == 0: # Print progress every 10 iterations
        print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}, Gradient = {gradient:.4f}")

print(f"\nFinal x after {num_iterations} iterations: {x:.4f}")
print(f"Final f(x) (Loss) after {num_iterations} iterations: {f(x):.4f}")

# --- Visualization ---
# Generate x values for plotting the function
x_vals = np.linspace(-12, 12, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='$f(x) = x^2$', color='blue')
plt.scatter(x_history, f_x_history, color='red', s=50, zorder=5, label='Gradient Descent Path')
plt.plot(x_history, f_x_history, color='red', linestyle='--', alpha=0.6)
plt.title('Gradient Descent for $f(x) = x^2$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.grid(True)
plt.legend()
plt.show()