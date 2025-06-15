import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Define our simple function (our "loss function")
def f(x):
    return x**2

# 2. Define its derivative (our "gradient")
def df_dx(x):
    return 2 * x # This is the analytical gradient for x^2

# --- Gradient Descent Parameters ---
learning_rate = 0.1
num_iterations = 50

# 3. Starting point (our initial "weight" or "parameter")
x_start = 10.0 # Use a specific starting variable for clarity

# --- Gradient Descent Calculation ---
# We'll store all steps to animate them later
x_path = [x_start]
f_x_path = [f(x_start)]

current_x = x_start
for i in range(num_iterations):
    gradient = df_dx(current_x)
    current_x = current_x - learning_rate * gradient
    x_path.append(current_x)
    f_x_path.append(f(current_x))

print(f"Final x after {num_iterations} iterations: {x_path[-1]:.4f}")
print(f"Final f(x) (Loss) after {num_iterations} iterations: {f_x_path[-1]:.4f}")

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the function background
x_vals_plot = np.linspace(min(x_path) - 2, max(x_path) + 2, 400)
y_vals_plot = f(x_vals_plot)
ax.plot(x_vals_plot, y_vals_plot, label='$f(x) = x^2$', color='blue', alpha=0.7)
ax.set_title('Gradient Descent Animation for $f(x) = x^2$')
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.grid(True)
ax.legend()
ax.set_xlim(min(x_vals_plot), max(x_vals_plot))
ax.set_ylim(0, max(f_x_path) + 5) # Adjust y-limit to ensure path is visible

# Initialize the elements that will be animated
# This line will show the path taken by gradient descent
path_line, = ax.plot([], [], 'r--', alpha=0.6, label='Descent Path')
# This point will represent the current position
current_point, = ax.plot([], [], 'ro', markersize=8, label='Current Step')
# This text will display iteration and current values
text_display = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black',
                       bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7)) # Position text in top-left with background

# Initialization function: plot the background of each frame
def init():
    path_line.set_data([], [])
    current_point.set_data([], [])
    text_display.set_text('')
    return path_line, current_point, text_display

# Animation function: this is called sequentially
def update(frame):
    # 'frame' corresponds to the index in x_path and f_x_path
    x_curr = x_path[frame]
    f_x_curr = f_x_path[frame]

    # Update the path line data up to the current frame
    path_line.set_data(x_path[:frame+1], f_x_path[:frame+1])

    # Update the current point position
    current_point.set_data([x_curr], [f_x_curr])

    # Update the text display
    text_display.set_text(f'Iteration: {frame}\n$x$: {x_curr:.4f}\n$f(x)$: {f_x_curr:.4f}')

    return path_line, current_point, text_display

# Create the animation
# frames: The number of frames to generate (equal to number of iterations + 1 for initial state)
# interval: Delay between frames in milliseconds
# blit=True: Only re-draw the parts that have changed, makes it faster
# Setting blit=False to ensure the text box with background is redrawn correctly each frame.
ani = FuncAnimation(fig, update, frames=len(x_path),
                    init_func=init, blit=False, interval=100)

plt.show()

# Optional: To save the animation as a GIF or MP4
# You might need to install 'imagemagick' (for GIF) or 'ffmpeg' (for MP4) on your system
# On Mac, you can install with Homebrew: `brew install imagemagick` or `brew install ffmpeg`

# Uncomment the following lines to save the animation
# from matplotlib.animation import PillowWriter # For GIF
# writer_gif = PillowWriter(fps=10) # frames per second
# ani.save("gradient_descent_animation.gif", writer=writer_gif)
# print("Animation saved as gradient_descent_animation.gif")

# from matplotlib.animation import FFMpegWriter # For MP4
# writer_mp4 = FFMpegWriter(fps=10)
# ani.save("gradient_descent_animation.mp4", writer=writer_mp4)
# print("Animation saved as gradient_descent_animation.mp4")
