import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

# Set up the canvas
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_axis_off()

# Initialize the starting random noise
# Make the grid SMALLER for bigger blocks (e.g., 15x15)
image_data = np.random.rand(15, 15)

# Create the final target image (simple smiley face)
target_image = np.zeros((15, 15))
# Draw eyes
target_image[4, 3] = 1
target_image[4, 11] = 1
# Draw mouth
target_image[10, 5:10] = 1

# Show the initial image
img = ax.imshow(image_data, cmap='plasma', interpolation='nearest', vmin=0, vmax=1)

# Update function for each frame
def update(frame):
    global image_data
    total_frames = 60
    if frame < total_frames * 0.7:
        # First 70% of the time: just random noise evolution
        noise = np.random.normal(0, 0.03, image_data.shape)
        image_data = np.clip(image_data + noise, 0, 1)
    else:
        # Last 30% of the time: blend toward the target image
        blend_factor = (frame - total_frames * 0.7) / (total_frames * 0.3)
        image_data = (1 - blend_factor) * image_data + blend_factor * target_image
    img.set_data(image_data)
    return [img]

# Create the animation
ani = FuncAnimation(fig, update, frames=60, interval=100, blit=True)

# Save the animation as a GIF
ani.save('diffusion_to_smiley.gif', writer=PillowWriter(fps=10))

print("GIF created successfully: diffusion_to_smiley.gif")
