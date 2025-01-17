import matplotlib.pyplot as plt
import numpy as np

# Define the vector v and the line l
v = np.array([3, 4])  # The original vector to be reflected
l = np.array([1, 1])  # A vector parallel to the line of reflection

# Calculate projection of v onto l
proj_l_v = (np.dot(v, l) / np.dot(l, l)) * l

# Calculate the reflection of v across l
ref_v = 2 * proj_l_v - v

# Define figure size explicitly
plt.figure(figsize=(8, 8))  # This ensures the figure size is fixed

# Plot the original vector v
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label=f'Original Vector (v): ({v[0]}, {v[1]})')

# Plot the projection of v onto l
plt.quiver(0, 0, proj_l_v[0], proj_l_v[1], angles='xy', scale_units='xy', scale=1, color='orange', label=f'Projection (Proj_l(v)): ({proj_l_v[0]:.1f}, {proj_l_v[1]:.1f})')

# Plot the reflected vector
plt.quiver(0, 0, ref_v[0], ref_v[1], angles='xy', scale_units='xy', scale=1, color='green', label=f'Reflected Vector (Ref_l(v)): ({int(ref_v[0])}, {int(ref_v[1])})')

# Add the line of reflection (l)
x = np.linspace(-5, 10, 100)
y = x  # Line y = x, which is parallel to l
plt.plot(x, y, 'red', linestyle='--', label='Line of Reflection (l: y=x)')

# Optional: Add dashed line connecting v and ref_v
# Uncomment if needed
plt.plot([v[0], ref_v[0]], [v[1], ref_v[1]], 'purple', linestyle='--', label='Dashed Line: v to Ref_l(v)')

# Annotate points
plt.scatter(v[0], v[1], color='blue')
plt.scatter(proj_l_v[0], proj_l_v[1], color='orange')
plt.scatter(ref_v[0], ref_v[1], color='green')

# Adding grid, legend, and labels
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()

# Set limits to keep the frame size consistent
plt.xlim(-2, 6)
plt.ylim(-2, 6)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Reflection of a Vector Across a Line")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.savefig(r"C:\Users\ASUS\Documents\code_images\overleaf_images\presentation\reflection_2.png", dpi=300)

# Display the plot
plt.show()
