# import matplotlib.pyplot as plt
# import numpy as np
#
# # Define the vector v and the line l
# v = np.array([3, 4])  # The original vector to be reflected
# l = np.array([1, 1])  # A vector parallel to the line of reflection
#
# # Calculate projection of v onto l
# proj_l_v = (np.dot(v, l) / np.dot(l, l)) * l
#
# # Calculate the reflection of v across l
# ref_v = 2 * proj_l_v - v
#
# # Plotting
# plt.figure(figsize=(8, 8))
#
# # Plot the original vector v
# plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label=f'Original Vector (v): ({v[0]}, {v[1]})')
#
# # Plot the projection of v onto l
# plt.quiver(0, 0, proj_l_v[0], proj_l_v[1], angles='xy', scale_units='xy', scale=1, color='orange', label=f'Projection (Proj_l(v)): ({proj_l_v[0]:.1f}, {proj_l_v[1]:.1f})')
#
# # Plot the reflected vector
# plt.quiver(0, 0, ref_v[0], ref_v[1], angles='xy', scale_units='xy', scale=1, color='green', label=f'Reflected Vector (Ref_l(v)): ({int(ref_v[0])}, {int(ref_v[1])})')
#
# # Add the line of reflection (l)
# x = np.linspace(-5, 10, 100)
# y = x  # Line y = x, which is parallel to l
# plt.plot(x, y, 'red', linestyle='--', label='Line of Reflection (l: y=x)')
#
# # Annotate points
# plt.scatter(v[0], v[1], color='blue')
# plt.scatter(proj_l_v[0], proj_l_v[1], color='orange')
# plt.scatter(ref_v[0], ref_v[1], color='green')
#
# # Adding grid, legend, and labels
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
# plt.legend()
# plt.xlim(-5, 10)
# plt.ylim(-5, 10)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title("Reflection of a Vector Across a Line")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
#
# # Display the plot
# plt.show()



######################################################################################3






