#################################################
#################### fig 1  ####################
#
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# Load and preprocess an image
image = color.rgb2gray(data.astronaut())
image = resize(image, (256, 256), anti_aliasing=True)

# Compute the Fourier Transform
f_transform = fft2(image)
magnitude = np.abs(f_transform)
phase = np.angle(f_transform)

# Generate a new image with the same magnitude but randomized phase
random_phase = np.random.uniform(-np.pi, np.pi, phase.shape)
new_f_transform = magnitude * np.exp(1j * random_phase)
reconstructed_image = np.abs(ifft2(new_f_transform))

# # Plot original, magnitude, and reconstructed images
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#
# # Original Image
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title("Original Image")
# ax[0].axis('off')
#
# # Magnitude Spectrum
# ax[1].imshow(np.log(1 + fftshift(magnitude)), cmap='gray')
# ax[1].set_title("Fourier Magnitude (Log Scale)")
# ax[1].axis('off')
#
# # Reconstructed Image (Same Magnitude, Random Phase)
# ax[2].imshow(reconstructed_image, cmap='gray')
# ax[2].set_title("Reconstructed (Random Phase)")
# ax[2].axis('off')
#
# plt.tight_layout()
# plt.show()


#################################################
#################### fig 2  ####################

# Load a second image (use a different built-in grayscale image)
image2 = data.coins()  # Another grayscale image
image2 = resize(image2, (256, 256), anti_aliasing=True)

# Fourier transform of the second image
f_transform2 = fft2(image2)
phase2 = np.angle(f_transform2)

# Replace the phase of the first image with the phase of the second
combined_f_transform = magnitude * np.exp(1j * phase2)
combined_image = np.abs(ifft2(combined_f_transform))

# Plot original, phase-replaced, and the second image
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#
# # Original image
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title("Original Image")
# ax[0].axis('off')
#
# # Phase-replaced image
# ax[1].imshow(combined_image, cmap='gray')
# ax[1].set_title("Image with Phase from Second Image")
# ax[1].axis('off')
#
# # Second image (providing the phase)
# ax[2].imshow(image2, cmap='gray')
# ax[2].set_title("Second Image (Phase Source)")
# ax[2].axis('off')
#
# plt.tight_layout()
# plt.show()





#################################################
#################### fig 3  ####################

# Compute histograms of the original and reconstructed images
original_hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 1))
reconstructed_hist, _ = np.histogram(reconstructed_image.flatten(), bins=256, range=(0, 1))

# Plot histograms
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Original image histogram
ax[0].plot(bins[:-1], original_hist, label="Original", color="blue")
ax[0].set_title("Histogram of Original Image")
ax[0].set_xlabel("Gray Level")
ax[0].set_ylabel("Frequency")
ax[0].grid(True)

# Reconstructed image histogram
ax[1].plot(bins[:-1], reconstructed_hist, label="Reconstructed", color="red")
ax[1].set_title("Histogram of Reconstructed Image (Random Phase)")
ax[1].set_xlabel("Gray Level")
ax[1].set_ylabel("Frequency")
ax[1].grid(True)

plt.tight_layout()
plt.show()




#################################################
#################### fig 4  ####################

from skimage.filters import sobel

# Compute edges (sharp changes) in the original and combined images
edges_original = sobel(image)  # Edge detection using Sobel filter
edges_combined = sobel(combined_image)

# # Plot the results
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#
# # Edges in the original image
# ax[0].imshow(edges_original, cmap='gray')
# ax[0].set_title("Edges in Original Image")
# ax[0].axis('off')
#
# # Edges in the phase-modified image
# ax[1].imshow(edges_combined, cmap='gray')
# ax[1].set_title("Edges in Image with Modified Phase")
# ax[1].axis('off')
#
# plt.tight_layout()
# plt.show()
