# #################################################
# #################### fig 1  ####################
# #
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import data, color
# from skimage.transform import resize
# from numpy.fft import fft2, ifft2, fftshift, ifftshift
#
# # Load and preprocess an image
# image = color.rgb2gray(data.astronaut())
# image = resize(image, (256, 256), anti_aliasing=True)
#
# # Compute the Fourier Transform
# f_transform = fft2(image)
# magnitude = np.abs(f_transform)
# phase = np.angle(f_transform)
#
# # Generate a new image with the same magnitude but randomized phase
# random_phase = np.random.uniform(-np.pi, np.pi, phase.shape)
# new_f_transform = magnitude * np.exp(1j * random_phase)
# reconstructed_image = np.abs(ifft2(new_f_transform))
#
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
#
#
# #################################################
# #################### fig 2  ####################
#
# # Load a second image (use a different built-in grayscale image)
# image2 = data.coins()  # Another grayscale image
# image2 = resize(image2, (256, 256), anti_aliasing=True)
#
# # Fourier transform of the second image
# f_transform2 = fft2(image2)
# phase2 = np.angle(f_transform2)
#
# # Replace the phase of the first image with the phase of the second
# combined_f_transform = magnitude * np.exp(1j * phase2)
# combined_image = np.abs(ifft2(combined_f_transform))
#
# # Plot original, phase-replaced, and the second image
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
#
#
#
#
# #################################################
# #################### fig 3  ####################
# ##########################
# # Adjust the grid creation to avoid using the unavailable function
# from skimage import draw
#
# # Create a grid pattern (a periodic image)
# def create_grid_image(size, spacing):
#     image = np.zeros((size, size))
#     for i in range(0, size, spacing):
#         image[i, :] = 1  # Horizontal lines
#         image[:, i] = 1  # Vertical lines
#     return image
#
# # Re-import necessary libraries for image manipulation
# from skimage import data, color
# from skimage.transform import resize
# from numpy.fft import fft2, ifft2, fftshift, ifftshift
#
# # Create the periodic grid image
# size = 256  # Image size (256x256 pixels)
# spacing = 16  # Spacing between grid lines
# grid_image = create_grid_image(size, spacing)
#
# # Use the coins image for phase replacement
# image2 = data.coins()
# image2 = resize(image2, (size, size), anti_aliasing=True)
#
# # Fourier transform of both images
# f_transform_grid = fft2(grid_image)
# f_transform_image2 = fft2(image2)
#
# # Extract magnitude and phase
# magnitude_grid = np.abs(f_transform_grid)
# phase_image2 = np.angle(f_transform_image2)
#
# # Combine magnitude of the grid image with the phase of the second image
# combined_f_transform = magnitude_grid * np.exp(1j * phase_image2)
# combined_image = np.abs(ifft2(combined_f_transform))
#
# # Plot the grid image, the second image (phase source), and the combined image
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))
#
# # Original grid image
# ax[0].imshow(grid_image, cmap='gray')
# ax[0].set_title("Original Grid Image")
# ax[0].axis('off')
#
# # Second image (phase source)
# ax[2].imshow(image2, cmap='gray')
# ax[2].set_title("Second Image (Phase Source)")
# ax[2].axis('off')
#
# # Combined image
# ax[1].imshow(combined_image, cmap='gray')
# ax[1].set_title("Combined Image (Grid Magnitude, Phase of Second Image)")
# ax[1].axis('off')
#
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift
import os

output_dir = r"C:\Users\ASUS\Documents\code_images\overleaf_images\intro2"
os.makedirs(output_dir, exist_ok=True)

# ###### FIG 1 ######
image = color.rgb2gray(data.astronaut())
image = resize(image, (256, 256), anti_aliasing=True)

f_transform = fft2(image)
magnitude = np.abs(f_transform)
phase = np.angle(f_transform)

random_phase = np.random.uniform(-np.pi, np.pi, phase.shape)
new_f_transform = magnitude * np.exp(1j * random_phase)
reconstructed_image = np.abs(ifft2(new_f_transform))

plt.imsave(os.path.join(output_dir, "fig1_original_image.png"), image, cmap='gray')
plt.imsave(os.path.join(output_dir, "fig1_magnitude_spectrum.png"), np.log(1 + fftshift(magnitude)), cmap='gray')
plt.imsave(os.path.join(output_dir, "fig1_reconstructed_image.png"), reconstructed_image, cmap='gray')


# ###### FIG 2 ######
image2 = data.coins()
image2 = resize(image2, (256, 256), anti_aliasing=True)

f_transform2 = fft2(image2)
phase2 = np.angle(f_transform2)

combined_f_transform = magnitude * np.exp(1j * phase2)
combined_image = np.abs(ifft2(combined_f_transform))

plt.imsave(os.path.join(output_dir, "fig2_original_image.png"), image, cmap='gray')
plt.imsave(os.path.join(output_dir, "fig2_phase_replaced_image.png"), combined_image, cmap='gray')
plt.imsave(os.path.join(output_dir, "fig2_second_image.png"), image2, cmap='gray')


# ###### FIG 3 ######
def create_grid_image(size, spacing):
    image = np.zeros((size, size))
    for i in range(0, size, spacing):
        image[i, :] = 1
        image[:, i] = 1
    return image

size = 256
spacing = 16
grid_image = create_grid_image(size, spacing)

f_transform_grid = fft2(grid_image)
f_transform_image2 = fft2(image2)

magnitude_grid = np.abs(f_transform_grid)
phase_image2 = np.angle(f_transform_image2)

combined_f_transform = magnitude_grid * np.exp(1j * phase_image2)
combined_image = np.abs(ifft2(combined_f_transform))

plt.imsave(os.path.join(output_dir, "fig3_grid_image.png"), grid_image, cmap='gray')
plt.imsave(os.path.join(output_dir, "fig3_second_image_phase_source.png"), image2, cmap='gray')
plt.imsave(os.path.join(output_dir, "fig3_combined_image.png"), combined_image, cmap='gray')

print(f"All images saved in {output_dir}")
