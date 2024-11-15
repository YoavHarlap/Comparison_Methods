import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# Create a synthetic image (e.g., a 2D Gaussian)
def generate_gaussian_image(size=128):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X**2 + Y**2) * 10))  # A 2D Gaussian
    return Z



# Fourier Transform with Magnitude and Phase
def apply_fourier_transform(image):
    F = fft2(image)
    magnitude = np.abs(F)
    phase = np.angle(F)
    return magnitude, phase

# Reconstruct using only magnitude and incorrect phase
def reconstruct_with_incorrect_phase(magnitude):
    random_phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)  # Random phase
    F_incorrect = magnitude * np.exp(1j * random_phase)  # Combine magnitude and random phase
    image_reconstructed = np.abs(ifft2(F_incorrect))  # Inverse Fourier Transform
    return image_reconstructed

# Generate original image and apply Fourier transform
original_image = generate_gaussian_image(size=128)
magnitude, original_phase = apply_fourier_transform(original_image)

# Reconstruct the image with an incorrect phase
reconstructed_image = reconstruct_with_incorrect_phase(magnitude)

# Plot the original and reconstructed images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reconstructed with same Magnitude and Incorrect Phase")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')

plt.show()
