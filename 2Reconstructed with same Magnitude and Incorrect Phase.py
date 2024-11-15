import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2


# Function to generate a 2D Gaussian image
def generate_gaussian_image(size=128):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X ** 2 + Y ** 2) * 10))  # A 2D Gaussian
    return Z


# Fourier Transform functions
def apply_fourier_transform(image):
    F = fft2(image)
    magnitude = np.abs(F)
    phase = np.angle(F)
    return magnitude, phase


def reconstruct_with_phase(magnitude, phase):
    F = magnitude * np.exp(1j * phase)  # Combine magnitude and phase
    image_reconstructed = np.abs(ifft2(F))  # Inverse Fourier Transform
    return image_reconstructed


# Initialize parameters
size = 128
key_frames = [0, 20, 40, 60, 80, 100]  # Percentage phase shifts for display

# Generate original image and Fourier transform components
original_image = generate_gaussian_image(size=size)
magnitude, correct_phase = apply_fourier_transform(original_image)

# Start with a random initial phase
initial_phase = np.random.uniform(-np.pi, np.pi, size=(size, size))

# Prepare the plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
# fig.suptitle("Phase Transition from Random to Correct Phase", fontsize=16)

# Loop to create and plot images for the specified phase shifts
for idx, shift in enumerate(key_frames):
    # Calculate the interpolation factor
    interpolation_factor = shift / 100  # Convert percentage to a 0-1 scale
    phase = (1 - interpolation_factor) * initial_phase + interpolation_factor * correct_phase

    # Reconstruct the image with the interpolated phase
    reconstructed_image = reconstruct_with_phase(magnitude, phase)

    # Plot each image in the corresponding subplot
    ax = axes[idx // 3, idx % 3]  # Calculate row and column
    ax.imshow(reconstructed_image, cmap='gray')
    ax.set_title(f'Phase Shift: {shift}%')
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
plt.show()
