import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fft2, ifft2
from io import BytesIO
from PIL import Image


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
frames = 100  # Total frames in the video

# Generate original image and Fourier transform components
original_image = generate_gaussian_image(size=size)
magnitude, correct_phase = apply_fourier_transform(original_image)

# Start with a random initial phase
initial_phase = np.random.uniform(-np.pi, np.pi, size=(size, size))

# Set up video writer using OpenCV
video_filename = 'phase_approach_simulation.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 10, (size, size), False)

# Loop to create frames with a gradual approach to the correct phase
for i in range(frames):
    # Interpolate phase between initial and correct phase
    interpolation_factor = i / (frames - 1)  # Ranges from 0 to 1
    phase = (1 - interpolation_factor) * initial_phase + interpolation_factor * correct_phase

    # Reconstruct the image with the interpolated phase
    reconstructed_image = reconstruct_with_phase(magnitude, phase)

    # Plot the image to capture the frame
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.imshow(reconstructed_image, cmap='gray')
    ax.axis('off')

    # Save the frame to an in-memory file
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Close the plot to save memory

    # Convert the image in buffer to grayscale and save to video
    frame = np.array(Image.open(buf).convert('L'))
    out.write(cv2.resize(frame, (size, size)))  # Resize to match video dimensions

# Release the video writer
out.release()
print(f"Video saved as {video_filename}")
