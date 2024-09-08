import os
from PIL import Image

# Define the folder path containing the images
folder_path = r"C:\Users\ASUS\Documents\code_images\andrew"

# Define the crop amounts (in pixels)
crop_top = 76
crop_bottom = 285
crop_left = 203
crop_right = 993

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a PNG image
    if filename.lower().endswith('.png'):
        # Full path to the current image
        image_path = os.path.join(folder_path, filename)

        # Open the image
        image = Image.open(image_path)

        # Get the original image dimensions
        width, height = image.size

        # Define the coordinates for the cropping box
        left = crop_left
        top = crop_top
        right = width - crop_right
        bottom = height - crop_bottom

        # Ensure the cropping box coordinates are within the image dimensions
        if left < right and top < bottom:
            # Crop the image
            cropped_image = image.crop((left, top, right, bottom))

            # Define the output path for the cropped image
            output_path = os.path.join(folder_path, f"cropped_{filename}")

            # Save the cropped image
            cropped_image.save(output_path)

            print(f"Cropped image saved to: {output_path}")
        else:
            print(f"Cropping dimensions are out of bounds for file: {filename}")
