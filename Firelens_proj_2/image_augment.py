import os
from PIL import Image

# Dir source folder and the destination folder for augmented images
src_folder = 'images/training_data/smoke/smoke_aug'  
dest_folder = 'images/training_data/smoke/smoke_rot' 

# Create the destination folder if it doesn't exist
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Function to augment an image by rotating it
def augment_image(file_path, degrees, save_folder):
    # Open the image
    with Image.open(file_path) as img:
        # Rotate the image
        rotated_img = img.rotate(degrees, expand=True)
        # Save rotated image
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        rotated_img.save(os.path.join(save_folder, f"{name}_{degrees}{ext}"), 'TIFF')

# Go through each file in the source folder
for file_name in os.listdir(src_folder):
    if file_name.lower().endswith('.tif'):
        # Dir of the source image
        file_path = os.path.join(src_folder, file_name)
        # Augment the images by rotating 90, 180, and 270 degrees
        augment_image(file_path, 90, dest_folder)
        augment_image(file_path, 180, dest_folder)
        augment_image(file_path, 270, dest_folder)

print(f"Augmentation complete. Rotated images are saved in {dest_folder}.")


#####****************************************************************************
def resize_and_replace_images_in_folder(folder_path, target_size=(256, 256)):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                # Convert to RGB if the image is not in RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize the image
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                # Replace the original image with the resized one
                resized_img.save(file_path, 'TIFF')

# # Set path to images folder
# folder_path = 'images/training_data/haze/haze_aug'
# resize_and_replace_images_in_folder(folder_path)