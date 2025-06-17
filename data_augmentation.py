import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import shutil

# --- Configuration ---
# 1. Define the path to your training directory
train_dir = 'path/to/your/train'

# 2. Set your target number of images per class.
# A good starting point is the average number of images, or a fixed high number.
# Based on your data, a value between 400 and 500 would be reasonable.
TARGET_IMAGES_PER_CLASS = 450

# 3. Define the image dimensions
IMG_WIDTH, IMG_HEIGHT = 224, 224


# --- Augmentation Setup ---
# Create an ImageDataGenerator instance with a wide range of augmentations.
# This creates realistic variations of your existing images.
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)


# --- Main Augmentation Logic ---
def augment_minority_classes(directory, target_count):
    """
    Finds subdirectories with fewer images than target_count and generates
    new images using data augmentation until the target is met.
    """
    print("Starting data augmentation process...")
    
    # Get all class subdirectories
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for subdir in subdirectories:
        class_path = os.path.join(directory, subdir)
        
        try:
            # Get current number of image files
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            current_count = len(image_files)
            
            # Check if augmentation is needed
            if current_count < target_count:
                images_to_generate = target_count - current_count
                print(f"\nAugmenting class '{subdir}':")
                print(f"  Current images: {current_count}. Target: {target_count}.")
                print(f"  Generating {images_to_generate} new images...")

                # Get paths of original images to augment from
                original_image_paths = [os.path.join(class_path, f) for f in image_files]
                
                generated_count = 0
                while generated_count < images_to_generate:
                    # Randomly pick an image to augment
                    random_image_path = np.random.choice(original_image_paths)
                    
                    # Load the image and convert it to a numpy array
                    img = load_img(random_image_path)
                    x = img_to_array(img)  # Shape: (height, width, channels)
                    x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels) for flow()

                    # Generate one augmented image
                    for batch in datagen.flow(x, batch_size=1,
                                              save_to_dir=class_path,
                                              save_prefix='aug',
                                              save_format='jpeg'):
                        generated_count += 1
                        # Break the loop after generating one image
                        break 
                                              
                print(f"  Successfully generated {generated_count} images for class '{subdir}'.")
            else:
                print(f"\nClass '{subdir}' already has {current_count} images (meets target of {target_count}). Skipping.")

        except Exception as e:
            print(f"Could not process directory {class_path}: {e}")
            
    print("\nData augmentation process finished.")


# --- Execution ---
if __name__ == '__main__':
    # Make sure the training directory exists
    if not os.path.isdir(train_dir):
        print(f"Error: Training directory not found at '{train_dir}'.")
        print("Please update the 'train_dir' variable in the script.")
    else:
        augment_minority_classes(train_dir, TARGET_IMAGES_PER_CLASS)
        print("\nNow, you can re-run the counting script to verify the new image distribution.")

