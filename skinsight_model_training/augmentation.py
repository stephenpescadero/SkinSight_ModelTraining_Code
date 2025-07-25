from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,           # Rotate images by up to 30 degrees
    width_shift_range=0.2,       # Shift width randomly
    height_shift_range=0.2,      # Shift height randomly
    shear_range=0.2,             # Shear transformation
    zoom_range=0.2,              # Zoom in/out
    horizontal_flip=True,        # Flip images horizontally
    vertical_flip=True,          # Flip images vertically
    brightness_range=[0.7, 1.3], # Adjust brightness
    fill_mode='nearest'
)

def augment_images(folder, target_count):
    """Augments images in a given folder until it reaches target_count."""
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(image_files)
    
    if current_count >= target_count:
        print(f"✅ Skipping {folder} (already has {current_count} images).")
        return
    
    print(f"📢 Augmenting {folder} from {current_count} to {target_count} images...")

    num_aug_per_image = max((target_count - current_count) // current_count, 1)  # Ensure at least 1 augmentation per image

    for img_name in image_files:
        img_path = os.path.join(folder, img_name)
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Generate augmented images
        generated = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=folder, save_prefix='aug', save_format='jpg'):
            generated += 1
            if generated >= num_aug_per_image:
                break  # Stop generating when enough augmentations are created

        current_count += num_aug_per_image
        if current_count >= target_count:
            break  # Stop when target is reached

def recursive_augmentation(base_dir, target_count=1900):
    """Recursively applies augmentation to all subdirectories of base_dir."""
    for root, dirs, files in os.walk(base_dir):
        if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in files):  # If images are found
            augment_images(root, target_count)

# 🔹 Apply augmentation to all conditions inside train/
base_dir = "data/val"  # Change to "data/stage2/test" or "data/stage2/val" if needed
recursive_augmentation(base_dir, target_count=200)
