import os
import shutil
import random

def split_train_val_test(data_dir, val_split=0.2, test_split=0.1):
    """
    Splits the dataset into training, validation, and testing sets while preserving the subdirectory structure.
    
    Parameters:
    - data_dir (str): Path to the dataset folder containing "train".
    - val_split (float): Percentage of data to be used for validation.
    - test_split (float): Percentage of data to be used for testing.

    The function assumes data is organized as:
    data_dir/
        ├── train/
        │   ├── category1/
        │   │   ├── subcategory1/
        │   │   ├── subcategory2/
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Ensure validation and test directories exist
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for root, _, files in os.walk(train_dir):
        if not files:
            continue  # Skip empty directories

        relative_path = os.path.relpath(root, train_dir)
        val_class_path = os.path.join(val_dir, relative_path)
        test_class_path = os.path.join(test_dir, relative_path)

        # Ensure subdirectories exist in val and test
        os.makedirs(val_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        random.shuffle(files)  # Shuffle images

        test_size = int(len(files) * test_split)
        val_size = int(len(files) * val_split)

        test_images = files[:test_size]
        val_images = files[test_size:test_size + val_size]

        # Move test images
        for img in test_images:
            shutil.move(os.path.join(root, img), os.path.join(test_class_path, img))

        # Move validation images
        for img in val_images:
            shutil.move(os.path.join(root, img), os.path.join(val_class_path, img))

        print(f"Moved {len(test_images)} images from {relative_path} to test set.")
        print(f"Moved {len(val_images)} images from {relative_path} to validation set.")

# Example usage:
split_train_val_test("data/")  # Change this to your dataset path
