import os
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import random
import numpy as np
import sys
sys.path.append('../src')  # Add the 'src' folder to Python's module search path
sys.path.append('../datasets')  # Add the 'datasets' folder to Python's module search path
sys.path.append('../notebooks')  # Add the 'notebooks' folder to Python's module search path


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Call set_seed at the beginning of the file
set_seed(42)




# Define Dataset class for MURA
class MURADataset(Dataset):
    def __init__(self, image_csv, label_csv, root_dir, transform=None, augmentation_transforms=None, include_original=True):
        """
        Initializes a MURADataset object.

        Parameters:
            image_csv (str): Path to the CSV file containing image paths.
            label_csv (str): Path to the CSV file containing labels.
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Transform for validation data or original images.
            augmentation_transforms (list, optional): List of transforms for training augmentation.
            include_original (bool): Whether to include the original image in the dataset.
        """
        self.image_paths = self._read_csv(image_csv)
        self.labels = self._read_labels(label_csv)
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation_transforms = augmentation_transforms
        self.include_original = include_original

    @staticmethod
    def _read_csv(file_path):
        """
        Reads a CSV file and returns a list of paths.

        Parameters:
            file_path (str): Path to the CSV file.

        Returns:
            list: List of image paths from the CSV file.
        """
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0:  # Ensure the row is not empty
                    data.append(row[0].strip())  # Use only the first column
        return data

    @staticmethod
    def _read_labels(label_csv):
        """
        Reads a csv file and returns a dictionary mapping folder paths to labels

        Parameters:
            label_csv (str): Path to the csv file

        Returns:
            dict: A dictionary mapping folder paths to labels
        """
        label_map = {}
        with open(label_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    folder_path, label = row
                    label_map[folder_path] = int(label)  # Ensure label is an integer
        return label_map

    def __len__(self):
        """
        Returns the total number of images in the dataset, considering augmentations.
        If `include_original` is True, count is effectively doubled (original + augmentations).
        """
        if self.include_original and self.augmentation_transforms:
            return len(self.image_paths) * (len(self.augmentation_transforms) + 1)
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Parameters:
            idx (int): Index of the image to return.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        # Determine whether this is an original or augmented sample
        original_idx = idx % len(self.image_paths)
        augmentation_idx = idx // len(self.image_paths) - 1        
        
        # Get the image path from the CSV
        img_path = self.image_paths[original_idx]

        # Remove any leading dataset directory prefix, like "MURA-v1.1/"
        if img_path.startswith("MURA-v1.1/"):
            img_path = img_path[len("MURA-v1.1/"):]  # Strip "MURA-v1.1/" from the start

        # Remove any leading dataset directory prefix, like "train/"
        if img_path.startswith("train/"):
            img_path = img_path[len("train/"):]  # Strip "train/" from the start            
        # Remove any leading dataset directory prefix, like "valid/"
        elif img_path.startswith("valid/"):
            img_path = img_path[len("valid/"):]  # Strip "valid/" from the start            

        # Construct the full image path
        full_img_path = os.path.normpath(os.path.join(self.root_dir, img_path))
        relative_study_dir = os.path.dirname(img_path)
        label = self.labels.get(relative_study_dir, -1)

        # Load the image
        image = Image.open(full_img_path).convert("RGB")

        # Apply the appropriate transformation
        if self.include_original and augmentation_idx == -1:
            # Original image with validation-style transform
            if self.transform:
                image = self.transform(image)
        elif self.augmentation_transforms:
            # Apply augmentation transformation
            transform = self.augmentation_transforms[augmentation_idx]
            image = transform(image)

        return image, label


# Preprocessing and Data Augmentation
def get_transforms(train=True):
    """
    Returns a torchvision.transforms.Compose object containing the preprocessing and data augmentation steps for the MURA dataset.

    Parameters:
        train (bool): If True, returns a transform object for training data. If False, returns a transform object for validation data.

    Returns:
        transform (torchvision.transforms.Compose): A torchvision.transforms.Compose object containing the preprocessing and data augmentation steps for the MURA dataset.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
def get_augmented_transforms():
    """
    Returns multiple torchvision.transforms.Compose objects for data augmentation.

    Returns:
        list: A list of torchvision.transforms.Compose objects for augmentation.
    """
    return [
        transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=20, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ColorJitter(hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]


# Function to load the datasets
def load_data(data_dir, batch_size=32):
    """
    Loads the MURA dataset from a given directory and returns a train data loader and a validation data loader.

    Parameters:
        data_dir (str): The directory containing the MURA dataset. This directory should contain a "train" folder and a "valid" folder, each containing the respective images. Additionally, there should be two csv files: "train_image_paths.csv" and "valid_image_paths.csv", each containing the paths to the images in the respective folders.

    Returns:
        train_loader (DataLoader): A DataLoader for the training data.
        valid_loader (DataLoader): A DataLoader for the validation data.
    """
    # File paths for training
    train_image_csv = os.path.join(data_dir, "train_image_paths.csv")
    train_label_csv = os.path.join(data_dir, "train_labeled_studies.csv")
    train_dir = os.path.join(data_dir, "train")

    # File paths for validation
    valid_image_csv = os.path.join(data_dir, "valid_image_paths.csv")
    valid_label_csv = os.path.join(data_dir, "valid_labeled_studies.csv")
    valid_dir = os.path.join(data_dir, "valid")

    # Augmentation for training
    augmentation_transforms = get_augmented_transforms()

    train_dataset = MURADataset(
        train_image_csv, train_label_csv, train_dir,
        transform=get_transforms(train=False),  # Validation-style transform for originals
        augmentation_transforms=augmentation_transforms,
        include_original=True  # Include original images
    )
    valid_dataset = MURADataset(
        valid_image_csv, valid_label_csv, valid_dir,
        transform=get_transforms(train=False)  # Standard validation transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: set_seed(42))  # Ensure workers are seeded properly)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: set_seed(42))  # Ensure workers are seeded properly)

    # Print dataset statistics
    print(f"Found {len(train_dataset)} validated image filenames belonging to {len(set([train_dataset[i][1] for i in range(len(train_dataset))]))} classes in the training set.")
    print(f"Found {len(valid_dataset)} validated image filenames belonging to {len(set([valid_dataset[i][1] for i in range(len(valid_dataset))]))} classes in the validation set.")


    return train_loader, valid_loader


# Main execution (example)
if __name__ == "__main__":
    data_dir = "datasets/MURA-v1.1"
    batch_size = 32

    train_loader = load_data(data_dir, batch_size=batch_size)

    # Quick check
    for images, labels in train_loader:
        print(f"Batch size: {len(images)}, Labels: {labels}")
        break