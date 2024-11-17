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
from collections import Counter
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
            img_path = img_path[len("MURA-v1.1/"):]

        # Remove any leading dataset directory prefix, like "train/" or "valid/"
        if img_path.startswith("train/"):
            img_path = img_path[len("train/"):]
            prefix = "MURA-v1.1/train/"
        elif img_path.startswith("valid/"):
            img_path = img_path[len("valid/"):]
            prefix = "MURA-v1.1/valid/"
        else:
            raise ValueError(f"Invalid image path: {img_path}")

        # Construct the full image path
        full_img_path = os.path.normpath(os.path.join(self.root_dir, img_path))

        # Construct the full key to query the labels dictionary
        relative_study_dir = os.path.dirname(img_path).replace("\\", "/")  # Normalize slashes
        full_study_dir_key = f"{prefix}{relative_study_dir}/"  # Add the correct prefix and ensure trailing slash
        #print("Full study directory key being queried:", full_study_dir_key)  # Debugging

        # Fetch the label from `self.labels`
        label = self.labels.get(full_study_dir_key, -1)  # Default to -1 if the key is not found
        if label == -1:
            raise KeyError(f"Label not found for {full_study_dir_key}. Please check your dataset and labels.")

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
        transform=get_transforms(train=False),  # Validation-style transform for originals
        augmentation_transforms=augmentation_transforms,  # Use same augmentation as training
        include_original=True  # Include original images
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: set_seed(42))  # Ensure workers are seeded properly)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda _: set_seed(42))  # Ensure workers are seeded properly)

    # Print dataset statistics
    print(f"Found {len(train_dataset)} validated image filenames belonging to {len(set([train_dataset[i][1] for i in range(len(train_dataset))]))} classes in the training set.")
    print(f"Found {len(valid_dataset)} validated image filenames belonging to {len(set([valid_dataset[i][1] for i in range(len(valid_dataset))]))} classes in the validation set.")


    return train_loader, valid_loader


def confirm_images_and_labels(loader, dataset_name):
    """
    Confirms that all images and labels in a DataLoader are properly loaded.

    Parameters:
        loader (DataLoader): The DataLoader for the dataset.
        dataset_name (str): The name of the dataset ('train' or 'valid').
    """
    total_images = 0
    all_labels = []

    print(f"Checking {dataset_name} dataset...")
    for images, labels in loader:
        total_images += len(images)
        all_labels.extend(labels.numpy())  # Collect labels as a flat list

    unique_labels = np.unique(all_labels)  # Use NumPy for unique label extraction
    print(f"Total {dataset_name} images: {total_images}")
    print(f"Unique labels in {dataset_name} dataset: {unique_labels.tolist()}\n")

def count_body_parts(dataset, dataset_name):
    """
    Counts occurrences of each body part in the dataset.

    Parameters:
        dataset (MURADataset): The dataset object.
        dataset_name (str): The name of the dataset ('train' or 'valid').
    """
    body_parts = []
    for image_path in dataset.image_paths:
        # Extract the folder name directly after 'train/' or 'valid/'
        if "train" in image_path:
            body_part = image_path.split("train/")[1].split("/")[0]
        elif "valid" in image_path:
            body_part = image_path.split("valid/")[1].split("/")[0]
        else:
            body_part = "Unknown"  # Fallback case if the path structure is unexpected
        body_parts.append(body_part)

    # Use NumPy or collections.Counter for counting
    counts = Counter(body_parts)
    print(f"{dataset_name.capitalize()} dataset body part distribution:")
    for part, count in counts.items():
        print(f"{part}: {count}")
    print()
    
def count_body_parts_with_augmentations(dataset, dataset_name, num_augmentations):
    """
    Counts occurrences of each body part in the dataset, including augmented samples.

    Parameters:
        dataset (MURADataset): The dataset object.
        dataset_name (str): The name of the dataset ('train' or 'valid').
        num_augmentations (int): The number of augmentations applied per image.
    """
    body_parts = []
    for image_path in dataset.image_paths:
        if "train" in image_path:
            body_part = image_path.split("train/")[1].split("/")[0]
        elif "valid" in image_path:
            body_part = image_path.split("valid/")[1].split("/")[0]
        else:
            body_part = "Unknown"
        body_parts.append(body_part)

    # Count original occurrences
    counts = Counter(body_parts)

    print(f"{dataset_name.capitalize()} dataset body part distribution (with augmentations):")
    for part, count in counts.items():
        augmented_count = count * (1 + num_augmentations)  # Original + augmented
        print(f"{part}: Original: {count}, Augmented: {augmented_count}")
    print()
    
def count_positive_negative(dataset, dataset_name, num_augmentations=0):
    """
    Counts positive and negative cases for each body part in the dataset, including augmented samples.

    Parameters:
        dataset (MURADataset): The dataset object.
        dataset_name (str): The name of the dataset ('train' or 'valid').
        num_augmentations (int): Number of augmentations applied per image.

    Returns:
        None
    """
    # Extract body parts and labels as NumPy arrays
    body_parts = []
    labels = []

    # Normalize all keys in the labels dictionary to use forward slashes
    normalized_labels = {
        key.replace("\\", "/"): value for key, value in dataset.labels.items()
    }

    #print("Labels dictionary keys (first 5):")
    #print(list(normalized_labels.keys())[:5])  # Debugging: print some keys from normalized labels

    for path in dataset.image_paths:
        # Extract the body part from the path (e.g., XR_ELBOW, XR_SHOULDER)
        if "train" in path:
            body_part = path.split("train/")[1].split("/")[0]
        elif "valid" in path:
            body_part = path.split("valid/")[1].split("/")[0]
        else:
            body_part = "Unknown"

        # Get the corresponding label
        study_dir = os.path.dirname(path).replace("\\", "/")  # Normalize to forward slashes
        study_dir_key = study_dir + "/"  # Ensure trailing slash
        #print("Study directory key being queried:")
        #print(study_dir_key)  # Debugging: print the constructed study_dir_key

        # Query the normalized labels dictionary
        label = normalized_labels.get(study_dir_key, None)  # Default to None if the key is missing
        if label is None:
            raise KeyError(f"Key not found for study_dir: {study_dir_key}")
        body_parts.append(body_part)
        labels.append(label)

    # Convert to NumPy arrays for vectorized operations
    body_parts = np.array(body_parts)
    labels = np.array(labels)

    unique_parts = np.unique(body_parts)
    counts = {}

    # Vectorized counting
    for part in unique_parts:
        part_mask = (body_parts == part)  # Boolean mask for the current body part
        part_labels = labels[part_mask]
        positive_count = np.sum(part_labels == 1)  # Count positives
        negative_count = np.sum(part_labels == 0)  # Count negatives

        # Include augmented data in the counts
        augmented_positive = positive_count * (1 + num_augmentations)
        augmented_negative = negative_count * (1 + num_augmentations)

        counts[part] = {
            "positive": positive_count,
            "negative": negative_count,
            "augmented_positive": augmented_positive,
            "augmented_negative": augmented_negative,
        }

    # Print the results
    print(f"{dataset_name.capitalize()} dataset positive/negative distribution (with augmentations):")
    for part, count in counts.items():
        print(f"{part}: Positive: {count['positive']} (Augmented: {count['augmented_positive']}), "
              f"Negative: {count['negative']} (Augmented: {count['augmented_negative']})")
    print()

# # Main execution (example)
# if __name__ == "__main__":
#     data_dir = "datasets/MURA-v1.1"
#     batch_size = 32

#     train_loader = load_data(data_dir, batch_size=batch_size)

#     # Quick check
#     for images, labels in train_loader:
#         print(f"Batch size: {len(images)}, Labels: {labels}")
#         break