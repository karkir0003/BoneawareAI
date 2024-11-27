import os
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import random
import numpy as np
import pandas as pd
import sys
from collections import Counter

# Add the 'src' folder to Python's module search path
sys.path.append("../src")
# Add the 'datasets' folder to Python's module search path
sys.path.append("../datasets")
# Add the 'notebooks' folder to Python's module search path
sys.path.append("../notebooks")


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Call set_seed at the beginning of the file
set_seed(42)

# Define global mean and std for all other files
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# Define Dataset class for MURA
class MURADataset(Dataset):
    def __init__(self, image_csv, label_csv, root_dir, augmentation_transforms=None):
        """
        Initializes a MURADataset object.

        Parameters:
            image_csv (str): Path to the CSV file containing image paths.
            label_csv (str): Path to the CSV file containing labels.
            root_dir (str): Root directory of the dataset.
            augmentation_transforms (list, optional): List of transforms for training augmentation, including an identity transform for the original image.
        """
        # Load image paths and labels into pandas DataFrames
        self.image_df = pd.read_csv(image_csv, header=None, names=["image_path"])
        label_df = pd.read_csv(label_csv, header=None, names=["study_path", "label"])

        # Normalize paths for consistency
        self.image_df["image_path"] = self.image_df["image_path"].str.replace("\\", "/")
        label_df["study_path"] = label_df["study_path"].str.replace("\\", "/")

        self.label_map = pd.Series(
            label_df["label"].values, index=label_df["study_path"]
        ).to_dict()
        self.root_dir = root_dir
        self.augmentation_transforms = augmentation_transforms or []

    def __len__(self):
        """
        Returns the total number of augmented images in the dataset.
        """
        return len(self.image_df) * len(self.augmentation_transforms)

    def __getitem__(self, idx):
        """
        Returns the augmented image and label at the given index.
        """
        # Determine image and transform indices
        original_idx = idx % len(self.image_df)
        transform_idx = idx // len(self.image_df)

        # Get the image path
        img_path = self.image_df.iloc[original_idx]["image_path"]
        rel_path_prefix = "/".join(self.root_dir.split("/")[-2:])
        relative_img_path = os.path.relpath(img_path, start=rel_path_prefix)
        full_img_path = os.path.normpath(os.path.join(self.root_dir, relative_img_path))

        # Determine dataset type for label lookup
        if "train" in self.root_dir:
            dataset_type = "train"
        elif "valid" in self.root_dir:
            dataset_type = "valid"
        else:
            raise ValueError(
                f"Unrecognized dataset type in root directory: {self.root_dir}"
            )

        # Add 'MURA-v1.1/train/' or 'MURA-v1.1/valid/' to match label_map keys
        relative_study_dir = os.path.dirname(relative_img_path).replace("\\", "/")
        full_study_dir_key = f"MURA-v1.1/{dataset_type}/{relative_study_dir}/".replace(
            "\\", "/"
        )

        # Fetch the label
        label = self.label_map.get(full_study_dir_key, -1)
        if label == -1:
            raise KeyError(f"Label not found for study path: {full_study_dir_key}")

        # Load the image
        image = Image.open(full_img_path).convert("RGB")

        # Apply the appropriate augmentation transform
        transform = self.augmentation_transforms[transform_idx]
        image = transform(image)

        return image, label


def get_augmented_transforms():
    """
    Returns multiple torchvision.transforms.Compose objects for data augmentation.

    Returns:
        list: A list of torchvision.transforms.Compose objects for augmentation.
    """
    return [
        transforms.Compose(
            [  # Identity transform for the original image
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=20, scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ColorJitter(hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
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
    # Define file paths
    train_image_csv = os.path.join(data_dir, "train_image_paths.csv")
    train_label_csv = os.path.join(data_dir, "train_labeled_studies.csv")
    train_dir = os.path.join(data_dir, "train")
    train_dir = train_dir.replace("\\", "/")

    valid_image_csv = os.path.join(data_dir, "valid_image_paths.csv")
    valid_label_csv = os.path.join(data_dir, "valid_labeled_studies.csv")
    valid_dir = os.path.join(data_dir, "valid")
    valid_dir = valid_dir.replace("\\", "/")

    # Define augmentation transforms
    augmentation_transforms = get_augmented_transforms()

    # Create datasets
    train_dataset = MURADataset(
        train_image_csv,
        train_label_csv,
        train_dir,
        augmentation_transforms=augmentation_transforms,
    )
    valid_dataset = MURADataset(
        # Identity only
        valid_image_csv,
        valid_label_csv,
        valid_dir,
        augmentation_transforms=augmentation_transforms[:1],
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Loaded {len(train_dataset)} training samples and {len(valid_dataset)} validation samples."
    )
    return train_loader, valid_loader


def confirm_images_and_labels(dataset, dataset_name):
    """
    Confirms that all images and labels in a dataset are properly loaded.

    Parameters:
        dataset (Dataset): The dataset object (not DataLoader).
        dataset_name (str): The name of the dataset ('train' or 'valid').
    """
    print(f"Checking {dataset_name} dataset...")

    # Extract all labels in a vectorized manner
    labels = [dataset[i][1] for i in range(len(dataset))]
    labels = np.array(labels)  # Convert to NumPy array

    # Compute statistics
    total_images = len(labels)
    unique_labels = np.unique(labels)

    print(f"Total {dataset_name} images: {total_images}")
    print(f"Unique labels in {dataset_name} dataset: {unique_labels.tolist()}\n")


def count_body_parts(dataset, dataset_name):
    """
    Counts occurrences of each body part in the dataset and displays a summary table.

    Parameters:
        dataset (MURADataset): The dataset object.
        dataset_name (str): The name of the dataset ('train' or 'valid').
    """
    # Extract body parts from image paths in `dataset.image_df`
    body_parts = dataset.image_df["image_path"].apply(
        lambda path: path.split("train/" if "train" in path else "valid/")[1].split(
            "/"
        )[0]
        if "train" in path or "valid" in path
        else "Unknown"
    )

    # Create a DataFrame for analysis
    df = pd.DataFrame({"BodyPart": body_parts})
    summary = df["BodyPart"].value_counts().reset_index()
    summary.columns = ["BodyPart", "Count"]

    print(f"{dataset_name.capitalize()} dataset body part distribution:")
    display(summary)  # Display the summary


def count_body_parts_with_augmentations(dataset, dataset_name, num_augmentations):
    """
    Counts occurrences of each body part in the dataset, including augmented samples,
    and displays a summary table.

    Parameters:
        dataset (MURADataset): The dataset object.
        dataset_name (str): The name of the dataset ('train' or 'valid').
        num_augmentations (int): The number of augmentations applied per image.
    """
    # Extract body parts
    body_parts = dataset.image_df["image_path"].apply(
        lambda path: path.split("train/" if "train" in path else "valid/")[1].split(
            "/"
        )[0]
        if "train" in path or "valid" in path
        else "Unknown"
    )

    # Create a DataFrame for analysis
    df = pd.DataFrame({"BodyPart": body_parts})
    body_part_counts = df["BodyPart"].value_counts()

    # Calculate augmented counts
    augmented_counts = body_part_counts * (1 + num_augmentations)

    # Create a summary DataFrame
    summary = pd.DataFrame(
        {
            "BodyPart": body_part_counts.index,
            "OriginalCount": body_part_counts.values,
            "AugmentedCount": augmented_counts.values,
        }
    )

    print(
        f"{dataset_name.capitalize()} dataset body part distribution (with augmentations):"
    )
    display(summary)


def count_positive_negative(dataset, dataset_name, num_augmentations=0):
    """
    Counts positive and negative cases for each body part in the dataset, including augmented samples,
    and displays a summary table.

    Parameters:
        dataset (MURADataset): The dataset object.
        dataset_name (str): The name of the dataset ('train' or 'valid').
        num_augmentations (int): Number of augmentations applied per image.
    """
    # Extract body parts and corresponding labels
    body_parts = dataset.image_df["image_path"].apply(
        lambda path: path.split("train/" if "train" in path else "valid/")[1].split(
            "/"
        )[0]
        if "train" in path or "valid" in path
        else "Unknown"
    )
    labels = dataset.image_df["image_path"].apply(
        lambda path: dataset.label_map.get(
            os.path.dirname(path).replace("\\", "/") + "/", -1
        )
    )

    # Create a DataFrame for analysis
    df = pd.DataFrame({"BodyPart": body_parts, "Label": labels})

    # Group by BodyPart and Label, and calculate counts
    summary = (
        df.groupby(["BodyPart", "Label"]).size().unstack(fill_value=0).reset_index()
    )
    summary.columns = ["BodyPart", "Negative", "Positive"]

    # Add augmented counts
    summary["AugmentedNegative"] = summary["Negative"] * (1 + num_augmentations)
    summary["AugmentedPositive"] = summary["Positive"] * (1 + num_augmentations)

    print(
        f"{dataset_name.capitalize()} dataset positive/negative distribution (with augmentations):"
    )
    display(summary)
