import os
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('../src')  # Add the 'src' folder to Python's module search path
sys.path.append('../datasets')  # Add the 'datasets' folder to Python's module search path
sys.path.append('../notebooks')  # Add the 'notebooks' folder to Python's module search path
# Define Dataset class for MURA
class MURADataset(Dataset):
    def __init__(self, image_csv, label_csv, root_dir, transform=None):
        """
        Initializes a MURADataset object
        
        Parameters:
            image_csv (str): Path to the csv file containing image paths
            label_csv (str): Path to the csv file containing labels
            root_dir (str): Root directory of the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.image_paths = self._read_csv(image_csv)  # Read image paths
        self.labels = self._read_labels(label_csv)    # Read labels
        self.root_dir = root_dir
        self.transform = transform

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
        Returns the total number of images in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Parameters:
            idx (int): Index of the image to return.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        # Get the image path from the CSV
        img_path = self.image_paths[idx]

        # Remove any leading dataset directory prefix, like "MURA-v1.1/"
        if img_path.startswith("MURA-v1.1/"):
            img_path = img_path[len("MURA-v1.1/"):]  # Strip "MURA-v1.1/" from the start

        # Remove any leading dataset directory prefix, like "train/"
        if img_path.startswith("train/"):
            img_path = img_path[len("train/"):]  # Strip "train/" from the start            
        # Remove any leading dataset directory prefix, like "valid/"
        elif img_path.startswith("valid/"):
            img_path = img_path[len("valid/"):]  # Strip "valid/" from the start            


        # Construct the full path dynamically
        full_img_path = os.path.normpath(os.path.join(self.root_dir, img_path))

        # Debugging output
        #print(f"Raw img_path from CSV: {self.image_paths[idx]}")
        #print(f"Adjusted img_path: {img_path}")
        #print(f"Constructed full_img_path: {full_img_path}")

        # Extract the study directory relative to root_dir to find the corresponding label
        relative_study_dir = os.path.dirname(img_path)
        label = self.labels.get(relative_study_dir, -1)  # Default to -1 if not found

        # Load and transform the image
        image = Image.open(full_img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

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

    # Create datasets
    train_dataset = MURADataset(train_image_csv, train_label_csv, train_dir, transform=get_transforms(train=True))
    valid_dataset = MURADataset(valid_image_csv, valid_label_csv, valid_dir, transform=get_transforms(train=False))

    # Count samples and classes for training data
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    num_train_samples = len(train_dataset)
    num_train_classes = len(set(train_labels))
    print(f"Found {num_train_samples} validated image filenames belonging to {num_train_classes} classes in the training set.")

    # Count samples and classes for validation data
    valid_labels = [valid_dataset[i][1] for i in range(len(valid_dataset))]
    num_valid_samples = len(valid_dataset)
    num_valid_classes = len(set(valid_labels))
    print(f"Found {num_valid_samples} validated image filenames belonging to {num_valid_classes} classes in the validation set.")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

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