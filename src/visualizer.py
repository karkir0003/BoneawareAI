import json
import urllib.request

#import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
#from pyxtend import struct
from torchvision.models.resnet import ResNet18_Weights
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader



def find_last_conv_layer(model: nn.Module) -> tuple:
    last_conv_layer_name = None
    last_conv_layer = None

    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer_name = layer_name
            last_conv_layer = layer

    return last_conv_layer_name, last_conv_layer


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks for gradients and activations
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.full_backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def full_backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_heatmap(self, input_batch, class_idx=None):
        # Forward pass
        logits = self.model(input_batch)  # Shape [batch_size] for binary classification
        self.model.zero_grad()

        # Adjust for binary classification
        if len(logits.shape) == 1:  # Binary case: logits.shape = [batch_size]
            logits = logits.unsqueeze(1)  # Make it [batch_size, 1]

        # Compute probabilities
        probs = torch.sigmoid(logits) if logits.size(1) == 1 else torch.softmax(logits, dim=1)

        # Determine the predicted class if class_idx is not specified
        if class_idx is None:
            if logits.size(1) == 1:  # Binary classification
                predicted_prob = probs[0].item()  # Single probability for positive class
                class_idx = 1 if predicted_prob >= 0.5 else 0
            else:  # Multi-class classification
                class_idx = torch.argmax(probs, dim=1).item()
                predicted_prob = probs[0, class_idx].item()


        # Compute gradients for the target class
        one_hot_output = torch.zeros_like(logits)
        if logits.size(1) == 1:  # Binary classification case
            one_hot_output[:, 0] = 1  # Always compute gradients for the positive class
        else:
            one_hot_output[:, class_idx] = 1

        logits.backward(gradient=one_hot_output)

        # Compute Grad-CAM heatmap
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)  # ReLU removes negative values

        # Upsample the heatmap to match the input image size
        heatmap = torch.nn.functional.interpolate(
            heatmap, size=(input_batch.shape[2], input_batch.shape[3]), mode='bilinear', align_corners=False
        )
        heatmap = heatmap.squeeze().cpu().numpy()

        # Normalize the heatmap
        heatmap -= heatmap.min()
        heatmap /= heatmap.max() if heatmap.max() > 0 else 1  # Avoid division by zero

        return heatmap, class_idx, predicted_prob





def overlay_heatmap_on_image(image, heatmap, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap on the input image with a color map.

    Args:
        image (torch.Tensor): Input image tensor (C, H, W) in the range [0, 1].
        heatmap (np.ndarray): Grad-CAM heatmap array (H, W).
        alpha (float): Opacity of the heatmap overlay.

    Returns:
        PIL.Image: Image with heatmap overlay.
    """
    # Normalize heatmap to [0, 1]
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = plt.cm.jet(heatmap)[:, :, :3]  # Apply color map and keep RGB channels only
    heatmap_color = np.uint8(heatmap_color * 255)  # Convert to [0, 255]

    # Resize heatmap to match the input image
    heatmap_pil = Image.fromarray(heatmap_color).resize(
        (image.shape[2], image.shape[1]), resample=Image.BILINEAR
    )

    # Convert input image tensor to PIL image
    image_pil = to_pil_image(image)

    # Overlay heatmap on the image
    overlay = Image.blend(image_pil, heatmap_pil, alpha)
    return overlay

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor that was normalized using given mean and std.

    Args:
        tensor (torch.Tensor): Normalized tensor of shape (C, H, W).
        mean (list): Mean values used for normalization.
        std (list): Standard deviation values used for normalization.

    Returns:
        torch.Tensor: Denormalized tensor of shape (C, H, W).
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def run_gradcam(model, dataloader, target_layer, class_names, device='cuda', num_images=5):
    """
    Run Grad-CAM for an example batch from the data loader.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for fetching samples.
        target_layer (torch.nn.Module): Target layer for Grad-CAM.
        class_names (list): List of class names.
        device (str): Device to run the model.

    Returns:
        None
    """
    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Fetch a single batch
    inputs, labels = next(iter(dataloader))

    # Dynamically fetch paths from dataset if available
    dataset = dataloader.dataset
    paths = dataset.image_df.iloc[:len(inputs)]["image_path"].values  # Fetch paths for the batch

    # Clone original images to ensure integrity
    original_images = inputs.detach().cpu().clone()  # Clone the input tensors for original image storage
    inputs, labels = inputs.to(device), labels.to(device)

    # Define the mean and std used in preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Generate Grad-CAM heatmaps for the batch
    for i in range(min(len(inputs), num_images)):  # Process up to `n` images
        input_image = inputs[i].unsqueeze(0)  # Add batch dimension
        label = labels[i].item()
        image_path = paths[i]  # Get the image path


        # Extract and denormalize the original image
        original_image_tensor = original_images[i]  # Fetch the cloned original tensor
        denormalized_image_tensor = denormalize(original_image_tensor, mean, std)  # Denormalize the image
        original_image = to_pil_image(denormalized_image_tensor.clamp(0, 1))  # Convert to PIL format

        # Compute heatmap
        heatmap, class_idx, predicted_prob = gradcam.compute_heatmap(input_image)

        # Overlay heatmap on the original image
        overlay = overlay_heatmap_on_image(denormalized_image_tensor, heatmap)

        # Plot original image, heatmap, and overlay
        plt.figure(figsize=(12, 4))

        # Display the original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)  # Use the original image variable
        plt.title(f"Original Image\nActual: {class_names[label]}")
        plt.axis("off")

        # Display the heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title(f"Heatmap\nClass: {class_names[class_idx]}\nProb: {predicted_prob:.4f}")
        plt.axis("off")

        # Display the overlay
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        # Add image path below the figure
        plt.gcf().text(0.5, 0.02, f"Image Path: {image_path}", ha='center', fontsize=10)

        plt.show()



def run_gradcam_filtered(model, dataloader, target_layer, class_names, body_part=None, n=5, device='cuda'):
    """
    Run Grad-CAM for up to `n` images filtered by a specific body part.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for fetching samples.
        target_layer (torch.nn.Module): Target layer for Grad-CAM.
        class_names (list): List of class names.
        body_part (str): Specific body part to filter (e.g., "XR_ELBOW").
        n (int): Number of images to process (default: 5).
        device (str): Device to run the model.

    Returns:
        None
    """
    # Filter the dataset by body part if provided
    if body_part and hasattr(dataloader.dataset, "image_df"):
        filtered_df = dataloader.dataset.image_df[
            dataloader.dataset.image_df["image_path"].str.contains(body_part)
        ]
        filtered_indices = filtered_df.index

        if len(filtered_indices) == 0:
            print(f"No images found for body part: {body_part}")
            return

        # Limit the filtered indices to `n` samples
        filtered_indices = filtered_indices[:n]
    else:
        # Default to the first `n` samples if no body part is specified
        filtered_indices = range(min(len(dataloader.dataset), n))

    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)

    # Define the mean and std used in preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Process filtered indices
    for idx in filtered_indices:
        image, label = dataloader.dataset[idx]
        input_image = image.unsqueeze(0).to(device)
        label = int(label)

        # Extract image path
        image_path = dataloader.dataset.image_df.iloc[idx]["image_path"]

        # Clone the original image and denormalize it
        denormalized_image_tensor = denormalize(image, mean, std)
        original_image = to_pil_image(denormalized_image_tensor.clamp(0, 1))

        # Compute heatmap
        heatmap, class_idx, predicted_prob = gradcam.compute_heatmap(input_image)

        # Overlay heatmap on the original image
        overlay = overlay_heatmap_on_image(denormalized_image_tensor, heatmap)

        # Plot original image, heatmap, and overlay
        plt.figure(figsize=(12, 4))

        # Display the original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title(f"Original Image\nActual: {class_names[label]}")
        plt.axis("off")

        # Display the heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title(f"Heatmap\nClass: {class_names[class_idx]}\nProb: {predicted_prob:.4f}")
        plt.axis("off")

        # Display the overlay
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        # Add image path below the figure
        plt.gcf().text(0.5, 0.02, f"Image Path: {image_path}", ha='center', fontsize=10)

        plt.show()


def run_gradcam_for_path_person_or_bodypart(
    model,
    dataloader,
    target_layer,
    class_names,
    image_path=None,
    person_id=None,
    body_part=None,
    device="cuda",
):
    """
    Run Grad-CAM for a specific image path, all images for a specific person, 
    or a specific body part for a person.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for fetching samples.
        target_layer (torch.nn.Module): Target layer for Grad-CAM.
        class_names (list): List of class names.
        image_path (str): Specific path of the image to analyze.
        person_id (str): ID of the person (e.g., patient ID).
        body_part (str): Body part to filter for (e.g., "XR_WRIST").
        device (str): Device to run the model.

    Returns:
        None
    """
    # Filter dataset based on input criteria
    if image_path and hasattr(dataloader.dataset, "image_df"):
        filtered_df = dataloader.dataset.image_df[
            dataloader.dataset.image_df["image_path"] == image_path
        ]
    elif person_id and hasattr(dataloader.dataset, "image_df"):
        filtered_df = dataloader.dataset.image_df[
            dataloader.dataset.image_df["image_path"].str.contains(person_id)
        ]
        if body_part:  # Further filter by body part if specified
            filtered_df = filtered_df[
                filtered_df["image_path"].str.contains(body_part)
            ]
    else:
        raise ValueError(
            "You must provide either `image_path`, `person_id`, or both `person_id` and `body_part`."
        )

    # Check if any matches were found
    if filtered_df.empty:
        print("No matching images found for the specified criteria.")
        return

    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)

    # Define the mean and std used in preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Process each filtered image
    for idx in filtered_df.index:
        image, label = dataloader.dataset[idx]
        input_image = image.unsqueeze(0).to(device)
        label = int(label)

        # Extract image path
        image_path = dataloader.dataset.image_df.iloc[idx]["image_path"]

        # Clone the original image and denormalize it
        denormalized_image_tensor = denormalize(image, mean, std)
        original_image = to_pil_image(denormalized_image_tensor.clamp(0, 1))

        # Compute heatmap
        heatmap, class_idx, predicted_prob = gradcam.compute_heatmap(input_image)

        # Overlay heatmap on the original image
        overlay = overlay_heatmap_on_image(denormalized_image_tensor, heatmap)

        # Plot original image, heatmap, and overlay
        plt.figure(figsize=(12, 4))

        # Display the original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title(f"Original Image\nActual: {class_names[label]}")
        plt.axis("off")

        # Display the heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title(f"Heatmap\nClass: {class_names[class_idx]}\nProb: {predicted_prob:.4f}")
        plt.axis("off")

        # Display the overlay
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        # Add image path below the figure
        plt.gcf().text(0.5, 0.02, f"Image Path: {image_path}", ha="center", fontsize=10)

        plt.show()

