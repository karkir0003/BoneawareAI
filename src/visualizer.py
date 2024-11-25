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




preprocess = transforms.Compose(
    [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

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

        #print(f"[DEBUG] Class Index: {class_idx}, Predicted Probability: {predicted_prob}")

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


def run_gradcam(model, dataloader, target_layer, class_names, device='cuda'):
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
    #print(f"[DEBUG] Batch Inputs Shape: {inputs.shape}, Labels: {labels}")
    
    # Clone original images to ensure integrity
    original_images = inputs.detach().cpu().clone()  # Clone the input tensors for original image storage
    inputs, labels = inputs.to(device), labels.to(device)

    # Define the mean and std used in preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Generate Grad-CAM heatmaps for the batch
    for i in range(len(inputs)):
        input_image = inputs[i].unsqueeze(0)  # Add batch dimension
        label = labels[i].item()

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

        plt.show()

        # Optional: Save the original image for debugging
        #original_image.save(f"original_image_{i}.png")
        break


def visualize_gradcam_examples(
    model, dataset, num_examples=5, dataset_type="valid", 
    body_part=None, patient_name=None
):
    """
    Visualize Grad-CAM examples for a model and dataset.

    Args:
        model (nn.Module): The trained model.
        dataset (Dataset): Dataset object (train/validation).
        gradcam (GradCam): Grad-CAM instance.
        num_examples (int): Number of examples to display.
        dataset_type (str): "train" or "validation".
        body_part (str): Filter by body part type.
        patient_name (str): Filter by patient name.
    """
    # Filter dataset by dataset_type
    dataset_filtered = [
        item for item in dataset if dataset_type in item['image_path']
    ]
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    # Further filter by body part
    if body_part:
        dataset_filtered = [
            item for item in dataset_filtered 
            if body_part in item['image_path']
        ]
    
    # Further filter by patient name
    if patient_name:
        dataset_filtered = [
            item for item in dataset_filtered 
            if patient_name in item['image_path']
        ]

    # Select up to `num_examples`
    examples = dataset_filtered[:num_examples]

    # Visualize examples
    fig, axes = plt.subplots(len(examples), 4, figsize=(15, 5 * len(examples)))

    for i, example in enumerate(examples):
        image, label = example['image'], example['label']
        img_tensor = image.unsqueeze(0).to(device)

        # Generate heatmap using Grad-CAM
        heatmap = GradCAM.compute_heatmap(img_tensor)
        heatmap_resized = np.uint8(255 * heatmap)
        
        # Overlay heatmap on original image
        overlay = to_pil_image(
            (image.cpu() * 0.5 + torch.tensor(heatmap_resized).float() / 255).clamp(0, 1)
        )
        
        # Display images
        axes[i, 0].imshow(to_pil_image(image.cpu()))
        axes[i, 0].set_title(f"Actual: {label}")
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Heatmap Overlay")
        axes[i, 3].imshow(to_pil_image(image.cpu()))
        axes[i, 3].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[i, 3].set_title("Blended View")
        
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
# Assume `model`, `train_dataset`, `val_dataset`, and `gradcam` are defined
# visualize_gradcam_examples(
#     model=model, 
#     dataset=val_dataset, 
#     gradcam=gradcam, 
#     num_examples=5, 
#     dataset_type="validation", 
#     body_part="hand", 
#     patient_name="patient123"
# )