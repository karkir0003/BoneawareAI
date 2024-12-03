
import os
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, classification_report,
    precision_recall_fscore_support, cohen_kappa_score, roc_curve
)


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    """
    Plots the confusion matrix using Matplotlib.

    Args:
        cm (ndarray): Confusion matrix.
        classes (list): List of class labels.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_probs, title="ROC Curve"):
    """
    Plots the ROC Curve using Matplotlib.

    Args:
        y_true (list): True labels.
        y_probs (list): Predicted probabilities.
        title (str): Title of the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC: {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_class_weights(dataset):
    """
    Computes class weights for binary classification based on the dataset.

    Args:
        dataset (MURADataset): The training dataset.

    Returns:
        Tuple: (w_normal, w_abnormal)
    """
    # Extract labels efficiently using NumPy
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Use pandas for counting
    label_counts = pd.Series(labels).value_counts(normalize=False)
    total_samples = len(labels)

    # Compute weights
    w_normal = total_samples / (4 * label_counts[0])
    w_abnormal = total_samples / (4 * label_counts[1])

    print(f"Class weights: w_normal={w_normal}, w_abnormal={w_abnormal}")
    return w_normal, w_abnormal


def calculate_kappa_confidence_interval(y_true, y_pred, confidence=0.95):
    """
    Calculate Cohen's Kappa and its confidence interval.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        confidence (float): Confidence level (default 0.95).

    Returns:
        dict: Cohen's Kappa and its confidence interval.
    """
    from scipy.stats import norm

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Observed agreement
    po = np.mean(np.array(y_true) == np.array(y_pred))

    # Expected agreement
    confusion = confusion_matrix(y_true, y_pred)
    total = np.sum(confusion)
    pe = sum((confusion.sum(axis=0) / total) * (confusion.sum(axis=1) / total))

    # Standard error of kappa
    se_kappa = np.sqrt((po * (1 - po)) / len(y_true) +
                       (pe * (1 - pe)) / len(y_true))

    # Z-score for desired confidence level
    z = norm.ppf((1 + confidence) / 2)

    # Confidence interval
    lower_bound = kappa - z * se_kappa
    upper_bound = kappa + z * se_kappa

    return {
        "Cohen's Kappa": kappa,
        "95% CI Lower": lower_bound,
        "95% CI Upper": upper_bound
    }


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculates performance metrics using NumPy for efficiency.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        y_pred_proba (list): Predicted probabilities.

    Returns:
        dict: A dictionary of calculated metrics.
    """
    # Convert to NumPy arrays for fast operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "ROC-AUC": roc_auc,
        "Cohen's Kappa": kappa,
    }


def calculate_metrics_per_body_part(dataset, y_true, y_pred, y_pred_proba):
    """
    Calculates metrics per body part efficiently using Pandas.

    Args:
        dataset (MURADataset): The dataset object.
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        y_pred_proba (list): Predicted probabilities.

    Returns:
        dict: A dictionary of metrics per body part.
    """
    # Extract body parts
    body_parts = dataset.image_df["image_path"].apply(
        lambda path: path.split(
            "train/" if "train" in path else "valid/")[1].split("/")[0]
    )

    # Create a DataFrame for easier processing
    df = pd.DataFrame({
        "BodyPart": body_parts,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    })

    # Group by BodyPart and compute metrics
    results = {}
    for body_part, group in df.groupby("BodyPart"):
        metrics = calculate_metrics(
            group["y_true"].values,
            group["y_pred"].values,
            group["y_pred_proba"].values,
        )
        results[body_part] = metrics

    return results


def evaluate_model(model, loader, dataset=None, criterion=None):
    """
    Evaluate the model on a DataLoader and calculate all metrics, including metrics per body part.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): DataLoader for evaluation.
        dataset (MURADataset): Optional, the dataset to calculate per-body part metrics.

    Returns:
        pd.DataFrame: DataFrame summarizing the metrics.
    """
    model.eval()
    all_preds, all_labels, all_probs, all_losses = [], [], [], []
    device = 'mps' if torch.backends.mps.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs)  # Convert logits to probabilities
            preds = (probs > 0.5).float()  # Threshold at 0.5

            # Calculate loss if criterion is provided
            if criterion:
                loss = criterion(outputs, labels).item()
                all_losses.extend([loss] * len(labels))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_losses = np.array(all_losses) if all_losses else None

    # Calculate global metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary')
    specificity = confusion_matrix(all_labels, all_preds)[
        0, 0] / sum(confusion_matrix(all_labels, all_preds)[0])
    roc_auc = roc_auc_score(all_labels, all_probs)
    kappa_metrics = calculate_kappa_confidence_interval(all_labels, all_preds)

    # Global loss
    global_loss = all_losses.mean() if all_losses is not None else None

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        cm, classes=["Normal", "Abnormal"], title="Confusion Matrix")

    # ROC Curve
    plot_roc_curve(all_labels, all_probs, title="ROC Curve")

    # Global metrics
    global_metrics = {
        "Accuracy": np.mean(all_preds == all_labels),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "ROC-AUC": roc_auc,
        "Cohen's Kappa": kappa_metrics["Cohen's Kappa"],
        "Kappa 95% CI Lower": kappa_metrics["95% CI Lower"],
        "Kappa 95% CI Upper": kappa_metrics["95% CI Upper"],
        "Loss": global_loss
    }

    # Print Metrics and classification report
    print("Global Metrics:")
    print(global_metrics)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    # Per-body part metrics
    if dataset:
        body_parts = dataset.image_df["image_path"].apply(
            lambda path: path.split(
                "train/" if "train" in path else "valid/")[1].split("/")[0]
        )
        body_part_metrics = {}
        for body_part in body_parts.unique():
            indices = body_parts[body_parts == body_part].index
            part_labels = all_labels[indices]
            part_preds = all_preds[indices]
            part_probs = all_probs[indices]
            part_losses = all_losses[indices] if all_losses is not None else None

            precision, recall, f1, _ = precision_recall_fscore_support(
                part_labels, part_preds, average='binary')
            specificity = confusion_matrix(part_labels, part_preds)[
                0, 0] / sum(confusion_matrix(part_labels, part_preds)[0])
            roc_auc = roc_auc_score(part_labels, part_probs)
            kappa_metrics = calculate_kappa_confidence_interval(
                part_labels, part_preds)
            part_loss = part_losses.mean() if part_losses is not None else None

            body_part_metrics[body_part] = {
                "Accuracy": np.mean(part_preds == part_labels),
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Specificity": specificity,
                "ROC-AUC": roc_auc,
                "Cohen's Kappa": kappa_metrics["Cohen's Kappa"],
                "Kappa 95% CI Lower": kappa_metrics["95% CI Lower"],
                "Kappa 95% CI Upper": kappa_metrics["95% CI Upper"],
                "Loss": part_loss
            }

        body_part_df = pd.DataFrame(body_part_metrics).T
        print("Metrics per body part:")
        print(body_part_df)

    return pd.DataFrame([global_metrics])


# Combine Main and Body Part Model Ensembles for Metrics
def evaluate_ensemble(main_model, body_part_models, loader, dataset, device, class_names=["Normal", "Abnormal"]):
    """
    Evaluate the ensemble of main model and body part models and calculate metrics.

    Args:
        main_model (nn.Module): Main trained model.
        body_part_models (dict): Dictionary of body part-specific models.
        loader (DataLoader): DataLoader for evaluation.
        dataset (MURADataset): Dataset object for metrics.
        device (torch.device): Device for computation.
        class_names (list): List of class names.

    Returns:
        dict: Metrics for the ensemble and per body part.
    """
    # Step 1: Get Predictions
    main_model.eval()
    ensemble_preds, ensemble_labels, ensemble_probs = [], [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            # Main model predictions
            main_preds = torch.sigmoid(main_model(inputs))

            # Body part model predictions
            part_preds = []
            for model in body_part_models.values():
                part_preds.append(torch.sigmoid(model(inputs)))

            # Combine main and body part predictions (average ensemble)
            final_preds = (
                main_preds + torch.mean(torch.stack(part_preds), dim=0)) / 2
            ensemble_preds.extend((final_preds > 0.5).cpu().numpy())
            ensemble_probs.extend(final_preds.cpu().numpy())
            ensemble_labels.extend(labels.cpu().numpy())

    # Step 2: Compute Overall Metrics
    ensemble_metrics = calculate_metrics(
        ensemble_labels,
        ensemble_preds,
        ensemble_probs
    )

    print("\nGlobal Ensemble Metrics:")
    print(ensemble_metrics)

    # Step 3: Compute Metrics Per Body Part
    body_part_metrics = calculate_metrics_per_body_part(
        dataset, ensemble_labels, ensemble_preds, ensemble_probs
    )
    print("\nBody Part Metrics:")
    for part, metrics in body_part_metrics.items():
        print(f"{part}: {metrics}")

    # Step 4: Calculate Cohen's Kappa and CI
    kappa_ci = calculate_kappa_confidence_interval(
        ensemble_labels, ensemble_preds)
    print("\nCohen's Kappa and Confidence Interval:")
    print(kappa_ci)

    # Step 5: Plot Confusion Matrix
    cm = confusion_matrix(ensemble_labels, ensemble_preds)
    plot_confusion_matrix(cm, classes=class_names,
                          title="Ensemble Confusion Matrix")

    # Step 6: Plot ROC Curve
    plot_roc_curve(ensemble_labels, ensemble_probs, title="Ensemble ROC Curve")

    return ensemble_metrics, body_part_metrics, kappa_ci
