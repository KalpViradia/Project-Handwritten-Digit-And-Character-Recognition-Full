"""
Visualization Module for Handwritten Digit Recognition

This module provides visualization functions for EDA, training progress,
and model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


def ensure_output_dir(output_dir: str = 'outputs'):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_sample_images(X: np.ndarray, y: np.ndarray, num_samples: int = 25, 
                       save_path: str = None):
    """
    Display a grid of sample digit images.
    
    Args:
        X: Image array of shape (n, 28, 28) or (n, 28, 28, 1)
        y: Label array (can be one-hot encoded or integer)
        num_samples: Number of samples to display
        save_path: Optional path to save the figure
    """
    # Handle different input shapes
    if len(X.shape) == 4:
        X = X.squeeze()
    
    # Convert one-hot to integer labels if necessary
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle('Sample MNIST Digits', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples and i < len(X):
            ax.imshow(X[i], cmap='gray')
            ax.set_title(f'Label: {y[i]}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample images to {save_path}")
    
    plt.show()


def plot_class_distribution(y: np.ndarray, save_path: str = None):
    """
    Plot the distribution of digit classes.
    
    Args:
        y: Label array (can be one-hot encoded or integer)
        save_path: Optional path to save the figure
    """
    # Convert one-hot to integer labels if necessary
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    
    unique, counts = np.unique(y, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique)))
    bars = ax.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Digit Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution in MNIST Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(unique)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path: str = None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras History object from model.fit()
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', 
                 linewidth=2, color='#2ecc71')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy',
                 linewidth=2, color='#e74c3c', linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss',
                 linewidth=2, color='#2ecc71')
    axes[1].plot(history.history['val_loss'], label='Validation Loss',
                 linewidth=2, color='#e74c3c', linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          save_path: str = None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels (integer or one-hot)
        y_pred: Predicted labels (integer or one-hot/probabilities)
        save_path: Optional path to save the figure
    """
    # Convert to integer labels if necessary
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()
    
    return cm


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    # Convert to integer labels if necessary
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, 
                                target_names=[f'Digit {i}' for i in range(10)]))


def plot_predictions(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                     num_samples: int = 16, show_errors_only: bool = False,
                     save_path: str = None):
    """
    Display predictions with actual vs predicted labels.
    
    Args:
        X: Image array
        y_true: True labels
        y_pred: Predicted probabilities or labels
        num_samples: Number of samples to display
        show_errors_only: If True, only show misclassified samples
        save_path: Optional path to save the figure
    """
    # Handle different input shapes
    if len(X.shape) == 4:
        X = X.squeeze()
    
    # Convert to integer labels
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_pred_probs = np.max(y_pred, axis=1)
    else:
        y_pred_labels = y_pred
        y_pred_probs = None
    
    if show_errors_only:
        # Get indices of misclassified samples
        error_indices = np.where(y_true != y_pred_labels)[0]
        if len(error_indices) == 0:
            print("No misclassified samples found!")
            return
        indices = error_indices[:num_samples]
        title = 'Misclassified Samples'
    else:
        indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
        title = 'Sample Predictions'
    
    grid_size = int(np.ceil(np.sqrt(len(indices))))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(indices):
            idx = indices[i]
            ax.imshow(X[idx], cmap='gray')
            
            true_label = y_true[idx]
            pred_label = y_pred_labels[idx]
            
            if y_pred_probs is not None:
                conf = y_pred_probs[idx] * 100
                label_text = f'True: {true_label}\nPred: {pred_label} ({conf:.1f}%)'
            else:
                label_text = f'True: {true_label}\nPred: {pred_label}'
            
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(label_text, fontsize=9, color=color)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved predictions to {save_path}")
    
    plt.show()
