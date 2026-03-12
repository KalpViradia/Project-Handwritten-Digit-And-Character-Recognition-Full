"""
Data Preprocessing Module for Handwritten Digit Recognition

This module handles loading, preprocessing, and preparing the MNIST dataset
for training CNN and baseline models.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import io


def load_and_preprocess_data(validation_split: float = 0.2):
    """
    Load MNIST dataset and preprocess for training.
    
    Steps:
    1. Load MNIST from Keras datasets
    2. Normalize pixel values to [0, 1]
    3. Reshape to (28, 28, 1) for CNN
    4. One-hot encode labels
    5. Split training data into train/validation sets
    
    Args:
        validation_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("Loading MNIST dataset...")
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    
    print(f"Original training samples: {X_train_full.shape[0]}")
    print(f"Original test samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train_full.shape[1:]} -> (28, 28)")
    
    # Normalize pixel values to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension for CNN: (28, 28) -> (28, 28, 1)
    X_train_full = X_train_full.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    y_train_full_cat = to_categorical(y_train_full, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full_cat,
        test_size=validation_split,
        random_state=42,
        stratify=y_train_full
    )
    
    print(f"\nAfter preprocessing:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Output classes: {y_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test_cat


def get_raw_data():
    """
    Get raw MNIST data without preprocessing for EDA.
    
    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    return mnist.load_data()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess a single image for inference.
    
    Handles images from file upload or canvas drawing.
    Converts to grayscale, resizes to 28x28, normalizes, and reshapes.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Preprocessed image array of shape (1, 28, 28, 1)
    """
    # Open image and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Invert if necessary (MNIST has white digits on black background)
    # Check if background is light (image needs inversion)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model input: (28, 28) -> (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array


def preprocess_for_baseline(X: np.ndarray) -> np.ndarray:
    """
    Flatten images for baseline fully-connected model.
    
    Args:
        X: Input array of shape (n, 28, 28, 1)
        
    Returns:
        Flattened array of shape (n, 784)
    """
    return X.reshape(X.shape[0], -1)
