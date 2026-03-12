"""
EMNIST Dataset Loading and Preprocessing Module

Handles loading the EMNIST Letters dataset for character recognition.
EMNIST Letters contains 26 classes (A-Z) in handwritten form.
"""

import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Character mapping: EMNIST letters are labeled 1-26, we map to 0-25
# Index 0 = 'A', Index 25 = 'Z'
CHAR_LABELS = [chr(ord('A') + i) for i in range(26)]


def load_emnist_letters(validation_split: float = 0.2):
    """
    Load and preprocess EMNIST Letters dataset.
    
    EMNIST Letters contains:
    - 88,800 training samples
    - 14,800 test samples
    - 26 classes (uppercase letters A-Z)
    - 28x28 grayscale images
    
    Note: EMNIST images need to be transposed and flipped to match 
    the orientation of MNIST digits.
    
    Args:
        validation_split: Fraction of training data for validation
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("Loading EMNIST Letters dataset...")
    print("(This may take a few minutes on first run to download)")
    
    # Load EMNIST letters split
    ds_train, ds_test = tfds.load(
        'emnist/letters',
        split=['train', 'test'],
        as_supervised=True,
        with_info=False
    )
    
    # Convert to numpy arrays
    X_train_full = []
    y_train_full = []
    for image, label in tfds.as_numpy(ds_train):
        X_train_full.append(image)
        y_train_full.append(label)
    
    X_test = []
    y_test = []
    for image, label in tfds.as_numpy(ds_test):
        X_test.append(image)
        y_test.append(label)
    
    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Original training samples: {X_train_full.shape[0]}")
    print(f"Original test samples: {X_test.shape[0]}")
    print(f"Original shape: {X_train_full.shape}")
    
    # EMNIST images from tfds have shape (N, 28, 28, 1) - need to transpose
    # to fix the orientation (letters appear rotated/mirrored)
    # Squeeze channel dim, transpose, then add back
    if len(X_train_full.shape) == 4:
        # Remove channel dim, transpose axes 1 and 2, add channel back
        X_train_full = np.transpose(X_train_full[:, :, :, 0], (0, 2, 1))
        X_test = np.transpose(X_test[:, :, :, 0], (0, 2, 1))
    elif len(X_train_full.shape) == 3:
        # Shape is (N, 28, 28) - just transpose
        X_train_full = np.transpose(X_train_full, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))
    
    # Normalize pixel values to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    X_train_full = X_train_full.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # EMNIST labels are 1-26, convert to 0-25
    y_train_full = y_train_full - 1
    y_test = y_test - 1
    
    # One-hot encode labels (26 classes)
    y_train_full_cat = to_categorical(y_train_full, 26)
    y_test_cat = to_categorical(y_test, 26)
    
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
    print(f"Output classes: {y_train.shape[1]} (A-Z)")
    print(f"Label mapping: 0='A', 1='B', ..., 25='Z'")
    
    return X_train, X_val, X_test, y_train, y_val, y_test_cat


def index_to_char(index: int) -> str:
    """Convert prediction index to character."""
    if 0 <= index < 26:
        return CHAR_LABELS[index]
    return '?'


def get_character_labels() -> list:
    """Get list of character labels."""
    return CHAR_LABELS.copy()


if __name__ == '__main__':
    # Quick test
    print("Testing EMNIST loading...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_emnist_letters()
    print(f"\nSuccess! Loaded {len(X_train)} training samples.")
    print(f"Sample shape: {X_train[0].shape}")
    print(f"Label shape: {y_train[0].shape}")
