"""
Training Pipeline for Handwritten Character Recognition

Trains a CNN on EMNIST Letters dataset (A-Z).
Reuses utilities from digit training pipeline.
"""

import os
import sys
import numpy as np
from datetime import datetime

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from utils.load_emnist import load_emnist_letters, CHAR_LABELS
from utils.visualization import (
    plot_sample_images, plot_class_distribution, plot_training_history,
    plot_confusion_matrix, print_classification_report, plot_predictions
)
from models.char_cnn import create_char_cnn


# Configuration - Enhanced for character recognition
CONFIG = {
    'batch_size': 128,
    'epochs': 40,  # More epochs for characters
    'validation_split': 0.2,
    'output_dir': 'outputs/characters',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'use_augmentation': True,
}


def setup_directories():
    """Create necessary directories for outputs."""
    # Get backend root directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for dir_name in ['outputs/characters', 'checkpoints', 'logs']:
        full_path = os.path.join(backend_dir, dir_name)
        os.makedirs(full_path, exist_ok=True)
    
    # Update CONFIG paths to be absolute
    CONFIG['output_dir'] = os.path.join(backend_dir, 'outputs', 'characters')
    CONFIG['checkpoint_dir'] = os.path.join(backend_dir, 'checkpoints')
    CONFIG['log_dir'] = os.path.join(backend_dir, 'logs')
    
    print(f"Output directories created in: {backend_dir}")


def get_data_augmentation():
    """
    Create ImageDataGenerator with stronger augmentation for characters.
    
    Characters need more aggressive augmentation than digits because:
    - More classes (26 vs 10)
    - More similar shapes (O vs Q, I vs L, etc.)
    - Canvas drawings have thicker strokes
    """
    return ImageDataGenerator(
        rotation_range=20,       # More rotation for letter variation
        width_shift_range=0.15,  # Allow more shifting
        height_shift_range=0.15,
        zoom_range=0.15,         # Simulate size variation
        shear_range=0.2,         # More shear for style variation
        fill_mode='nearest'
    )


def get_callbacks(model_name: str):
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(CONFIG['checkpoint_dir'], 
                                  f'{model_name}_best.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # More patience for characters
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(CONFIG['log_dir'], f'{model_name}_{timestamp}'),
            histogram_freq=1
        )
    ]
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, model_name: str,
                use_augmentation: bool = False):
    """Train a model with optional data augmentation."""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"Data Augmentation: {'ENABLED' if use_augmentation else 'DISABLED'}")
    print(f"{'='*60}")
    
    model.summary()
    
    callbacks = get_callbacks(model_name)
    
    if use_augmentation:
        datagen = get_data_augmentation()
        datagen.fit(X_train)
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=CONFIG['batch_size']),
            steps_per_epoch=len(X_train) // CONFIG['batch_size'],
            validation_data=(X_val, y_val),
            epochs=CONFIG['epochs'],
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
    
    return history


def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate model and generate reports."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}")
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    y_pred = model.predict(X_test, verbose=0)
    
    # Classification report with character labels
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    from sklearn.metrics import classification_report
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_true_labels, y_pred_labels, 
                                target_names=CHAR_LABELS))
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(CONFIG['output_dir'], f'{model_name}_confusion_matrix.png')
    )
    
    return test_accuracy


def main():
    """Main training pipeline for character recognition."""
    print("="*60)
    print("CHARACTER RECOGNITION - TRAINING PIPELINE (EMNIST)")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    setup_directories()
    
    # Load EMNIST
    X_train, X_val, X_test, y_train, y_val, y_test = load_emnist_letters(
        validation_split=CONFIG['validation_split']
    )
    
    # EDA
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    plot_sample_images(
        X_train, y_train, num_samples=25,
        save_path=os.path.join(CONFIG['output_dir'], 'sample_characters.png')
    )
    
    plot_class_distribution(
        y_train,
        save_path=os.path.join(CONFIG['output_dir'], 'character_distribution.png')
    )
    
    # Train Character CNN
    char_model = create_char_cnn()
    char_history = train_model(
        char_model, X_train, y_train, X_val, y_val, 'char_cnn',
        use_augmentation=CONFIG['use_augmentation']
    )
    
    plot_training_history(
        char_history,
        save_path=os.path.join(CONFIG['output_dir'], 'char_cnn_training_history.png')
    )
    
    test_accuracy = evaluate_model(char_model, X_test, y_test, 'char_cnn')
    
    # Save final model
    print("\n" + "="*60)
    print("SAVING FINAL MODEL")
    print("="*60)
    
    char_model.save(os.path.join(CONFIG['checkpoint_dir'], 'char_cnn_final.keras'))
    print(f"Model saved to {CONFIG['checkpoint_dir']}/char_cnn_final.keras")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\nCharacter CNN Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Classes: 26 (A-Z)")
    
    print("\n✅ Character training complete!")
    print(f"📁 Check '{CONFIG['output_dir']}/' for visualizations.")


if __name__ == '__main__':
    main()
