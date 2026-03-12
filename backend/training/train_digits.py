"""
Training Pipeline for Handwritten Digit Recognition

This script trains and evaluates both baseline ANN and enhanced CNN models on MNIST.
Includes data augmentation, logging, checkpointing, and visualization of results.

Run from backend directory: python training/train_digits.py
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Local imports
from utils.preprocess import load_and_preprocess_data
from utils.visualization import (
    plot_sample_images, plot_class_distribution, plot_training_history,
    plot_confusion_matrix, print_classification_report, plot_predictions
)
from models.baseline_model import create_baseline_model
from models.cnn_model import create_cnn_model


# Configuration - Enhanced for better accuracy
CONFIG = {
    'batch_size': 128,
    'epochs': 30,
    'validation_split': 0.2,
    'output_dir': 'outputs',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'use_augmentation': True,
}


def setup_directories():
    """Create necessary directories for outputs."""
    # Get backend root directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for dir_name in [CONFIG['output_dir'], CONFIG['checkpoint_dir'], CONFIG['log_dir']]:
        full_path = os.path.join(backend_dir, dir_name)
        os.makedirs(full_path, exist_ok=True)
    
    # Update CONFIG paths to be absolute
    CONFIG['output_dir'] = os.path.join(backend_dir, 'outputs')
    CONFIG['checkpoint_dir'] = os.path.join(backend_dir, 'checkpoints')
    CONFIG['log_dir'] = os.path.join(backend_dir, 'logs')
    
    print(f"Output directories created in: {backend_dir}")


def get_data_augmentation():
    """
    Create ImageDataGenerator for data augmentation.
    
    Augmentation techniques:
    - Rotation: ±10 degrees (simulates tilted writing)
    - Width/Height shift: ±10% (simulates position variation)
    - Zoom: ±10% (simulates size variation)
    - Shear: slight shearing for natural handwriting variation
    """
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )


def get_callbacks(model_name: str):
    """
    Create training callbacks for a model.
    
    Args:
        model_name: Name of the model for file naming
        
    Returns:
        List of Keras callbacks
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=os.path.join(CONFIG['checkpoint_dir'], 
                                  f'{model_name}_best.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join(CONFIG['log_dir'], f'{model_name}_{timestamp}'),
            histogram_freq=1
        )
    ]
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, model_name: str,
                use_augmentation: bool = False):
    """
    Train a model with callbacks and optional data augmentation.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_name: Name for logging and checkpoints
        use_augmentation: Whether to use data augmentation
        
    Returns:
        Keras History object
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"Data Augmentation: {'ENABLED' if use_augmentation else 'DISABLED'}")
    print(f"{'='*60}")
    
    model.summary()
    
    callbacks = get_callbacks(model_name)
    
    if use_augmentation:
        datagen = get_data_augmentation()
        datagen.fit(X_train)
        
        # Train with augmented data
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
    """
    Evaluate model and generate reports.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
        model_name: Name for file naming
        
    Returns:
        Test accuracy
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}")
    
    # Get test accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Print classification report
    print_classification_report(y_test, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(CONFIG['output_dir'], f'{model_name}_confusion_matrix.png')
    )
    
    # Plot sample predictions
    plot_predictions(
        X_test, y_test, y_pred, num_samples=16,
        save_path=os.path.join(CONFIG['output_dir'], f'{model_name}_predictions.png')
    )
    
    # Plot misclassified samples
    plot_predictions(
        X_test, y_test, y_pred, num_samples=16, show_errors_only=True,
        save_path=os.path.join(CONFIG['output_dir'], f'{model_name}_errors.png')
    )
    
    return test_accuracy


def main():
    """Main training pipeline."""
    print("="*60)
    print("HANDWRITTEN DIGIT RECOGNITION - ENHANCED TRAINING PIPELINE")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"Data Augmentation: {'ENABLED' if CONFIG['use_augmentation'] else 'DISABLED'}")
    
    # Setup
    setup_directories()
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(
        validation_split=CONFIG['validation_split']
    )
    
    # Exploratory Data Analysis
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    plot_sample_images(
        X_train, y_train, num_samples=25,
        save_path=os.path.join(CONFIG['output_dir'], 'sample_images.png')
    )
    
    plot_class_distribution(
        y_train,
        save_path=os.path.join(CONFIG['output_dir'], 'class_distribution.png')
    )
    
    # Results storage
    results = {}
    
    # Train Baseline Model (no augmentation for baseline)
    baseline_model = create_baseline_model()
    baseline_history = train_model(
        baseline_model, X_train, y_train, X_val, y_val, 'baseline',
        use_augmentation=False
    )
    plot_training_history(
        baseline_history,
        save_path=os.path.join(CONFIG['output_dir'], 'baseline_training_history.png')
    )
    results['baseline'] = evaluate_model(baseline_model, X_test, y_test, 'baseline')
    
    # Train Enhanced CNN Model with data augmentation
    cnn_model = create_cnn_model()
    cnn_history = train_model(
        cnn_model, X_train, y_train, X_val, y_val, 'cnn',
        use_augmentation=CONFIG['use_augmentation']
    )
    plot_training_history(
        cnn_history,
        save_path=os.path.join(CONFIG['output_dir'], 'cnn_training_history.png')
    )
    results['cnn'] = evaluate_model(cnn_model, X_test, y_test, 'cnn')
    
    # Save final models
    print("\n" + "="*60)
    print("SAVING FINAL MODELS")
    print("="*60)
    
    baseline_model.save(os.path.join(CONFIG['checkpoint_dir'], 'baseline_final.keras'))
    cnn_model.save(os.path.join(CONFIG['checkpoint_dir'], 'cnn_final.keras'))
    
    print(f"Models saved to {CONFIG['checkpoint_dir']}/")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'Test Accuracy':<15}")
    print("-"*35)
    for model_name, accuracy in results.items():
        print(f"{model_name:<20} {accuracy*100:.2f}%")
    
    print(f"\nCNN improvement over baseline: "
          f"{(results['cnn'] - results['baseline'])*100:.2f}%")
    
    if results['cnn'] >= 0.994:
        print("\n🎉 Achieved 99.4%+ accuracy target!")
    
    print("\n✅ Training complete! Check 'outputs/' for visualizations.")
    print(f"📁 Models saved in '{CONFIG['checkpoint_dir']}/'")


if __name__ == '__main__':
    main()
