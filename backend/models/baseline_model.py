"""
Baseline Fully-Connected Neural Network Model

A simple ANN model for comparison with CNN.
This demonstrates why CNN is more effective for image classification.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, InputLayer


def create_baseline_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a baseline fully-connected neural network.
    
    Architecture:
        Input (784) → Dense(128, ReLU) → Dropout(0.2) → 
        Dense(64, ReLU) → Dropout(0.2) → Dense(10, Softmax)
    
    Args:
        input_shape: Shape of input images (default: (28, 28, 1))
        num_classes: Number of output classes (default: 10)
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        InputLayer(input_shape=input_shape),
        
        # Flatten 2D image to 1D vector
        Flatten(),
        
        # First hidden layer
        Dense(128, activation='relu', name='dense_1'),
        Dropout(0.2, name='dropout_1'),
        
        # Second hidden layer
        Dense(64, activation='relu', name='dense_2'),
        Dropout(0.2, name='dropout_2'),
        
        # Output layer
        Dense(num_classes, activation='softmax', name='output')
    ], name='baseline_ann')
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model):
    """Print and return model summary."""
    model.summary()
    return model


if __name__ == '__main__':
    # Quick test
    model = create_baseline_model()
    model.summary()
    
    print("\n" + "="*50)
    print("Baseline Model Created Successfully!")
    print(f"Total Parameters: {model.count_params():,}")
    print("="*50)
