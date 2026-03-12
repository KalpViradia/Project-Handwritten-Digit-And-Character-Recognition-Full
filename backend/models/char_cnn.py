"""
Character CNN Model for Handwritten Letter Recognition

CNN architecture for classifying handwritten letters (A-Z) from EMNIST.
Reuses the same architecture as digit CNN but with 26 output classes.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
    InputLayer, BatchNormalization
)
from tensorflow.keras.regularizers import l2


def create_char_cnn(input_shape=(28, 28, 1), num_classes=26):
    """
    Create a CNN model for character recognition.
    
    Architecture identical to digit CNN but optimized for 26 classes (A-Z).
    Character recognition is harder than digit recognition due to:
    - More classes (26 vs 10)
    - Greater similarity between some letters (O/Q, I/L, etc.)
    
    Architecture:
        Input (28,28,1)
        → Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
        → Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
        → Flatten → Dense(512) → BN → Dropout(0.5)
        → Dense(256) → Dropout(0.5)
        → Dense(26, Softmax)
    
    Note: Larger dense layers (512, 256) compared to digit model
    to handle increased complexity of 26 classes.
    
    Args:
        input_shape: Shape of input images (default: (28, 28, 1))
        num_classes: Number of output classes (default: 26 for A-Z)
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        InputLayer(input_shape=input_shape),
        
        # First Convolutional Block
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
               name='conv2d_1a'),
        BatchNormalization(name='bn_1a'),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
               name='conv2d_1b'),
        BatchNormalization(name='bn_1b'),
        MaxPooling2D(pool_size=(2, 2), name='maxpool_1'),
        Dropout(0.25, name='dropout_1'),
        
        # Second Convolutional Block
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
               name='conv2d_2a'),
        BatchNormalization(name='bn_2a'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
               name='conv2d_2b'),
        BatchNormalization(name='bn_2b'),
        MaxPooling2D(pool_size=(2, 2), name='maxpool_2'),
        Dropout(0.25, name='dropout_2'),
        
        # Third Convolutional Block (extra depth for character complexity)
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
               name='conv2d_3'),
        BatchNormalization(name='bn_3'),
        Dropout(0.25, name='dropout_3'),
        
        # Flatten and Dense layers
        Flatten(name='flatten'),
        
        Dense(512, activation='relu', kernel_regularizer=l2(0.001), name='dense_1'),
        BatchNormalization(name='bn_dense_1'),
        Dropout(0.5, name='dropout_dense_1'),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.001), name='dense_2'),
        Dropout(0.5, name='dropout_dense_2'),
        
        # Output layer - 26 classes for A-Z
        Dense(num_classes, activation='softmax', name='output')
    ], name='character_cnn')
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == '__main__':
    # Quick test
    print("Creating Character CNN Model...")
    model = create_char_cnn()
    model.summary()
    
    print("\n" + "="*50)
    print("Character CNN Model Created Successfully!")
    print(f"Total Parameters: {model.count_params():,}")
    print(f"Output Classes: 26 (A-Z)")
    print("="*50)
