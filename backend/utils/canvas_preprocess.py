"""
Robust Canvas Preprocessing for MNIST/EMNIST Compatibility

This module provides advanced preprocessing for canvas-drawn characters
to match the MNIST/EMNIST training data distribution exactly.

Uses OpenCV for robust image processing operations.
"""

import os
import base64
import numpy as np
import cv2


DEBUG_DIR = "debug_preprocessing"


def preprocess_canvas_image(
    image_bytes_or_base64,
    for_emnist: bool = False,
    debug: bool = False,
    debug_prefix: str = "canvas"
) -> np.ndarray:
    """
    Robust preprocessing for canvas drawings to match MNIST/EMNIST format.
    
    Pipeline:
    1. Decode base64 to image
    2. Convert to grayscale
    3. Invert colors (canvas has white-on-black, but we ensure consistency)
    4. Apply Otsu thresholding for clean binarization
    5. Find bounding box of drawn content
    6. Crop tightly around the character
    7. Make square with padding
    8. Add margin (MNIST has ~4px margin in 28x28)
    9. Center by center-of-mass
    10. Resize to 28x28 with anti-aliasing
    11. Apply EMNIST transpose if needed
    12. Normalize to [0, 1]
    
    Args:
        image_bytes_or_base64: Raw image bytes or base64 string
        for_emnist: If True, apply EMNIST-specific transpose
        debug: If True, save intermediate images to DEBUG_DIR
        debug_prefix: Prefix for debug image filenames
        
    Returns:
        Preprocessed image array of shape (1, 28, 28, 1)
    """
    # Setup debug directory
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # === Step 1: Decode input ===
    if isinstance(image_bytes_or_base64, str):
        if "base64," in image_bytes_or_base64:
            image_bytes_or_base64 = image_bytes_or_base64.split("base64,")[1]
        image_bytes = base64.b64decode(image_bytes_or_base64)
    else:
        image_bytes = image_bytes_or_base64
    
    # Convert bytes to numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Warning: Failed to decode image")
        return np.zeros((1, 28, 28, 1), dtype='float32')
    
    img = img.astype('float32')
    
    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_01_grayscale.png"), img)
    
    # === Step 2: Invert if light background ===
    # MNIST/EMNIST: white digit on black background (digit = high values)
    if np.mean(img) > 127:
        img = 255 - img
    
    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_02_inverted.png"), img)
    
    # === Step 3: Apply Otsu thresholding ===
    img_uint8 = img.astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_03_otsu.png"), binary)
    
    # === Step 4: Find bounding box using contours ===
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("Warning: No contours found in image")
        return np.zeros((1, 28, 28, 1), dtype='float32')
    
    # Get bounding box of all contours combined
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # === Step 5: Crop to bounding box (use original grayscale for quality) ===
    cropped = img[y_min:y_max, x_min:x_max]
    
    if cropped.size == 0:
        print("Warning: Empty crop region")
        return np.zeros((1, 28, 28, 1), dtype='float32')
    
    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_04_cropped.png"), cropped)
    
    # === Step 6: Make it square with padding ===
    h, w = cropped.shape
    max_dim = max(h, w)
    
    # Create square canvas (black background)
    square = np.zeros((max_dim, max_dim), dtype='float32')
    
    # Center the cropped image in the square
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_05_square.png"), square)
    
    # === Step 7: Add margin (MNIST-style: content occupies ~20x20 in 28x28) ===
    # Target: 20x20 content in 28x28 -> margin = 4px on each side
    # As ratio: margin = 4/28 ≈ 14% on each side, or ~28% added total
    target_size = 20  # Content will be scaled to this size first
    
    # Resize to target_size x target_size
    scaled = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Place in 28x28 with proper margins
    final_28 = np.zeros((28, 28), dtype='float32')
    margin = (28 - target_size) // 2  # = 4
    final_28[margin:margin+target_size, margin:margin+target_size] = scaled
    
    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_06_with_margin.png"), final_28)
    
    # === Step 8: Center by center-of-mass ===
    # Calculate center of mass
    M = cv2.moments(final_28)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculate shift to center (target: 14, 14 which is center of 28x28)
        shift_x = 14 - cx
        shift_y = 14 - cy
        
        # Apply translation
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        final_28 = cv2.warpAffine(
            final_28, 
            translation_matrix, 
            (28, 28),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    
    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_07_centered.png"), final_28)
    
    # === Step 9: No additional transform needed ===
    # EMNIST training data is transposed to natural orientation.
    # Canvas drawings are also in natural orientation, so they match.
    
    if debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_prefix}_08_final.png"), final_28)
    
    # === Step 10: Normalize to [0, 1] ===
    final_28 = final_28 / 255.0
    
    # === Step 11: Reshape for model input ===
    final_28 = final_28.reshape(1, 28, 28, 1).astype('float32')
    
    return final_28


def preprocess_canvas_for_digits(image_bytes_or_base64, debug: bool = False) -> np.ndarray:
    """
    Preprocess canvas drawing for MNIST digit recognition.
    
    Args:
        image_bytes_or_base64: Raw image bytes or base64 string
        debug: If True, save intermediate images
        
    Returns:
        Preprocessed image array of shape (1, 28, 28, 1)
    """
    return preprocess_canvas_image(
        image_bytes_or_base64,
        for_emnist=False,
        debug=debug,
        debug_prefix="digit"
    )


def preprocess_canvas_for_characters(image_bytes_or_base64, debug: bool = False) -> np.ndarray:
    """
    Preprocess canvas drawing for EMNIST character recognition.
    
    Args:
        image_bytes_or_base64: Raw image bytes or base64 string
        debug: If True, save intermediate images
        
    Returns:
        Preprocessed image array of shape (1, 28, 28, 1)
    """
    return preprocess_canvas_image(
        image_bytes_or_base64,
        for_emnist=True,
        debug=debug,
        debug_prefix="char"
    )
