"""
Sacred-38 Processing Module
Creates solid black/white masks for garment detection
"""
import cv2
import numpy as np

def apply_sacred38(image: np.ndarray) -> np.ndarray:
    """
    Apply the sacred-38 processing algorithm to an image.
    Creates a solid black and white mask optimized for garment detection.
    
    Args:
        image: Input image as numpy array (BGR)
    
    Returns:
        Processed image as numpy array (black and white mask)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use OTSU thresholding for automatic threshold selection
    # This creates solid white/black areas
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply adaptive thresholding for better local contrast
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,  # Block size
        3    # Constant
    )
    
    # Combine OTSU and adaptive results
    # OTSU gives us solid areas, adaptive gives us details
    combined = cv2.bitwise_and(otsu_thresh, adaptive)
    
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    
    # Close gaps to create more solid areas
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small noise
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill internal holes
    # Find contours and fill them
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(cleaned)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    
    # Final cleanup
    final = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Convert back to BGR for consistency
    result = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    
    return result

def apply_sacred38_simple(image: np.ndarray) -> np.ndarray:
    """
    Simpler sacred-38 processing for more reliable results.
    
    Args:
        image: Input image
    
    Returns:
        Processed image with solid black/white areas
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while smoothing
    bilateral = cv2.bilateralFilter(gray, 11, 50, 50)
    
    # Use OTSU's method for automatic thresholding
    _, binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((7, 7), np.uint8)
    
    # Close gaps
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove noise
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Convert to BGR
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
