"""
Sacred-38 Processing Module
Advanced garment-focused image processing
"""
import cv2
import numpy as np

def apply_sacred38(image: np.ndarray) -> np.ndarray:
    """
    Apply the sacred-38 processing algorithm to an image.
    This creates an optimal black and white mask for garment detection.
    
    Args:
        image: Input image as numpy array (BGR)
    
    Returns:
        Processed image as numpy array (black and white mask)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use adaptive thresholding for better local contrast handling
    # This is the "sacred" part - adaptive with specific parameters
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        39,  # Block size (sacred number close to 38)
        5    # Constant subtracted from mean
    )
    
    # Apply morphological operations to clean up
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # Remove small noise
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_small)
    
    # Close small gaps
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
    
    # Additional edge refinement
    # Detect edges using Canny
    edges = cv2.Canny(bilateral, 50, 150)
    
    # Dilate edges slightly
    edges_dilated = cv2.dilate(edges, kernel_small, iterations=1)
    
    # Combine adaptive threshold with edge information
    # This helps preserve garment boundaries
    combined = cv2.bitwise_or(cleaned, edges_dilated)
    
    # Final cleanup
    final = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_small)
    
    # Convert back to BGR for consistency
    result = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    
    return result

def apply_sacred38_advanced(image: np.ndarray, params: dict = None) -> np.ndarray:
    """
    Advanced sacred-38 processing with configurable parameters.
    
    Args:
        image: Input image
        params: Dictionary of processing parameters
    
    Returns:
        Processed image
    """
    if params is None:
        params = {
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'adaptive_block_size': 39,
            'adaptive_c': 5,
            'canny_low': 50,
            'canny_high': 150
        }
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter
    bilateral = cv2.bilateralFilter(
        gray, 
        params['bilateral_d'],
        params['bilateral_sigma_color'],
        params['bilateral_sigma_space']
    )
    
    # Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        params['adaptive_block_size'],
        params['adaptive_c']
    )
    
    # Edge detection
    edges = cv2.Canny(
        bilateral,
        params['canny_low'],
        params['canny_high']
    )
    
    # Combine and clean
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    combined = cv2.bitwise_or(adaptive_thresh, edges_dilated)
    
    # Morphological cleanup
    final = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
    
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
