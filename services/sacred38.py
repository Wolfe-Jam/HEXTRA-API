"""
Sacred-38 Processing Module - Background Removal Approach
Focuses on removing background first, then isolating garment
"""
import cv2
import numpy as np

def apply_sacred38(image: np.ndarray) -> np.ndarray:
    """
    Apply sacred-38 processing - simple and effective.
    
    Args:
        image: Input image as numpy array (BGR)
    
    Returns:
        Processed image with white foreground, black background
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

def apply_sacred38_edge_based(image: np.ndarray) -> np.ndarray:
    """
    Edge-based approach for background removal.
    
    Args:
        image: Input image
    
    Returns:
        Processed image with background black
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 30, 100)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for foreground
    mask = np.zeros_like(gray)
    
    # Find large contours (likely person/garment)
    min_area = gray.shape[0] * gray.shape[1] * 0.05  # 5% of image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    
    # Fill internal gaps
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Apply mask to original to check
    result_check = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Use adaptive threshold on the masked area
    _, final = cv2.threshold(result_check, 1, 255, cv2.THRESH_BINARY)
    
    # Convert to BGR
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

def apply_sacred38_watershed(image: np.ndarray) -> np.ndarray:
    """
    Watershed algorithm approach for better segmentation.
    
    Args:
        image: Input image
    
    Returns:
        Segmented image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Create result mask
    result = np.zeros_like(gray)
    result[markers > 1] = 255  # Keep only foreground regions
    
    # Clean up
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
