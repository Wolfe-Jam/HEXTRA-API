"""
Sacred-38 Processing Module - Background Removal Approach
Focuses on removing background first, then isolating garment
"""
import cv2
import numpy as np

def apply_sacred38(image: np.ndarray) -> np.ndarray:
    """
    Apply sacred-38 with background removal approach.
    
    Args:
        image: Input image as numpy array (BGR)
    
    Returns:
        Processed image with background removed (black)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use GrabCut algorithm for background removal
    # This is more sophisticated than simple thresholding
    mask = np.zeros(gray.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Define a rectangle around the subject (person)
    # Assumes person is roughly centered
    height, width = gray.shape
    rect = (int(width * 0.1), int(height * 0.05), 
            int(width * 0.8), int(height * 0.9))
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify the mask to get foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    
    # Apply additional threshold to ensure solid areas
    _, binary = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Convert to BGR
    result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    return result

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
