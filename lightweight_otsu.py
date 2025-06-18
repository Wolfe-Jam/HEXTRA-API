# Alternative 1: Using Pillow + NumPy only (no OpenCV)
import numpy as np
from PIL import Image

def apply_otsu_without_opencv(image_array):
    """
    Pure Python/NumPy implementation of OTSU threshold
    No OpenCV dependency needed!
    """
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        # Simple grayscale conversion
        image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Flatten the image
    pixel_values = image_array.flatten()
    
    # Calculate histogram
    histogram, _ = np.histogram(pixel_values, bins=256, range=(0, 256))
    
    # Total pixels
    total = len(pixel_values)
    
    # Find optimal threshold using OTSU's method
    current_max, threshold = 0, 0
    sum_total = np.dot(range(256), histogram)
    
    weight_background = 0
    sum_background = 0
    
    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue
            
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += i * histogram[i]
        
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Calculate between-class variance
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # Update threshold if variance is larger
        if variance > current_max:
            current_max = variance
            threshold = i
    
    # Apply threshold
    binary_image = (image_array > threshold).astype(np.uint8) * 255
    
    return binary_image

# Alternative 2: Using scikit-image (much smaller than OpenCV)
# from skimage import filters
# threshold = filters.threshold_otsu(image_array)
# binary = image_array > threshold
