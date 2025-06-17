"""
Utility functions for the HEXTRA API
"""

import base64
import io
from PIL import Image
import numpy as np


def image_to_base64(image_array: np.ndarray, format: str = 'PNG') -> str:
    """
    Convert numpy image array to base64 string.
    
    Args:
        image_array: numpy array representing image
        format: output format (PNG, JPEG, etc.)
        
    Returns:
        str: base64 encoded image with data URL prefix
    """
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to numpy image array.
    
    Args:
        base64_string: base64 encoded image (with or without data URL prefix)
        
    Returns:
        np.ndarray: image as numpy array
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    # Decode and convert to image
    image_data = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(image_data))
    
    return np.array(pil_image)


def validate_image_size(image_array: np.ndarray, max_width: int = 4096, max_height: int = 4096) -> bool:
    """
    Validate image dimensions.
    
    Args:
        image_array: numpy array representing image
        max_width: maximum allowed width
        max_height: maximum allowed height
        
    Returns:
        bool: True if image size is valid
    """
    height, width = image_array.shape[:2]
    return width <= max_width and height <= max_height


def get_image_info(image_array: np.ndarray) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image_array: numpy array representing image
        
    Returns:
        dict: image information
    """
    shape = image_array.shape
    
    return {
        "width": shape[1] if len(shape) > 1 else 1,
        "height": shape[0],
        "channels": shape[2] if len(shape) > 2 else 1,
        "total_pixels": image_array.size,
        "dtype": str(image_array.dtype),
        "memory_usage_mb": round(image_array.nbytes / (1024 * 1024), 2)
    }
