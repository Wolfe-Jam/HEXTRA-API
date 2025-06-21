"""
Focus Detection Module for Foreground/Background Segmentation
Distinguishes sharp (foreground) from blurry (background) regions
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class FocusDetector:
    def __init__(self):
        """Initialize the focus detector with default parameters"""
        self.laplacian_threshold = 100.0  # Threshold for focus detection
        self.gradient_threshold = 50.0    # Threshold for edge sharpness
        self.kernel_size = 5              # Kernel size for operations
        
    def calculate_laplacian_variance(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate Laplacian variance to measure focus/blur.
        Higher variance = more in focus
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Focus map where higher values = more focused
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate local variance using sliding window
        # This gives us a focus map rather than single value
        kernel = np.ones((self.kernel_size, self.kernel_size), np.float32) / (self.kernel_size * self.kernel_size)
        
        # Local mean of Laplacian
        local_mean = cv2.filter2D(laplacian, -1, kernel)
        
        # Local variance calculation
        laplacian_squared = laplacian * laplacian
        local_mean_squared = cv2.filter2D(laplacian_squared, -1, kernel)
        
        # Variance = E[X²] - E[X]²
        variance_map = local_mean_squared - (local_mean * local_mean)
        
        return np.abs(variance_map)
    
    def calculate_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate gradient magnitude to detect sharp edges.
        Higher magnitude = sharper edges = more focused
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Gradient magnitude map
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calculate gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return gradient_magnitude
    
    def create_focus_mask(self, image: np.ndarray, method: str = "combined") -> np.ndarray:
        """
        Create binary focus mask separating foreground from background.
        
        Args:
            image: Input image
            method: "laplacian", "gradient", or "combined"
            
        Returns:
            Binary mask where 255 = foreground (focused), 0 = background (blurry)
        """
        if method == "laplacian":
            focus_map = self.calculate_laplacian_variance(image)
            threshold = self.laplacian_threshold
            
        elif method == "gradient":
            focus_map = self.calculate_gradient_magnitude(image)
            threshold = self.gradient_threshold
            
        elif method == "combined":
            # Combine both methods for better results
            laplacian_map = self.calculate_laplacian_variance(image)
            gradient_map = self.calculate_gradient_magnitude(image)
            
            # Normalize both maps to 0-255 range
            laplacian_norm = cv2.normalize(laplacian_map, None, 0, 255, cv2.NORM_MINMAX)
            gradient_norm = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX)
            
            # Weighted combination (favor gradient detection)
            focus_map = 0.4 * laplacian_norm + 0.6 * gradient_norm
            threshold = 127  # Middle threshold for combined normalized map
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create binary mask
        _, binary_mask = cv2.threshold(focus_map.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Remove small noise
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return cleaned
    
    def apply_focus_filter(self, image: np.ndarray, focus_mask: np.ndarray) -> np.ndarray:
        """
        Apply focus mask to image, keeping only focused regions.
        
        Args:
            image: Original image
            focus_mask: Binary focus mask
            
        Returns:
            Image with background (blurry) regions set to black
        """
        if len(image.shape) == 3:
            # For color images, apply mask to all channels
            result = image.copy()
            result[focus_mask == 0] = [0, 0, 0]  # Set background to black
        else:
            # For grayscale images
            result = image.copy()
            result[focus_mask == 0] = 0
            
        return result
    
    def detect_foreground_focus(self, image: np.ndarray, return_intermediate: bool = False) -> dict:
        """
        Main function to detect focused foreground regions.
        
        Args:
            image: Input image
            return_intermediate: If True, return intermediate results for debugging
            
        Returns:
            Dictionary containing focus mask and optionally intermediate results
        """
        # Create focus mask
        focus_mask = self.create_focus_mask(image, method="combined")
        
        # Apply focus filter
        filtered_image = self.apply_focus_filter(image, focus_mask)
        
        result = {
            "focus_mask": focus_mask,
            "filtered_image": filtered_image
        }
        
        if return_intermediate:
            # Add intermediate results for debugging
            laplacian_map = self.calculate_laplacian_variance(image)
            gradient_map = self.calculate_gradient_magnitude(image)
            
            result.update({
                "laplacian_map": cv2.normalize(laplacian_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                "gradient_map": cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            })
        
        return result
    
    def set_thresholds(self, laplacian_threshold: float = None, gradient_threshold: float = None):
        """
        Adjust detection thresholds for different image types.
        
        Args:
            laplacian_threshold: Threshold for Laplacian variance
            gradient_threshold: Threshold for gradient magnitude
        """
        if laplacian_threshold is not None:
            self.laplacian_threshold = laplacian_threshold
        if gradient_threshold is not None:
            self.gradient_threshold = gradient_threshold
