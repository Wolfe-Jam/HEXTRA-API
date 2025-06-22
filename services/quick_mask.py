"""
Quick Mask Processor - Green "1" Button
Fast, simple garment detection for instant results
"""
import cv2
import numpy as np
from .focus_detector import FocusDetector

class QuickMaskProcessor:
    def __init__(self, focus_detector: FocusDetector):
        """Initialize quick mask processor"""
        self.focus_detector = focus_detector
        
    def extract_largest_blob_simple(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Simple largest blob extraction - core of quick masking
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            return binary_image
        
        # Find largest component (excluding background which is label 0)
        sizes = stats[1:, cv2.CC_STAT_AREA]
        if len(sizes) == 0:
            return binary_image
            
        largest_idx = np.argmax(sizes) + 1
        
        # Create mask for only the largest component
        result = np.zeros_like(binary_image)
        result[labels == largest_idx] = 255
        
        # Quick cleanup
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return result
    
    def calculate_image_complexity(self, image: np.ndarray) -> float:
        """
        Calculate image complexity score (0-1) to recommend processing method
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate color variance
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        color_variance = np.var(lab) / 10000  # Normalize
        
        # Calculate focus spread
        focus_result = self.focus_detector.detect_foreground_focus(image)
        focus_spread = np.std(focus_result["focus_mask"]) / 127.5
        
        # Combine metrics
        complexity = (edge_density * 0.4 + 
                     min(color_variance, 1.0) * 0.3 + 
                     focus_spread * 0.3)
        
        return min(complexity, 1.0)
    
    def process_focus_only(self, image: np.ndarray) -> dict:
        """
        Fastest method: Focus detection + largest blob
        """
        # Focus detection
        focus_result = self.focus_detector.detect_foreground_focus(image)
        
        # Create binary from focus mask
        focus_mask = focus_result["focus_mask"]
        
        # Find largest blob in focus mask
        garment_mask = self.extract_largest_blob_simple(focus_mask)
        
        # Quick edge smoothing
        garment_mask = cv2.GaussianBlur(garment_mask, (3, 3), 0)
        _, garment_mask = cv2.threshold(garment_mask, 127, 255, cv2.THRESH_BINARY)
        
        return {
            "garment_mask": garment_mask,
            "method": "focus_only",
            "processing_time": "~0.5s",
            "confidence": self.calculate_mask_confidence(garment_mask, image)
        }
    
    def process_sacred38_simple(self, image: np.ndarray) -> dict:
        """
        Medium method: Sacred-38 + largest blob (no preprocessing)
        """
        from .sacred38 import apply_sacred38
        
        # Apply Sacred-38 directly
        sacred38_result = apply_sacred38(image)
        
        # Convert to grayscale
        if len(sacred38_result.shape) == 3:
            gray = cv2.cvtColor(sacred38_result, cv2.COLOR_BGR2GRAY)
        else:
            gray = sacred38_result.copy()
        
        # Ensure binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Extract largest blob
        garment_mask = self.extract_largest_blob_simple(binary)
        
        return {
            "garment_mask": garment_mask,
            "method": "sacred38_simple",
            "processing_time": "~1s",
            "confidence": self.calculate_mask_confidence(garment_mask, image)
        }
    
    def process_hybrid_quick(self, image: np.ndarray) -> dict:
        """
        Best compromise: Light preprocessing + Sacred-38 + blob
        """
        from .sacred38 import apply_sacred38
        
        # Light background blur (not elimination)
        blurred_bg = self.light_background_blur(image)
        
        # Sacred-38 on lightly processed input
        sacred38_result = apply_sacred38(blurred_bg)
        
        # Convert to grayscale
        if len(sacred38_result.shape) == 3:
            gray = cv2.cvtColor(sacred38_result, cv2.COLOR_BGR2GRAY)
        else:
            gray = sacred38_result.copy()
        
        # Extract largest blob with basic cleanup
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        garment_mask = self.extract_largest_blob_simple(binary)
        
        # Additional cleanup for hybrid method
        kernel = np.ones((7, 7), np.uint8)
        garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return {
            "garment_mask": garment_mask,
            "method": "hybrid_quick",
            "processing_time": "~1.5s",
            "confidence": self.calculate_mask_confidence(garment_mask, image)
        }
    
    def light_background_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply light background processing without full elimination
        """
        # Get focus mask
        focus_result = self.focus_detector.detect_foreground_focus(image)
        focus_mask = focus_result["focus_mask"]
        
        # Blur the background areas slightly
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        
        # Blend: keep focused areas, slightly blur unfocused
        result = image.copy()
        background_mask = cv2.bitwise_not(focus_mask)
        
        # Apply 30% blur to background
        alpha = 0.3
        result[background_mask == 255] = (
            alpha * blurred[background_mask == 255] + 
            (1 - alpha) * image[background_mask == 255]
        ).astype(np.uint8)
        
        return result
    
    def calculate_mask_confidence(self, mask: np.ndarray, original_image: np.ndarray) -> int:
        """
        Calculate confidence score for the mask (0-100)
        """
        if mask is None or not np.any(mask):
            return 0
        
        height, width = original_image.shape[:2]
        image_area = height * width
        mask_area = np.sum(mask > 0)
        
        # Size confidence (reasonable garment size)
        size_ratio = mask_area / image_area
        if 0.1 <= size_ratio <= 0.6:  # Reasonable garment coverage
            size_conf = 100
        elif size_ratio < 0.1:
            size_conf = int(size_ratio * 1000)  # Too small
        else:
            size_conf = int(100 - (size_ratio - 0.6) * 200)  # Too large
        
        # Shape confidence (check for reasonable bounding box)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if 0.3 <= aspect_ratio <= 3.0:  # Reasonable aspect ratio
                shape_conf = 100
            else:
                shape_conf = max(0, 100 - abs(aspect_ratio - 1.0) * 50)
        else:
            shape_conf = 0
        
        # Position confidence (garment should be in body region)
        if contours:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_y = M["m01"] / M["m00"]
                vertical_pos = center_y / height
                
                if 0.3 <= vertical_pos <= 0.8:  # Body region
                    pos_conf = 100
                else:
                    pos_conf = max(0, 100 - abs(vertical_pos - 0.55) * 200)
            else:
                pos_conf = 50
        else:
            pos_conf = 0
        
        # Weighted average
        final_confidence = int(
            size_conf * 0.4 + 
            shape_conf * 0.3 + 
            pos_conf * 0.3
        )
        
        return max(0, min(100, final_confidence))
    
    def process_quick_mask(self, image: np.ndarray) -> dict:
        """
        Main quick mask processing - automatically selects best method
        """
        complexity = self.calculate_image_complexity(image)
        
        if complexity < 0.3:
            # Simple image - use focus only
            return self.process_focus_only(image)
        elif complexity < 0.7:
            # Medium complexity - use Sacred-38 simple
            return self.process_sacred38_simple(image)
        else:
            # Complex image - use hybrid quick
            return self.process_hybrid_quick(image)
