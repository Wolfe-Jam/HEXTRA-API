"""
Garment Masking Module
Intelligently isolates garments from processed images
"""
import cv2
import numpy as np
import os
from .focus_detector import FocusDetector

from ..utils.logger import logger

class GarmentMasker:
    def __init__(self):
        """Initialize the garment masker with face detection capability"""
        # Load Haar cascade for face detection
        self.face_cascade = self._load_face_cascade()
        
    def _load_face_cascade(self):
        """Load the Haar cascade classifier for face detection"""
        # Try multiple possible locations
        cascade_paths = [
            'static/haarcascade_frontalface_default.xml',
            '/Users/wolfejam/HEXTRA-API/static/haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    logger.info(f"Loaded face cascade from: {path}")
                    return cascade
        
        # If not found, try to download
        logger.info("Haar cascade not found locally, downloading...")
        return self._download_haar_cascade()
    
    def _download_haar_cascade(self):
        """Download Haar cascade if not found locally"""
        import requests
        
        url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Ensure static directory exists
            os.makedirs('static', exist_ok=True)
            
            # Save the file
            cascade_path = 'static/haarcascade_frontalface_default.xml'
            with open(cascade_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded Haar cascade to: {cascade_path}")
            return cv2.CascadeClassifier(cascade_path)
            
        except Exception as e:
            logger.error(f"Failed to download Haar cascade: {e}")
            return None
    
    def extract_garment(self, processed_image: np.ndarray) -> np.ndarray:
        """
        Extract garment mask from a processed image.
        
        Args:
            processed_image: Image that has been through sacred-38 processing
            
        Returns:
            Binary mask with white garment on black background
        """
        # Get image dimensions
        height, width = processed_image.shape[:2]
        
        # Convert to grayscale if needed
        if len(processed_image.shape) == 3:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_image.copy()
        
        # Step 1: Create head exclusion mask
        head_mask = self._create_head_exclusion_mask(gray)
        
        # Step 2: Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Step 3: Remove head area from binary image
        garment_candidates = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(head_mask))
        
        # Step 4: Find largest contour (likely the garment)
        garment_mask = self._find_largest_garment(garment_candidates)
        
        # Step 5: Clean up the mask
        if garment_mask is not None:
            garment_mask = self._clean_mask(garment_mask)
        
        return garment_mask
    
    def _create_head_exclusion_mask(self, gray_image: np.ndarray) -> np.ndarray:
        """Create a mask to exclude head and hair area"""
        height, width = gray_image.shape
        exclusion_mask = np.zeros_like(gray_image)
        
        if self.face_cascade is None:
            logger.warning("Face cascade not loaded, skipping face detection")
            return exclusion_mask
        
        # Detect faces with more sensitive parameters
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.05,  # More sensitive (was 1.1)
            minNeighbors=3,    # Less strict (was 5)
            minSize=(20, 20)   # Smaller minimum (was 30, 30)
        )
        
        if len(faces) > 0:
            # Use the first (largest) face
            x, y, w, h = faces[0]
            
            # Extend the face region to cover hair and neck
            # Balanced coverage - not too aggressive
            extend_up = int(h * 0.8)     # Hair above (reduced from 1.2)
            extend_down = int(h * 0.5)   # Neck below (reduced from 1.0)
            extend_sides = int(w * 0.5)  # Hair on sides (reduced from 0.8)
            
            # Calculate extended region
            x1 = max(0, x - extend_sides)
            y1 = max(0, y - extend_up)
            x2 = min(width, x + w + extend_sides)
            y2 = min(height, y + h + extend_down)
            
            # Draw filled rectangle for exclusion
            cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
            
            # Apply morphological operations to smooth the exclusion area
            kernel = np.ones((15, 15), np.uint8)  # Reduced from 25x25
            exclusion_mask = cv2.morphologyEx(exclusion_mask, cv2.MORPH_DILATE, kernel, iterations=1)
            exclusion_mask = cv2.morphologyEx(exclusion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return exclusion_mask
    
    def _find_largest_garment(self, binary_image: np.ndarray) -> np.ndarray:
        """Find the largest contour, which is likely the garment"""
        height, width = binary_image.shape
        
        # Find all contours
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Find the largest contour by area
        largest_contour = None
        max_area = 0
        min_area_threshold = width * height * 0.02  # Reduced to 2% for better capture
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_threshold and area > max_area:
                max_area = area
                largest_contour = contour
        
        if largest_contour is None:
            return None
        
        # Create mask from largest contour
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
        
        return mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up the mask with morphological operations"""
        # Close gaps in the garment
        kernel_close = np.ones((15, 15), np.uint8)  # Reduced from 25x25
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Remove small noise
        kernel_open = np.ones((5, 5), np.uint8)  # Reduced from 10x10
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Final smoothing with smaller kernel
        kernel_smooth = np.ones((7, 7), np.uint8)  # Reduced from 15x15
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
        
        # Apply gentle Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)  # Reduced from (5,5)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def process_with_params(self, image: np.ndarray, params: dict) -> np.ndarray:
        """
        Process image with custom parameters.
        
        Args:
            image: Input image
            params: Dictionary of parameters like:
                - min_area_ratio: Minimum garment area as ratio of image
                - head_extend_ratio: How much to extend head exclusion
                - morph_kernel_size: Size of morphological operations
        
        Returns:
            Garment mask
        """
        # This method allows fine-tuning of the extraction process
        # Implementation would use the params to adjust the processing
        return self.extract_garment(image)
    
    def extract_garment_with_original(self, processed_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Extract garment mask using both processed and original images.
        
        Args:
            processed_image: Image that has been through sacred-38 processing (B/W)
            original_image: Original color image for face detection
            
        Returns:
            Binary mask with white garment on black background
        """
        # Get image dimensions
        height, width = processed_image.shape[:2]
        
        # Convert processed image to grayscale if needed
        if len(processed_image.shape) == 3:
            processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            processed_gray = processed_image.copy()
        
        # Step 1: Create head exclusion mask using ORIGINAL image
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        head_mask = self._create_head_exclusion_mask(original_gray)
        
        # Step 2: The sacred-38 result is already binary, just ensure it's clean
        _, binary = cv2.threshold(processed_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Step 3: Remove head area from sacred-38 result
        garment_candidates = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(head_mask))
        
        # Step 4: Find largest contour (the garment)
        garment_mask = self._find_largest_garment(garment_candidates)
        
        # Step 5: Clean up the mask
        if garment_mask is not None:
            garment_mask = self._clean_mask(garment_mask)
        
        return garment_mask
    
    def extract_garment_with_color_hints(self, sacred38_result: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Extract garment using color information from original image.
        
        Args:
            sacred38_result: Black and white mask from sacred-38
            original_image: Original color image
            
        Returns:
            Clean garment mask
        """
        height, width = original_image.shape[:2]
        
        # Convert images
        if len(sacred38_result.shape) == 3:
            sacred_gray = cv2.cvtColor(sacred38_result, cv2.COLOR_BGR2GRAY)
        else:
            sacred_gray = sacred38_result.copy()
        
        # Create skin color mask to better identify face/arms
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Also detect face using cascade
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_original, 1.05, 3, minSize=(20, 20))
        
        # Create exclusion mask
        exclusion_mask = skin_mask.copy()
        
        # Add face region to exclusion
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Generous coverage for face and hair
                extend_up = int(h * 1.0)
                extend_down = int(h * 0.3)
                extend_sides = int(w * 0.6)
                
                x1 = max(0, x - extend_sides)
                y1 = max(0, y - extend_up)
                x2 = min(width, x + w + extend_sides)
                y2 = min(height, y + h + extend_down)
                
                cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
        
        # Dilate exclusion mask
        kernel = np.ones((15, 15), np.uint8)
        exclusion_mask = cv2.dilate(exclusion_mask, kernel, iterations=2)
        
        # Apply exclusion to sacred-38 result
        _, sacred_binary = cv2.threshold(sacred_gray, 127, 255, cv2.THRESH_BINARY)
        garment_candidates = cv2.bitwise_and(sacred_binary, sacred_binary, mask=cv2.bitwise_not(exclusion_mask))
        
        # Find largest remaining blob (the garment)
        contours, _ = cv2.findContours(garment_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create final mask
        garment_mask = np.zeros_like(sacred_gray)
        cv2.drawContours(garment_mask, [largest_contour], -1, 255, cv2.FILLED)
        
        # Clean up
        garment_mask = self._clean_mask(garment_mask)
        
        return garment_mask
    
    def extract_garment_background_first(self, sacred38_result: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Extract garment by first ensuring background is removed, then excluding face.
        
        Args:
            sacred38_result: Result from sacred-38 (background should be black)
            original_image: Original color image for face detection
            
        Returns:
            Clean garment mask
        """
        height, width = original_image.shape[:2]
        
        # Convert sacred38 result to grayscale
        if len(sacred38_result.shape) == 3:
            sacred_gray = cv2.cvtColor(sacred38_result, cv2.COLOR_BGR2GRAY)
        else:
            sacred_gray = sacred38_result.copy()
        
        # Ensure we have a clean binary mask
        _, binary_mask = cv2.threshold(sacred_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Step 1: Detect face in original image
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_original, 
            scaleFactor=1.03,  # Very sensitive
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # Create face exclusion mask
        face_mask = np.zeros_like(binary_mask)
        
        if len(faces) > 0:
            # Take the most prominent face (usually the largest)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Conservative exclusion - just face and hair
            extend_up = int(h * 0.6)    # Hair
            extend_down = int(h * 0.2)  # Small neck area
            extend_sides = int(w * 0.3) # Sides of hair
            
            x1 = max(0, x - extend_sides)
            y1 = max(0, y - extend_up)
            x2 = min(width, x + w + extend_sides)
            y2 = min(height, y + h + extend_down)
            
            cv2.rectangle(face_mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
            
            # Smooth the face mask
            kernel = np.ones((11, 11), np.uint8)
            face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
        
        # Step 2: Remove face area from sacred38 result
        garment_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=cv2.bitwise_not(face_mask))
        
        # Step 3: Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(garment_mask, connectivity=8)
        
        # Step 4: Find the largest component (excluding background)
        if num_labels > 1:
            # Get component sizes (excluding background which is label 0)
            sizes = stats[1:, cv2.CC_STAT_AREA]
            
            if len(sizes) > 0:
                # Sort components by size in descending order
                sorted_indices = np.argsort(sizes)[::-1]
                
                # Try to find the garment (usually the second largest after head)
                for idx in sorted_indices:
                    component_idx = idx + 1  # +1 because we excluded background
                    
                    # Get the bounding box of this component
                    x, y, w, h = stats[component_idx, cv2.CC_STAT_LEFT:cv2.CC_STAT_TOP + 4]
                    
                    # Check if this component is likely a garment (lower in image, wider)
                    component_center_y = y + h // 2
                    
                    # If component is in lower 2/3 of image and reasonably wide, it's likely the garment
                    if component_center_y > height * 0.4 and w > width * 0.3:
                        # Create mask for this component
                        final_mask = np.zeros_like(garment_mask)
                        final_mask[labels == component_idx] = 255
                        
                        # Clean up the final mask
                        kernel = np.ones((7, 7), np.uint8)
                        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                        
                        # Smooth edges
                        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
                        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
                        
                        return final_mask
                
                # If no garment-like component found, return the largest non-head component
                if len(sizes) > 1:
                    second_largest_idx = sorted_indices[1] + 1
                    final_mask = np.zeros_like(garment_mask)
                    final_mask[labels == second_largest_idx] = 255
                    return self._clean_mask(final_mask)
        
        # If no good component found, return cleaned garment mask
        return garment_mask
    
    def exclude_face_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Create a copy of the image with face/hair area blacked out.
        
        Args:
            image: Original color image
            
        Returns:
            Image with face/hair area set to black
        """
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with enhanced sensitivity
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.03,   # More sensitive detection
            minNeighbors=3,     # Slightly more lenient
            minSize=(25, 25),   # Detect smaller faces
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extra generous coverage for hoodies (more hair/head area)
                extend_up = int(h * 1.5)     # Extra height up for hair/hood
                extend_down = int(h * 1.0)   # More neck coverage  
                extend_sides = int(w * 0.8)  # More hair on sides
                
                x1 = max(0, x - extend_sides)
                y1 = max(0, y - extend_up)
                x2 = min(image.shape[1], x + w + extend_sides)
                y2 = min(image.shape[0], y + h + extend_down)
                
                # Set this region to black
                result[y1:y2, x1:x2] = 0
        
        return result
    
    def extract_largest_white_blob(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Extract the largest white blob from a binary image.
        Simple and effective for post-sacred38 processing.
        
        Args:
            binary_image: Binary image (black and white)
            
        Returns:
            Mask containing only the largest white blob
        """
        # Convert to grayscale if needed
        if len(binary_image.shape) == 3:
            gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_image.copy()
        
        # Ensure binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:
            return binary  # No components found or only background
        
        # Find the largest component (excluding background which is label 0)
        sizes = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(sizes) + 1
        
        # Create mask for only the largest component
        result = np.zeros_like(binary)
        result[labels == largest_idx] = 255
        
        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Smooth edges
        result = cv2.GaussianBlur(result, (5, 5), 0)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
        return result
    
    def extract_garment_exclude_face_after(self, sacred38_result: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Extract garment by excluding face from sacred-38 result.
        
        Args:
            sacred38_result: B/W mask from sacred-38
            original_image: Original color image for face detection
            
        Returns:
            Clean garment mask
        """
        # Convert sacred38 result to grayscale
        if len(sacred38_result.shape) == 3:
            sacred_gray = cv2.cvtColor(sacred38_result, cv2.COLOR_BGR2GRAY)
        else:
            sacred_gray = sacred38_result.copy()
        
        # Ensure binary
        _, binary_mask = cv2.threshold(sacred_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Detect faces in ORIGINAL image
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_original,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=3,    # Less strict
            minSize=(30, 30)
        )
        
        # Create a copy to modify
        result_mask = binary_mask.copy()
        
        # Black out face areas in the sacred-38 result
        if len(faces) > 0:
            # Process the largest face
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # More aggressive coverage for complete face removal
                extend_up = int(h * 1.5)    # Lots of hair coverage
                extend_down = int(h * 0.8)  # Good neck coverage
                extend_sides = int(w * 0.8) # Wide coverage
                
                x1 = max(0, x - extend_sides)
                y1 = max(0, y - extend_up)
                x2 = min(result_mask.shape[1], x + w + extend_sides)
                y2 = min(result_mask.shape[0], y + h + extend_down)
                
                # Set face region to black in the mask
                result_mask[y1:y2, x1:x2] = 0
        
        # Now find the largest white blob (should be the garment)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result_mask, connectivity=8)
        
        if num_labels <= 1:
            return result_mask
        
        # Find largest component (excluding background which is 0)
        sizes = stats[1:, cv2.CC_STAT_AREA]
        if len(sizes) > 0:
            largest_idx = np.argmax(sizes) + 1
            
            # Create final mask with only largest component
            final_mask = np.zeros_like(result_mask)
            final_mask[labels == largest_idx] = 255
            
            # Clean up with more aggressive morphology
            kernel = np.ones((15, 15), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Fill any internal holes
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(final_mask, contours, -1, 255, cv2.FILLED)
            
            # Final smoothing
            final_mask = cv2.GaussianBlur(final_mask, (9, 9), 0)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
            
            return final_mask
        
        return result_mask

    def extract_garment_with_focus(self, sacred38_result: np.ndarray, original_image: np.ndarray, focus_detector: FocusDetector = None) -> dict:
        """
        Enhanced garment extraction using focus detection for foreground/background separation.
        
        This method combines:
        1. Focus detection to separate focused (garment) from blurry (background) regions
        2. Sacred-38 binary segmentation
        3. Face detection and exclusion
        4. Largest blob extraction and cleanup
        
        Args:
            sacred38_result: B/W mask from sacred-38 processing
            original_image: Original color image for focus detection and face detection
            focus_detector: Optional FocusDetector instance (creates new if None)
            
        Returns:
            Dictionary containing:
            - garment_mask: Final clean garment mask
            - focus_mask: Focus detection mask (for debugging)
            - intermediate_result: Sacred-38 + focus combined mask (for debugging)
        """
        # Initialize focus detector if not provided
        if focus_detector is None:
            focus_detector = FocusDetector()
        
        # Step 1: Generate focus mask from original image
        focus_result = focus_detector.detect_foreground_focus(original_image, return_intermediate=True)
        focus_mask = focus_result["focus_mask"]
        
        # Step 2: Convert sacred38 result to binary
        if len(sacred38_result.shape) == 3:
            sacred_gray = cv2.cvtColor(sacred38_result, cv2.COLOR_BGR2GRAY)
        else:
            sacred_gray = sacred38_result.copy()
        
        _, sacred_binary = cv2.threshold(sacred_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Step 3: Combine focus mask with Sacred-38 result
        # CHANGED: Use focus as enhancement rather than hard filter
        # If focus detection finds focused areas, enhance those in Sacred-38
        # But don't completely remove areas that focus detection missed
        enhanced_sacred = sacred_binary.copy()
        
        # Enhance focused areas in the Sacred-38 result
        focus_enhanced_areas = cv2.bitwise_and(sacred_binary, focus_mask)
        enhanced_sacred = cv2.bitwise_or(enhanced_sacred, focus_enhanced_areas)
        
        # Use enhanced result instead of purely restrictive AND
        combined_mask = enhanced_sacred
        
        # Step 4: Face detection and exclusion on original image
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray_original,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30)
            )
            
            # Remove face areas from combined mask
            if len(faces) > 0:
                # Process the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Extended coverage for complete face removal
                extend_up = int(h * 1.5)    # Hair coverage
                extend_down = int(h * 0.8)  # Neck coverage
                extend_sides = int(w * 0.8) # Side coverage
                
                x1 = max(0, x - extend_sides)
                y1 = max(0, y - extend_up)
                x2 = min(combined_mask.shape[1], x + w + extend_sides)
                y2 = min(combined_mask.shape[0], y + h + extend_down)
                
                # Set face region to black
                combined_mask[y1:y2, x1:x2] = 0
        
        # Step 5: Find largest white blob (garment)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        
        final_mask = np.zeros_like(combined_mask)
        
        if num_labels > 1:
            # Find largest component (excluding background which is 0)
            sizes = stats[1:, cv2.CC_STAT_AREA]
            if len(sizes) > 0:
                largest_idx = np.argmax(sizes) + 1
                final_mask[labels == largest_idx] = 255
        
        # Step 6: Clean up the final mask
        if np.any(final_mask):
            # Aggressive morphological cleanup
            kernel = np.ones((15, 15), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Fill holes
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(final_mask, contours, -1, 255, cv2.FILLED)
            
            # Final smoothing
            final_mask = cv2.GaussianBlur(final_mask, (9, 9), 0)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Return comprehensive result
        return {
            "garment_mask": final_mask,
            "focus_mask": focus_mask,
            "intermediate_result": combined_mask,
            "focus_debug": focus_result if "laplacian_map" in focus_result else None
        }
