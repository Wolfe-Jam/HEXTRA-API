"""
Garment Masking Module
Intelligently isolates garments from processed images
"""
import cv2
import numpy as np
import os

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
                    print(f"Loaded face cascade from: {path}")
                    return cascade
        
        # If not found, try to download
        print("Haar cascade not found locally, downloading...")
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
            
            print(f"Downloaded Haar cascade to: {cascade_path}")
            return cv2.CascadeClassifier(cascade_path)
            
        except Exception as e:
            print(f"Failed to download Haar cascade: {e}")
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
            print("Warning: Face cascade not loaded, skipping face detection")
            return exclusion_mask
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Use the first (largest) face
            x, y, w, h = faces[0]
            
            # Extend the face region to cover hair and neck
            # Increased coverage for better exclusion
            extend_up = int(h * 1.2)     # More coverage for hair above
            extend_down = int(h * 1.0)   # More coverage for neck
            extend_sides = int(w * 0.8)  # More coverage for hair on sides
            
            # Calculate extended region
            x1 = max(0, x - extend_sides)
            y1 = max(0, y - extend_up)
            x2 = min(width, x + w + extend_sides)
            y2 = min(height, y + h + extend_down)
            
            # Draw filled rectangle for exclusion
            cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
            
            # Apply morphological operations to smooth the exclusion area
            kernel = np.ones((25, 25), np.uint8)  # Larger kernel for better coverage
            exclusion_mask = cv2.morphologyEx(exclusion_mask, cv2.MORPH_DILATE, kernel, iterations=2)
            exclusion_mask = cv2.morphologyEx(exclusion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
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
        min_area_threshold = width * height * 0.05  # At least 5% of image for garment
        
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
        kernel_close = np.ones((25, 25), np.uint8)  # Larger for better gap filling
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # Remove small noise
        kernel_open = np.ones((10, 10), np.uint8)  # Larger for better noise removal
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        # Final smoothing with larger kernel
        kernel_smooth = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=2)
        
        # Apply Gaussian blur for smoother edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
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
