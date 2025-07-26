"""
Real-World Background Handler
Specialized algorithms for handling complex backgrounds like parks, cities, etc.

Author: Claude 3 Opus (claude-3-opus-20240229) 
Purpose: Make HEXTRA work with ANY background - the true Holy Grail
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import logger

class RealWorldBackgroundHandler:
    """
    Handles complex real-world backgrounds that challenge traditional extraction:
    - Park benches and outdoor furniture
    - City buildings and urban environments  
    - Natural lighting with shadows
    - Multiple depth planes
    - Textured backgrounds
    """
    
    def __init__(self):
        """Initialize background analysis tools"""
        self.background_types = [
            'studio', 'outdoor_natural', 'urban', 'indoor_complex', 
            'mixed_lighting', 'high_contrast', 'low_contrast'
        ]
        
        # Matting models (would load actual models in production)
        self.matting_available = False
        self.depth_estimation_available = False
        
    def analyze_background_complexity(self, image: np.ndarray, 
                                    initial_mask: np.ndarray = None) -> Dict:
        """
        Comprehensive analysis of background complexity
        
        Returns detailed metrics about what makes this background challenging
        """
        height, width = image.shape[:2]
        
        # 1. Edge density analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # 2. Color complexity
        color_complexity = self._analyze_color_complexity(image)
        
        # 3. Texture analysis
        texture_complexity = self._analyze_texture_complexity(image)
        
        # 4. Lighting analysis
        lighting_info = self._analyze_lighting_conditions(image)
        
        # 5. Depth estimation
        depth_complexity = self._estimate_depth_complexity(image)
        
        # 6. Background type classification
        bg_type = self._classify_background_type(image)
        
        # 7. Calculate overall complexity score
        complexity_score = self._calculate_complexity_score(
            edge_density, color_complexity, texture_complexity, 
            lighting_info, depth_complexity
        )
        
        return {
            'overall_complexity': complexity_score,
            'background_type': bg_type,
            'edge_density': edge_density,
            'color_complexity': color_complexity,
            'texture_complexity': texture_complexity,
            'lighting_info': lighting_info,
            'depth_complexity': depth_complexity,
            'challenges': self._identify_challenges(complexity_score, bg_type),
            'recommended_approach': self._recommend_approach(complexity_score, bg_type)
        }
    
    def _analyze_color_complexity(self, image: np.ndarray) -> float:
        """Analyze color distribution complexity"""
        # Convert to LAB for perceptual color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Reshape for clustering
        pixels = lab.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        n_colors = min(8, len(pixels))
        if n_colors < 2:
            return 0.0
            
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Calculate color distribution entropy
        labels = kmeans.labels_
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        
        # Shannon entropy for color complexity
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize to 0-1 range
        max_entropy = np.log2(n_colors)
        complexity = entropy / max_entropy if max_entropy > 0 else 0
        
        return complexity
    
    def _analyze_texture_complexity(self, image: np.ndarray) -> float:
        """Analyze texture complexity using Gabor filters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple Gabor filters for different orientations
        gabor_responses = []
        orientations = [0, 45, 90, 135]
        
        for angle in orientations:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi/3, 0.5, 0)
            response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(response)
        
        # Calculate texture energy
        texture_energy = 0
        for response in gabor_responses:
            texture_energy += np.var(response)
        
        # Normalize
        texture_complexity = min(texture_energy / 10000, 1.0)
        
        return texture_complexity
    
    def _analyze_lighting_conditions(self, image: np.ndarray) -> Dict:
        """Analyze lighting conditions and shadows"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate lighting statistics
        mean_brightness = np.mean(l_channel)
        brightness_std = np.std(l_channel)
        
        # Shadow detection
        shadow_threshold = mean_brightness * 0.6
        shadow_mask = l_channel < shadow_threshold
        shadow_percentage = np.sum(shadow_mask) / l_channel.size
        
        # Highlight detection
        highlight_threshold = mean_brightness * 1.4
        highlight_mask = l_channel > highlight_threshold
        highlight_percentage = np.sum(highlight_mask) / l_channel.size
        
        # Dynamic range
        dynamic_range = np.max(l_channel) - np.min(l_channel)
        
        return {
            'mean_brightness': float(mean_brightness),
            'brightness_variation': float(brightness_std),
            'shadow_percentage': float(shadow_percentage),
            'highlight_percentage': float(highlight_percentage),
            'dynamic_range': float(dynamic_range),
            'has_shadows': shadow_percentage > 0.1,
            'has_highlights': highlight_percentage > 0.1,
            'lighting_type': self._classify_lighting(mean_brightness, brightness_std)
        }
    
    def _classify_lighting(self, brightness: float, variation: float) -> str:
        """Classify lighting conditions"""
        if variation < 20:
            return 'studio_even'
        elif variation > 50:
            if brightness > 150:
                return 'harsh_sunlight'
            else:
                return 'dramatic_shadows'
        elif brightness > 180:
            return 'bright_outdoor'
        elif brightness < 100:
            return 'dim_indoor'
        else:
            return 'natural_mixed'
    
    def _estimate_depth_complexity(self, image: np.ndarray) -> float:
        """Estimate depth complexity without depth camera"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use blur variation as depth proxy
        # Areas at different depths have different focus levels
        blur_map = cv2.Laplacian(gray, cv2.CV_64F)
        blur_variance = np.var(blur_map)
        
        # Edge gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_variation = np.std(gradient_magnitude)
        
        # Combine metrics for depth complexity
        depth_complexity = min((blur_variance + gradient_variation) / 10000, 1.0)
        
        return depth_complexity
    
    def _classify_background_type(self, image: np.ndarray) -> str:
        """Classify the type of background"""
        # Analyze color distribution
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Green dominance (nature/parks)
        green_mask = (h_channel > 35) & (h_channel < 85) & (s_channel > 50)
        green_percentage = np.sum(green_mask) / green_mask.size
        
        # Blue dominance (sky/water)
        blue_mask = (h_channel > 100) & (h_channel < 130) & (s_channel > 50)
        blue_percentage = np.sum(blue_mask) / blue_mask.size
        
        # Gray/brown dominance (urban)
        gray_mask = s_channel < 30
        gray_percentage = np.sum(gray_mask) / gray_mask.size
        
        # High saturation (artificial/indoor)
        high_sat_mask = s_channel > 100
        high_sat_percentage = np.sum(high_sat_mask) / high_sat_mask.size
        
        # Classification logic
        if green_percentage > 0.3:
            return 'outdoor_natural'
        elif blue_percentage > 0.2:
            return 'outdoor_natural'  # Sky visible
        elif gray_percentage > 0.6:
            return 'urban'
        elif high_sat_percentage > 0.4:
            return 'indoor_complex'
        elif np.std(v_channel) < 20:
            return 'studio'
        else:
            return 'mixed_lighting'
    
    def _calculate_complexity_score(self, edge_density: float, color_complexity: float,
                                  texture_complexity: float, lighting_info: Dict,
                                  depth_complexity: float) -> float:
        """Calculate overall complexity score (0-1)"""
        # Weighted combination of complexity factors
        score = (
            edge_density * 0.2 +
            color_complexity * 0.2 +
            texture_complexity * 0.2 +
            (lighting_info['brightness_variation'] / 100) * 0.2 +
            depth_complexity * 0.2
        )
        
        # Adjust for specific challenging conditions
        if lighting_info['has_shadows'] and lighting_info['shadow_percentage'] > 0.2:
            score += 0.1
        
        if lighting_info['has_highlights'] and lighting_info['highlight_percentage'] > 0.1:
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_challenges(self, complexity_score: float, bg_type: str) -> List[str]:
        """Identify specific challenges for this background"""
        challenges = []
        
        if complexity_score > 0.7:
            challenges.append('high_complexity')
        
        if bg_type == 'outdoor_natural':
            challenges.extend(['natural_textures', 'variable_lighting'])
        elif bg_type == 'urban':
            challenges.extend(['geometric_patterns', 'artificial_textures'])
        elif bg_type == 'mixed_lighting':
            challenges.extend(['shadow_handling', 'light_variation'])
        
        return challenges
    
    def _recommend_approach(self, complexity_score: float, bg_type: str) -> Dict:
        """Recommend best extraction approach for this background"""
        if complexity_score < 0.3:
            return {
                'primary_method': 'sacred38_only',
                'backup_methods': ['basic_thresholding'],
                'confidence': 'high'
            }
        elif complexity_score < 0.6:
            return {
                'primary_method': 'sacred38_plus_grabcut',
                'backup_methods': ['watershed', 'region_growing'],
                'confidence': 'medium'
            }
        else:
            return {
                'primary_method': 'full_magic_pipeline',
                'backup_methods': ['deep_matting', 'manual_refinement'],
                'confidence': 'challenging'
            }
    
    def extract_from_complex_background(self, image: np.ndarray, 
                                      initial_mask: np.ndarray) -> Dict:
        """
        Main extraction method for complex backgrounds
        """
        logger.info("🌍 Starting complex background extraction...")
        
        # Analyze the background
        analysis = self.analyze_background_complexity(image, initial_mask)
        logger.info(f"📊 Background type: {analysis['background_type']}")
        logger.info(f"📊 Complexity score: {analysis['overall_complexity']:.2f}")
        
        # Choose extraction method based on analysis
        if analysis['overall_complexity'] < 0.4:
            result = self._simple_extraction(image, initial_mask)
        elif analysis['overall_complexity'] < 0.7:
            result = self._moderate_extraction(image, initial_mask, analysis)
        else:
            result = self._complex_extraction(image, initial_mask, analysis)
        
        # Add analysis info to result
        result['background_analysis'] = analysis
        
        return result
    
    def _simple_extraction(self, image: np.ndarray, initial_mask: np.ndarray) -> Dict:
        """Handle simple backgrounds with basic refinement"""
        logger.info("🟢 Using simple extraction approach")
        
        # Basic morphological operations
        kernel = np.ones((5, 5), np.uint8)
        refined_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        
        # Edge smoothing
        blurred = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        confidence = 85  # High confidence for simple backgrounds
        
        return {
            'refined_mask': final_mask,
            'confidence': confidence,
            'method_used': 'simple_morphological',
            'processing_time': 'fast'
        }
    
    def _moderate_extraction(self, image: np.ndarray, initial_mask: np.ndarray,
                           analysis: Dict) -> Dict:
        """Handle moderately complex backgrounds"""
        logger.info("🟡 Using moderate extraction approach")
        
        # Start with GrabCut for better edge refinement
        grabcut_result = self._apply_grabcut(image, initial_mask)
        
        if grabcut_result['success']:
            refined_mask = grabcut_result['mask']
        else:
            refined_mask = initial_mask
        
        # Handle specific challenges
        if analysis['lighting_info']['has_shadows']:
            refined_mask = self._handle_shadows(image, refined_mask, analysis['lighting_info'])
        
        if analysis['background_type'] in ['outdoor_natural', 'urban']:
            refined_mask = self._handle_textured_background(image, refined_mask)
        
        # Final cleanup
        refined_mask = self._final_cleanup(refined_mask)
        
        confidence = 75  # Medium confidence
        
        return {
            'refined_mask': refined_mask,
            'confidence': confidence,
            'method_used': 'grabcut_plus_refinement',
            'processing_time': 'medium'
        }
    
    def _complex_extraction(self, image: np.ndarray, initial_mask: np.ndarray,
                          analysis: Dict) -> Dict:
        """Handle very complex backgrounds with full pipeline"""
        logger.info("🔴 Using complex extraction approach")
        
        # Multi-stage approach
        results = []
        
        # 1. Try GrabCut
        grabcut_result = self._apply_grabcut(image, initial_mask)
        if grabcut_result['success']:
            results.append(('grabcut', grabcut_result['mask']))
        
        # 2. Try watershed segmentation
        watershed_mask = self._apply_watershed(image, initial_mask)
        if watershed_mask is not None:
            results.append(('watershed', watershed_mask))
        
        # 3. Try alpha matting simulation
        if self.matting_available:
            matting_mask = self._apply_alpha_matting(image, initial_mask)
            if matting_mask is not None:
                results.append(('alpha_matting', matting_mask))
        else:
            matting_mask = self._simulate_alpha_matting(image, initial_mask)
            results.append(('simulated_matting', matting_mask))
        
        # 4. Combine results using voting
        if len(results) > 1:
            final_mask = self._combine_masks_voting([mask for _, mask in results])
        elif len(results) == 1:
            final_mask = results[0][1]
        else:
            final_mask = initial_mask
        
        # 5. Final complex refinement
        final_mask = self._complex_refinement(image, final_mask, analysis)
        
        confidence = 60  # Lower confidence for complex scenes
        methods_used = [method for method, _ in results]
        
        return {
            'refined_mask': final_mask,
            'confidence': confidence,
            'method_used': f"multi_stage_{'+'.join(methods_used)}",
            'processing_time': 'slow'
        }
    
    def _apply_grabcut(self, image: np.ndarray, initial_mask: np.ndarray) -> Dict:
        """Apply GrabCut algorithm"""
        try:
            rect = cv2.boundingRect(initial_mask)
            
            if rect[2] <= 0 or rect[3] <= 0:
                return {'success': False, 'mask': initial_mask}
            
            # Initialize models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Create GrabCut mask
            grabcut_mask = np.zeros(image.shape[:2], np.uint8)
            
            # Set probable foreground/background based on initial mask
            grabcut_mask[initial_mask == 255] = cv2.GC_PR_FGD
            grabcut_mask[initial_mask == 0] = cv2.GC_PR_BGD
            
            # Apply GrabCut
            cv2.grabCut(image, grabcut_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            
            # Create final mask
            final_mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 255).astype('uint8')
            
            return {'success': True, 'mask': final_mask}
            
        except Exception as e:
            logger.warning(f"⚠️  GrabCut failed: {e}")
            return {'success': False, 'mask': initial_mask}
    
    def _apply_watershed(self, image: np.ndarray, initial_mask: np.ndarray) -> Optional[np.ndarray]:
        """Apply watershed segmentation"""
        try:
            # Distance transform
            dist_transform = cv2.distanceTransform(initial_mask, cv2.DIST_L2, 5)
            
            # Find sure foreground
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            # Find sure background
            kernel = np.ones((3, 3), np.uint8)
            sure_bg = cv2.dilate(initial_mask, kernel, iterations=3)
            
            # Unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Create markers
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            markers = cv2.watershed(image, markers)
            
            # Create result mask
            result_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            result_mask[markers > 1] = 255
            
            return result_mask
            
        except Exception as e:
            logger.warning(f"⚠️  Watershed failed: {e}")
            return None
    
    def _simulate_alpha_matting(self, image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """Simulate alpha matting with advanced edge refinement"""
        # Create trimap
        trimap = self._generate_trimap(initial_mask)
        
        # Simulate alpha matting with color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Unknown regions get analyzed for similarity to foreground
        unknown_mask = trimap == 128
        if np.any(unknown_mask):
            # Analyze foreground colors
            fg_mask = trimap == 255
            if np.any(fg_mask):
                fg_mean = np.mean(l_channel[fg_mask])
                fg_std = np.std(l_channel[fg_mask])
                
                # Calculate similarity in unknown regions
                unknown_values = l_channel[unknown_mask]
                similarity = np.exp(-np.abs(unknown_values - fg_mean) / (fg_std + 1))
                
                # Create alpha values
                alpha = np.zeros_like(l_channel, dtype=float)
                alpha[fg_mask] = 1.0
                alpha[unknown_mask] = similarity
                
                # Convert to binary mask
                result_mask = (alpha > 0.5).astype(np.uint8) * 255
            else:
                result_mask = initial_mask
        else:
            result_mask = initial_mask
        
        return result_mask
    
    def _generate_trimap(self, binary_mask: np.ndarray) -> np.ndarray:
        """Generate trimap for matting"""
        # Erode for sure foreground
        kernel = np.ones((7, 7), np.uint8)
        sure_fg = cv2.erode(binary_mask, kernel, iterations=2)
        
        # Dilate for sure background boundary
        sure_bg_boundary = cv2.dilate(binary_mask, kernel, iterations=2)
        
        # Create trimap
        trimap = np.zeros_like(binary_mask)
        trimap[sure_fg == 255] = 255  # Sure foreground
        trimap[sure_bg_boundary == 0] = 0   # Sure background
        trimap[(sure_bg_boundary == 255) & (sure_fg == 0)] = 128  # Unknown
        
        return trimap
    
    def _handle_shadows(self, image: np.ndarray, mask: np.ndarray, 
                       lighting_info: Dict) -> np.ndarray:
        """Handle shadow artifacts in extraction"""
        if not lighting_info['has_shadows']:
            return mask
        
        # Detect shadow regions
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        shadow_threshold = lighting_info['mean_brightness'] * 0.7
        shadow_mask = l_channel < shadow_threshold
        
        # Areas that are both shadow and garment might need connection
        shadow_garment = cv2.bitwise_and(mask, shadow_mask.astype(np.uint8) * 255)
        
        if np.sum(shadow_garment) > 0:
            # Use morphological closing to connect shadow-broken parts
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            connected_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Blend masks in shadow regions
            result_mask = mask.copy()
            shadow_pixels = shadow_mask
            result_mask[shadow_pixels] = connected_mask[shadow_pixels]
            
            return result_mask
        
        return mask
    
    def _handle_textured_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Handle textured backgrounds like grass, buildings"""
        # Use edge-preserving filter
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Re-extract mask from filtered image
        gray_filtered = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding on filtered image
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Combine with original mask
        combined = cv2.bitwise_and(mask, adaptive_thresh)
        
        # Fill small holes
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return combined
    
    def _combine_masks_voting(self, masks: List[np.ndarray]) -> np.ndarray:
        """Combine multiple masks using voting"""
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # Normalize all masks to 0/1
        normalized_masks = []
        for mask in masks:
            if mask is not None:
                normalized = (mask > 127).astype(np.uint8)
                normalized_masks.append(normalized)
        
        if not normalized_masks:
            return masks[0]
        
        # Sum votes
        vote_sum = np.sum(normalized_masks, axis=0)
        
        # Majority vote
        threshold = len(normalized_masks) // 2
        final_mask = (vote_sum > threshold).astype(np.uint8) * 255
        
        return final_mask
    
    def _complex_refinement(self, image: np.ndarray, mask: np.ndarray, 
                          analysis: Dict) -> np.ndarray:
        """Final refinement for complex cases"""
        # Multiple refinement passes
        refined = mask.copy()
        
        # 1. Handle specific background type
        if analysis['background_type'] == 'outdoor_natural':
            refined = self._refine_natural_background(image, refined)
        elif analysis['background_type'] == 'urban':
            refined = self._refine_urban_background(image, refined)
        
        # 2. Edge enhancement
        refined = self._enhance_edges(image, refined)
        
        # 3. Final cleanup
        refined = self._final_cleanup(refined)
        
        return refined
    
    def _refine_natural_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Specific refinements for natural backgrounds"""
        # Use color space that separates natural colors better
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Natural backgrounds often have green/brown dominance
        # White garments should stand out in certain channels
        u_channel = yuv[:, :, 1]
        v_channel = yuv[:, :, 2]
        
        # White areas have specific characteristics in UV channels
        white_like = (np.abs(u_channel - 128) < 20) & (np.abs(v_channel - 128) < 20)
        
        # Combine with existing mask
        enhanced_mask = cv2.bitwise_and(mask, white_like.astype(np.uint8) * 255)
        
        # Fill gaps
        kernel = np.ones((9, 9), np.uint8)
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel)
        
        return enhanced_mask
    
    def _refine_urban_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Specific refinements for urban backgrounds"""
        # Urban backgrounds have geometric patterns
        # Use Hough transforms to detect lines that might interfere
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        # Create mask of geometric structures
        geometric_mask = np.zeros_like(mask)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(geometric_mask, (x1, y1), (x2, y2), 255, 5)
        
        # Remove geometric interference from garment mask
        cleaned_mask = cv2.bitwise_and(mask, cv2.bitwise_not(geometric_mask))
        
        # Fill any holes created
        kernel = np.ones((7, 7), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        return cleaned_mask
    
    def _enhance_edges(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enhance mask edges using image gradients"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find mask edges
        mask_edges = cv2.Canny(mask, 50, 150)
        
        # Align mask edges with image gradients
        # This helps snap mask boundaries to actual object edges
        
        # Dilate mask edges slightly
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(mask_edges, kernel)
        
        # Find strong gradients near mask edges
        gradient_norm = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(np.uint8)
        strong_gradients = gradient_norm > 50
        
        # Combine edge information
        aligned_edges = cv2.bitwise_and(
            dilated_edges, 
            strong_gradients.astype(np.uint8) * 255
        )
        
        # Update mask with better edges
        enhanced_mask = cv2.bitwise_or(mask, aligned_edges)
        
        # Clean up
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        return enhanced_mask
    
    def _final_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Final cleanup operations"""
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Keep only largest component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background)
            sizes = stats[1:, cv2.CC_STAT_AREA]
            if len(sizes) > 0:
                largest_idx = np.argmax(sizes) + 1
                final_mask = np.zeros_like(cleaned)
                final_mask[labels == largest_idx] = 255
                cleaned = final_mask
        
        # Final edge smoothing
        cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
        _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        
        return cleaned


# Factory function
def create_realworld_background_handler():
    """Create a real-world background handler"""
    return RealWorldBackgroundHandler()


# Example usage
if __name__ == "__main__":
    handler = create_realworld_background_handler()
    
    # Test with a complex background
    test_image = cv2.imread("/path/to/complex/background/image.jpg")
    test_mask = cv2.imread("/path/to/initial/mask.png", 0)
    
    if test_image is not None and test_mask is not None:
        # Analyze background
        analysis = handler.analyze_background_complexity(test_image, test_mask)
        logger.info(f"Background Analysis: {analysis}")
        
        # Extract with complex background handling
        result = handler.extract_from_complex_background(test_image, test_mask)
        
        logger.info(f"Extraction Result:")
        logger.info(f"- Method: {result['method_used']}")
        logger.info(f"- Confidence: {result['confidence']}%")
        logger.info(f"- Processing Time: {result['processing_time']}")
        
        # Save result
        cv2.imwrite('complex_extraction_result.png', result['refined_mask'])
        logger.info("✅ Result saved as 'complex_extraction_result.png'")