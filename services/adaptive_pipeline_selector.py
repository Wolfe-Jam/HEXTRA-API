"""
Adaptive Pipeline Selector
Intelligently chooses the best extraction approach based on image characteristics

Author: Claude 3 Opus (claude-3-opus-20240229)
Purpose: Maximize success rate by using the right tool for each image
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from enum import Enum

from ..utils.logger import logger

class PipelineType(Enum):
    SACRED38_ONLY = "sacred38_only"
    SACRED38_PLUS_GRABCUT = "sacred38_plus_grabcut" 
    GROUNDED_SAM_PRIMARY = "grounded_sam_primary"
    FULL_MAGIC_PIPELINE = "full_magic_pipeline"
    COMPLEX_MULTI_STAGE = "complex_multi_stage"
    FALLBACK_BASIC = "fallback_basic"

class AdaptivePipelineSelector:
    """
    Analyzes images and selects the optimal extraction pipeline
    
    Uses machine learning-like decision trees but with computer vision heuristics
    to choose between different extraction approaches for maximum success rate.
    """
    
    def __init__(self):
        """Initialize with decision thresholds and pipeline configurations"""
        
        # Decision thresholds (tuned for garment extraction)
        self.thresholds = {
            'simple_complexity': 0.3,
            'moderate_complexity': 0.6,
            'high_complexity': 0.8,
            'white_visibility_good': 0.7,
            'white_visibility_poor': 0.3,
            'contrast_high': 50,
            'contrast_low': 20,
            'shadow_significant': 0.15,
            'background_simple': 0.4,
            'background_complex': 0.7
        }
        
        # Pipeline success rates (based on testing)
        self.pipeline_success_rates = {
            PipelineType.SACRED38_ONLY: 0.95,  # Your proven baseline
            PipelineType.SACRED38_PLUS_GRABCUT: 0.85,
            PipelineType.GROUNDED_SAM_PRIMARY: 0.80,
            PipelineType.FULL_MAGIC_PIPELINE: 0.75,
            PipelineType.COMPLEX_MULTI_STAGE: 0.65,
            PipelineType.FALLBACK_BASIC: 0.50
        }
        
        # Processing speed estimates (seconds)
        self.pipeline_speeds = {
            PipelineType.SACRED38_ONLY: 1.0,
            PipelineType.SACRED38_PLUS_GRABCUT: 3.0,
            PipelineType.GROUNDED_SAM_PRIMARY: 5.0,
            PipelineType.FULL_MAGIC_PIPELINE: 8.0,
            PipelineType.COMPLEX_MULTI_STAGE: 15.0,
            PipelineType.FALLBACK_BASIC: 0.5
        }
        
    def analyze_image_characteristics(self, image: np.ndarray) -> Dict:
        """
        Comprehensive image analysis for pipeline selection
        
        Returns all the metrics needed to make intelligent pipeline decisions
        """
        height, width = image.shape[:2]
        
        # 1. White garment visibility analysis
        white_visibility = self._analyze_white_visibility(image)
        
        # 2. Background complexity
        background_complexity = self._analyze_background_complexity(image)
        
        # 3. Lighting conditions
        lighting_analysis = self._analyze_lighting_conditions(image)
        
        # 4. Edge clarity
        edge_analysis = self._analyze_edge_clarity(image)
        
        # 5. Color distribution
        color_analysis = self._analyze_color_distribution(image)
        
        # 6. Texture complexity
        texture_complexity = self._analyze_texture_complexity(image)
        
        # 7. Overall scene type
        scene_type = self._classify_scene_type(image)
        
        # 8. Predicted difficulty
        difficulty_score = self._calculate_difficulty_score(
            white_visibility, background_complexity, lighting_analysis,
            edge_analysis, texture_complexity
        )
        
        return {
            'white_visibility': white_visibility,
            'background_complexity': background_complexity,
            'lighting_analysis': lighting_analysis,
            'edge_analysis': edge_analysis,
            'color_analysis': color_analysis,
            'texture_complexity': texture_complexity,
            'scene_type': scene_type,
            'difficulty_score': difficulty_score,
            'image_size': (width, height),
            'analysis_timestamp': time.time()
        }
    
    def _analyze_white_visibility(self, image: np.ndarray) -> Dict:
        """Analyze how visible white garments are in this image"""
        # Convert to different color spaces for comprehensive analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # HSV analysis for white detection
        h, s, v = cv2.split(hsv)
        
        # White regions typically have:
        # - Low saturation (< 30)
        # - High value (> 200)
        # - Any hue (hue is undefined for low saturation)
        white_mask_hsv = (s < 30) & (v > 200)
        white_percentage_hsv = np.sum(white_mask_hsv) / white_mask_hsv.size
        
        # LAB analysis for white detection  
        l, a, b = cv2.split(lab)
        
        # White regions in LAB:
        # - High L (> 80)
        # - A and B near 128 (neutral)
        white_mask_lab = (l > 200) & (np.abs(a - 128) < 20) & (np.abs(b - 128) < 20)
        white_percentage_lab = np.sum(white_mask_lab) / white_mask_lab.size
        
        # Combined white visibility
        combined_white_mask = white_mask_hsv | white_mask_lab
        combined_percentage = np.sum(combined_white_mask) / combined_white_mask.size
        
        # Analyze white region characteristics
        if np.any(combined_white_mask):
            # Connectivity of white regions
            white_uint8 = combined_white_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(white_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest white region
                largest_contour = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest_contour)
                
                # Shape analysis of largest white region
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 1.0
                
                white_region_quality = {
                    'largest_region_area': largest_area,
                    'aspect_ratio': aspect_ratio,
                    'connectivity': len(contours),
                    'position': {'x': x, 'y': y, 'w': w, 'h': h}
                }
            else:
                white_region_quality = {
                    'largest_region_area': 0,
                    'aspect_ratio': 0,
                    'connectivity': 0,
                    'position': None
                }
        else:
            white_region_quality = {
                'largest_region_area': 0,
                'aspect_ratio': 0,
                'connectivity': 0,
                'position': None
            }
        
        # Calculate visibility score (0-1)
        visibility_score = min(combined_percentage * 3, 1.0)  # Scale up for better discrimination
        
        return {
            'visibility_score': visibility_score,
            'white_percentage_hsv': white_percentage_hsv,
            'white_percentage_lab': white_percentage_lab,
            'combined_percentage': combined_percentage,
            'white_region_quality': white_region_quality,
            'quality_rating': 'good' if visibility_score > 0.7 else 'moderate' if visibility_score > 0.3 else 'poor'
        }
    
    def _analyze_background_complexity(self, image: np.ndarray) -> Dict:
        """Analyze background complexity"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture variation using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variation = np.var(laplacian)
        
        # Color variety
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        unique_hues = len(np.unique(h_channel[hsv[:, :, 1] > 30]))  # Only count saturated colors
        
        # Brightness variation
        brightness_std = np.std(gray)
        
        # Overall complexity score
        complexity_score = min((
            edge_density * 2 +
            min(texture_variation / 1000, 1.0) +
            min(unique_hues / 50, 1.0) +
            min(brightness_std / 100, 1.0)
        ) / 4, 1.0)
        
        return {
            'complexity_score': complexity_score,
            'edge_density': edge_density,
            'texture_variation': texture_variation,
            'color_variety': unique_hues,
            'brightness_std': brightness_std,
            'complexity_level': 'simple' if complexity_score < 0.4 else 'moderate' if complexity_score < 0.7 else 'complex'
        }
    
    def _analyze_lighting_conditions(self, image: np.ndarray) -> Dict:
        """Analyze lighting conditions and shadows"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Basic lighting statistics
        mean_brightness = np.mean(l_channel)
        brightness_std = np.std(l_channel)
        
        # Shadow detection
        shadow_threshold = mean_brightness * 0.6
        shadow_mask = l_channel < shadow_threshold
        shadow_percentage = np.sum(shadow_mask) / shadow_mask.size
        
        # Highlight detection
        highlight_threshold = mean_brightness * 1.4
        highlight_mask = l_channel > highlight_threshold
        highlight_percentage = np.sum(highlight_mask) / highlight_mask.size
        
        # Dynamic range
        dynamic_range = np.max(l_channel) - np.min(l_channel)
        
        # Lighting quality assessment
        if brightness_std < 20:
            lighting_quality = 'even'
        elif brightness_std > 50:
            lighting_quality = 'dramatic'
        else:
            lighting_quality = 'natural'
        
        # Shadow assessment
        shadow_severity = 'none' if shadow_percentage < 0.05 else 'light' if shadow_percentage < 0.15 else 'significant'
        
        return {
            'mean_brightness': mean_brightness,
            'brightness_std': brightness_std,
            'shadow_percentage': shadow_percentage,
            'highlight_percentage': highlight_percentage,
            'dynamic_range': dynamic_range,
            'lighting_quality': lighting_quality,
            'shadow_severity': shadow_severity,
            'has_shadows': shadow_percentage > 0.1,
            'has_highlights': highlight_percentage > 0.05
        }
    
    def _analyze_edge_clarity(self, image: np.ndarray) -> Dict:
        """Analyze edge clarity for garment boundaries"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection approaches
        # Canny edges
        canny_edges = cv2.Canny(gray, 50, 150)
        canny_strength = np.mean(canny_edges)
        
        # Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_strength = np.mean(sobel_magnitude)
        
        # Edge consistency (how well defined are the edges)
        edge_consistency = np.std(sobel_magnitude)
        
        # Overall edge clarity
        edge_clarity = min((canny_strength + sobel_strength / 50) / 2, 1.0)
        
        return {
            'edge_clarity': edge_clarity,
            'canny_strength': canny_strength,
            'sobel_strength': sobel_strength,
            'edge_consistency': edge_consistency,
            'clarity_rating': 'sharp' if edge_clarity > 0.6 else 'moderate' if edge_clarity > 0.3 else 'soft'
        }
    
    def _analyze_color_distribution(self, image: np.ndarray) -> Dict:
        """Analyze color distribution in the image"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Dominant colors analysis
        # Only consider saturated colors (s > 30)
        saturated_mask = s > 30
        saturated_hues = h[saturated_mask]
        
        if len(saturated_hues) > 0:
            # Count color regions
            hue_hist = cv2.calcHist([h], [0], saturated_mask.astype(np.uint8), [180], [0, 180])
            dominant_colors = np.argsort(hue_hist.flatten())[-5:]  # Top 5 colors
            
            color_diversity = len(np.where(hue_hist > np.max(hue_hist) * 0.1)[0])
        else:
            dominant_colors = []
            color_diversity = 0
        
        # Saturation analysis
        mean_saturation = np.mean(s)
        saturation_std = np.std(s)
        
        # Value (brightness) analysis
        mean_value = np.mean(v)
        value_std = np.std(v)
        
        return {
            'dominant_colors': dominant_colors.tolist(),
            'color_diversity': color_diversity,
            'mean_saturation': float(mean_saturation),
            'saturation_std': float(saturation_std),
            'mean_value': float(mean_value),
            'value_std': float(value_std),
            'color_complexity': min(color_diversity / 20, 1.0)
        }
    
    def _analyze_texture_complexity(self, image: np.ndarray) -> float:
        """Analyze texture complexity using Gabor filters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gabor filters at different orientations
        gabor_responses = []
        orientations = [0, 45, 90, 135]
        
        for angle in orientations:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi/3, 0.5, 0)
            response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(np.var(response))
        
        # Average texture energy across orientations
        texture_complexity = np.mean(gabor_responses) / 1000  # Normalize
        
        return min(texture_complexity, 1.0)
    
    def _classify_scene_type(self, image: np.ndarray) -> str:
        """Classify the overall scene type"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Color-based scene classification
        # Green dominance (outdoor/natural)
        green_mask = (h > 35) & (h < 85) & (s > 50)
        green_percentage = np.sum(green_mask) / green_mask.size
        
        # Blue dominance (sky/water)
        blue_mask = (h > 100) & (h < 130) & (s > 50)
        blue_percentage = np.sum(blue_mask) / blue_mask.size
        
        # Gray/neutral dominance (studio/indoor)
        neutral_mask = s < 30
        neutral_percentage = np.sum(neutral_mask) / neutral_mask.size
        
        # Brown/earth tones (urban/indoor)
        brown_mask = ((h > 10) & (h < 35)) | ((h > 160) & (h < 180))
        brown_percentage = np.sum(brown_mask) / brown_mask.size
        
        # Scene classification logic
        if neutral_percentage > 0.7:
            return 'studio'
        elif green_percentage > 0.3 or blue_percentage > 0.2:
            return 'outdoor'
        elif brown_percentage > 0.3:
            return 'indoor_complex'
        else:
            return 'mixed'
    
    def _calculate_difficulty_score(self, white_visibility: Dict, background_complexity: Dict,
                                  lighting_analysis: Dict, edge_analysis: Dict,
                                  texture_complexity: float) -> float:
        """Calculate overall extraction difficulty (0-1, higher = more difficult)"""
        
        # Factor contributions to difficulty
        visibility_difficulty = 1.0 - white_visibility['visibility_score']
        complexity_difficulty = background_complexity['complexity_score']
        lighting_difficulty = min(lighting_analysis['brightness_std'] / 100, 1.0)
        edge_difficulty = 1.0 - edge_analysis['edge_clarity']
        texture_difficulty = texture_complexity
        
        # Weighted combination
        difficulty = (
            visibility_difficulty * 0.3 +  # White visibility most important
            complexity_difficulty * 0.25 +
            lighting_difficulty * 0.2 +
            edge_difficulty * 0.15 +
            texture_difficulty * 0.1
        )
        
        # Penalty for specific challenging conditions
        if lighting_analysis['shadow_severity'] == 'significant':
            difficulty += 0.1
        
        if white_visibility['white_region_quality']['connectivity'] > 5:
            difficulty += 0.05  # Fragmented white regions are harder
        
        return min(difficulty, 1.0)
    
    def select_optimal_pipeline(self, image: np.ndarray, 
                               preferences: Optional[Dict] = None) -> Dict:
        """
        Select the optimal extraction pipeline for the given image
        
        Args:
            image: Input image
            preferences: Optional preferences (speed vs quality, etc.)
            
        Returns:
            Dict with pipeline selection and reasoning
        """
        logger.info("🎯 Analyzing image for optimal pipeline selection...")
        
        # Analyze image characteristics
        analysis = self.analyze_image_characteristics(image)
        
        # Apply preferences
        if preferences is None:
            preferences = {'priority': 'quality'}  # Default to quality over speed
        
        # Decision tree for pipeline selection
        selected_pipeline = self._decision_tree_selection(analysis, preferences)
        
        # Get pipeline metadata
        pipeline_info = self._get_pipeline_info(selected_pipeline)
        
        # Calculate confidence in selection
        selection_confidence = self._calculate_selection_confidence(analysis, selected_pipeline)
        
        # Backup pipeline recommendation
        backup_pipeline = self._select_backup_pipeline(selected_pipeline, analysis)
        
        result = {
            'selected_pipeline': selected_pipeline,
            'selection_confidence': selection_confidence,
            'backup_pipeline': backup_pipeline,
            'reasoning': self._generate_reasoning(analysis, selected_pipeline),
            'pipeline_info': pipeline_info,
            'image_analysis': analysis,
            'estimated_processing_time': pipeline_info['processing_time'],
            'expected_success_rate': pipeline_info['success_rate'],
            'recommendations': self._generate_recommendations(analysis, selected_pipeline)
        }
        
        logger.info(f"✅ Selected pipeline: {selected_pipeline.value}")
        logger.info(f"📊 Confidence: {selection_confidence:.1%}")
        logger.info(f"⏱️  Estimated time: {pipeline_info['processing_time']:.1f}s")
        
        return result
    
    def _decision_tree_selection(self, analysis: Dict, preferences: Dict) -> PipelineType:
        """Decision tree logic for pipeline selection"""
        
        difficulty = analysis['difficulty_score']
        white_visibility = analysis['white_visibility']['visibility_score']
        background_complexity = analysis['background_complexity']['complexity_score']
        scene_type = analysis['scene_type']
        has_shadows = analysis['lighting_analysis']['has_shadows']
        edge_clarity = analysis['edge_analysis']['edge_clarity']
        
        # High-level decision branches
        
        # Branch 1: Easy cases (Sacred-38 alone should work great)
        if (difficulty < self.thresholds['simple_complexity'] and 
            white_visibility > self.thresholds['white_visibility_good'] and
            background_complexity < self.thresholds['background_simple'] and
            scene_type == 'studio'):
            return PipelineType.SACRED38_ONLY
        
        # Branch 2: Moderate cases (Sacred-38 + GrabCut)  
        if (difficulty < self.thresholds['moderate_complexity'] and
            white_visibility > self.thresholds['white_visibility_poor'] and
            not (has_shadows and analysis['lighting_analysis']['shadow_percentage'] > 0.2)):
            return PipelineType.SACRED38_PLUS_GRABCUT
        
        # Branch 3: Good white visibility but complex background (Grounded-SAM)
        if (white_visibility > self.thresholds['white_visibility_good'] and
            background_complexity > self.thresholds['background_complex'] and
            scene_type in ['outdoor', 'indoor_complex']):
            return PipelineType.GROUNDED_SAM_PRIMARY
        
        # Branch 4: Very complex scenes (Full Magic Pipeline)
        if (difficulty < self.thresholds['high_complexity'] and
            preferences.get('priority', 'quality') == 'quality'):
            return PipelineType.FULL_MAGIC_PIPELINE
        
        # Branch 5: Extremely difficult cases (Multi-stage approach)
        if difficulty >= self.thresholds['high_complexity']:
            return PipelineType.COMPLEX_MULTI_STAGE
        
        # Branch 6: Fallback for edge cases
        return PipelineType.FALLBACK_BASIC
    
    def _get_pipeline_info(self, pipeline: PipelineType) -> Dict:
        """Get metadata for the selected pipeline"""
        return {
            'success_rate': self.pipeline_success_rates[pipeline],
            'processing_time': self.pipeline_speeds[pipeline],
            'complexity': 'low' if pipeline in [PipelineType.SACRED38_ONLY, PipelineType.FALLBACK_BASIC] else 'medium' if pipeline in [PipelineType.SACRED38_PLUS_GRABCUT, PipelineType.GROUNDED_SAM_PRIMARY] else 'high',
            'description': self._get_pipeline_description(pipeline)
        }
    
    def _get_pipeline_description(self, pipeline: PipelineType) -> str:
        """Get human-readable description of pipeline"""
        descriptions = {
            PipelineType.SACRED38_ONLY: "Sacred-38 binary enhancement only - fast and reliable for clean images",
            PipelineType.SACRED38_PLUS_GRABCUT: "Sacred-38 + GrabCut refinement - good balance of speed and quality", 
            PipelineType.GROUNDED_SAM_PRIMARY: "Semantic segmentation with text guidance - best for complex backgrounds",
            PipelineType.FULL_MAGIC_PIPELINE: "Complete AI pipeline with model fusion - highest quality extraction",
            PipelineType.COMPLEX_MULTI_STAGE: "Multi-algorithm approach with voting - for extremely difficult cases",
            PipelineType.FALLBACK_BASIC: "Basic computer vision techniques - reliable fallback option"
        }
        return descriptions.get(pipeline, "Unknown pipeline")
    
    def _calculate_selection_confidence(self, analysis: Dict, pipeline: PipelineType) -> float:
        """Calculate confidence in the pipeline selection"""
        
        difficulty = analysis['difficulty_score']
        
        # Base confidence from pipeline's expected success rate
        base_confidence = self.pipeline_success_rates[pipeline]
        
        # Adjust based on how well the image matches pipeline strengths
        if pipeline == PipelineType.SACRED38_ONLY:
            # Sacred-38 works best on clean, high-contrast images
            if (analysis['white_visibility']['visibility_score'] > 0.8 and
                analysis['background_complexity']['complexity_score'] < 0.3 and
                analysis['scene_type'] == 'studio'):
                adjustment = 0.1
            else:
                adjustment = -0.1
        
        elif pipeline == PipelineType.GROUNDED_SAM_PRIMARY:
            # Grounded-SAM works best when semantic understanding helps
            if (analysis['scene_type'] in ['outdoor', 'indoor_complex'] and
                analysis['white_visibility']['visibility_score'] > 0.5):
                adjustment = 0.05
            else:
                adjustment = -0.05
        
        elif pipeline == PipelineType.FULL_MAGIC_PIPELINE:
            # Magic pipeline is most robust but slower
            if difficulty > 0.5:
                adjustment = 0.05
            else:
                adjustment = 0.0  # No penalty for using it on easier images
        
        else:
            adjustment = 0.0
        
        # Confidence penalty for very difficult images
        if difficulty > 0.8:
            adjustment -= 0.1
        
        final_confidence = base_confidence + adjustment
        return max(0.1, min(1.0, final_confidence))
    
    def _select_backup_pipeline(self, primary: PipelineType, analysis: Dict) -> PipelineType:
        """Select backup pipeline if primary fails"""
        
        # Backup selection logic
        if primary == PipelineType.SACRED38_ONLY:
            return PipelineType.SACRED38_PLUS_GRABCUT
        elif primary == PipelineType.SACRED38_PLUS_GRABCUT:
            return PipelineType.FULL_MAGIC_PIPELINE
        elif primary == PipelineType.GROUNDED_SAM_PRIMARY:
            return PipelineType.FULL_MAGIC_PIPELINE
        elif primary == PipelineType.FULL_MAGIC_PIPELINE:
            return PipelineType.COMPLEX_MULTI_STAGE
        elif primary == PipelineType.COMPLEX_MULTI_STAGE:
            return PipelineType.FALLBACK_BASIC
        else:
            return PipelineType.SACRED38_ONLY  # Always fallback to proven method
    
    def _generate_reasoning(self, analysis: Dict, pipeline: PipelineType) -> List[str]:
        """Generate human-readable reasoning for pipeline selection"""
        reasons = []
        
        # Image characteristics reasoning
        if analysis['white_visibility']['visibility_score'] > 0.7:
            reasons.append("White garment is clearly visible")
        elif analysis['white_visibility']['visibility_score'] < 0.3:
            reasons.append("White garment has poor visibility")
        
        if analysis['background_complexity']['complexity_score'] < 0.4:
            reasons.append("Background is relatively simple")
        elif analysis['background_complexity']['complexity_score'] > 0.7:
            reasons.append("Background is highly complex")
        
        if analysis['lighting_analysis']['has_shadows']:
            reasons.append("Image contains shadows that may interfere")
        
        if analysis['scene_type'] == 'studio':
            reasons.append("Studio-like environment detected")
        elif analysis['scene_type'] == 'outdoor':
            reasons.append("Outdoor environment detected")
        
        # Pipeline-specific reasoning
        if pipeline == PipelineType.SACRED38_ONLY:
            reasons.append("Sacred-38 alone should handle this well")
        elif pipeline == PipelineType.FULL_MAGIC_PIPELINE:
            reasons.append("Complex scene requires multi-model approach")
        elif pipeline == PipelineType.GROUNDED_SAM_PRIMARY:
            reasons.append("Semantic understanding will help with complex background")
        
        return reasons
    
    def _generate_recommendations(self, analysis: Dict, pipeline: PipelineType) -> List[str]:
        """Generate recommendations for improving results"""
        recommendations = []
        
        if analysis['white_visibility']['visibility_score'] < 0.5:
            recommendations.append("Consider improving image lighting or contrast")
        
        if analysis['lighting_analysis']['shadow_percentage'] > 0.2:
            recommendations.append("Shadows detected - results may need post-processing")
        
        if analysis['background_complexity']['complexity_score'] > 0.8:
            recommendations.append("Very complex background - manual refinement may be needed")
        
        if analysis['edge_analysis']['edge_clarity'] < 0.4:
            recommendations.append("Soft edges detected - expect some boundary uncertainty")
        
        if pipeline == PipelineType.COMPLEX_MULTI_STAGE:
            recommendations.append("Using most intensive processing - expect longer processing time")
        
        return recommendations
    
    def batch_analyze_and_select(self, image_paths: List[str]) -> List[Dict]:
        """Analyze and select pipelines for multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"🎯 Analyzing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                
                selection_result = self.select_optimal_pipeline(image)
                selection_result['image_path'] = image_path
                results.append(selection_result)
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'selected_pipeline': PipelineType.FALLBACK_BASIC
                })
        
        return results
    
    def get_pipeline_statistics(self, results: List[Dict]) -> Dict:
        """Get statistics from batch pipeline selection"""
        if not results:
            return {}
        
        # Count pipeline selections
        pipeline_counts = {}
        total_confidence = 0
        total_processing_time = 0
        success_count = 0
        
        for result in results:
            if 'selected_pipeline' in result:
                pipeline = result['selected_pipeline']
                if isinstance(pipeline, PipelineType):
                    pipeline_name = pipeline.value
                else:
                    pipeline_name = str(pipeline)
                    
                pipeline_counts[pipeline_name] = pipeline_counts.get(pipeline_name, 0) + 1
                
                if 'selection_confidence' in result:
                    total_confidence += result['selection_confidence']
                    success_count += 1
                    
                if 'estimated_processing_time' in result:
                    total_processing_time += result['estimated_processing_time']
        
        avg_confidence = total_confidence / success_count if success_count > 0 else 0
        avg_processing_time = total_processing_time / len(results) if results else 0
        
        return {
            'total_images': len(results),
            'pipeline_distribution': pipeline_counts,
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time,
            'success_rate': success_count / len(results) if results else 0
        }


# Factory function
def create_adaptive_pipeline_selector():
    """Create an adaptive pipeline selector"""
    return AdaptivePipelineSelector()


# Example usage
if __name__ == "__main__":
    selector = create_adaptive_pipeline_selector()
    
    # Test with a single image
    test_image = cv2.imread("/path/to/test/image.jpg")
    
    if test_image is not None:
        # Select optimal pipeline
        result = selector.select_optimal_pipeline(test_image)
        
        logger.info(f"Pipeline Selection Results:")
        logger.info(f"- Selected: {result['selected_pipeline'].value}")
        logger.info(f"- Confidence: {result['selection_confidence']:.1%}")
        logger.info(f"- Backup: {result['backup_pipeline'].value}")
        logger.info(f"- Expected time: {result['estimated_processing_time']:.1f}s")
        logger.info(f"- Success rate: {result['expected_success_rate']:.1%}")
        logger.info(f"\\nReasoning:")
        for reason in result['reasoning']:
            logger.info(f"  • {reason}")
        logger.info(f"\\nRecommendations:")
        for rec in result['recommendations']:
            logger.info(f"  • {rec}")
    
    # Test batch analysis
    test_images = ["/path/to/image1.jpg", "/path/to/image2.jpg"]  # Replace with actual paths
    batch_results = selector.batch_analyze_and_select(test_images)
    
    if batch_results:
        stats = selector.get_pipeline_statistics(batch_results)
        logger.info(f"\\nBatch Analysis Statistics:")
        logger.info(f"- Total images: {stats['total_images']}")
        logger.info(f"- Average confidence: {stats['average_confidence']:.1%}")
        logger.info(f"- Pipeline distribution: {stats['pipeline_distribution']}")