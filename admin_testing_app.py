#!/usr/bin/env python3
"""
HEXTRA Admin Testing Interface
Focus: Garment Mask Parameter Refinement

This admin app provides systematic testing capabilities for refining
garment masking parameters without face processing interference.
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
import json

# Import our clean garment processing services
from services.garment_masker import GarmentMasker
from services.sacred38_pro import Sacred38ProProcessor  
from services.quick_mask import QuickMaskProcessor
from services.focus_detector import FocusDetector

app = Flask(__name__)

# Initialize processing components
focus_detector = FocusDetector()
garment_masker = GarmentMasker()
sacred38_pro = Sacred38ProProcessor(garment_masker, focus_detector)
quick_mask_processor = QuickMaskProcessor(focus_detector)

# Admin test results storage
ADMIN_RESULTS_DIR = "admin_test_results"
os.makedirs(ADMIN_RESULTS_DIR, exist_ok=True)

@app.route('/')
def admin_interface():
    """Main admin testing interface"""
    return render_template('admin_test.html')

@app.route('/test_parameters', methods=['POST'])
def test_parameters():
    """
    Test garment masking with different parameter combinations
    Focus: Mask quality, NOT face processing
    """
    try:
        # Get uploaded image
        image_file = request.files['image']
        if not image_file:
            return jsonify({"error": "No image uploaded"}), 400
        
        # Get test parameters from admin interface
        params = {
            'sacred38_intensity': int(request.form.get('sacred38_intensity', 38)),
            'mask_smoothing': float(request.form.get('mask_smoothing', 0.0)),
            'edge_refinement': request.form.get('edge_refinement', 'false').lower() == 'true',
            'background_threshold': float(request.form.get('background_threshold', 0.5)),
            'white_preservation': float(request.form.get('white_preservation', 0.9))
        }
        
        # Process image with admin parameters
        image = Image.open(image_file.stream)
        
        # Step 1: Clean background masking (no face interference)
        image_array = np.array(image)
        mask_result = quick_mask_processor.process(image_array)
        
        # Step 2: Sacred-38 enhancement with custom intensity  
        enhanced_result = sacred38_pro.process(mask_result)
        
        # Step 3: Apply additional refinements
        final_result = apply_admin_refinements(
            enhanced_result, 
            params
        )
        final_result_pil = array_to_pil(final_result)
        
        # Generate comparison images
        results = {
            'original': image_to_base64(image),
            'step1_mask': image_to_base64(array_to_pil(mask_result)),
            'step2_enhanced': image_to_base64(array_to_pil(enhanced_result)),
            'final_result': image_to_base64(final_result_pil),
            'parameters': params,
            'quality_score': calculate_quality_score(final_result_pil),
            'test_timestamp': datetime.now().isoformat()
        }
        
        # Save test results for analysis
        save_admin_test_result(results)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

def apply_admin_refinements(image_array, params):
    """
    Apply additional refinements based on admin parameters
    Focus: Garment quality improvements only
    Input: numpy array, Output: numpy array
    """
    result = image_array.copy()
    
    # Edge refinement for cleaner garment boundaries
    if params['edge_refinement']:
        result = refine_garment_edges(result)
    
    # Mask smoothing to reduce small artifacts
    if params['mask_smoothing'] > 0:
        result = smooth_garment_mask(result, params['mask_smoothing'])
    
    # White area preservation enhancement
    if params['white_preservation'] > 0:
        result = enhance_white_preservation(result, params['white_preservation'])
    
    return result

def refine_garment_edges(image_array):
    """Improve garment edge definition - numpy array input/output"""
    # Apply gentle edge smoothing
    kernel = np.ones((3,3), np.uint8)
    smoothed = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel)
    return smoothed

def smooth_garment_mask(image_array, smoothing_factor):
    """Reduce small artifacts in garment mask - numpy array input/output"""
    # Handle both grayscale and color images
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
    
    # Apply Gaussian blur to reduce noise
    blur_amount = int(smoothing_factor * 10)
    if blur_amount > 0:
        blurred = cv2.GaussianBlur(gray, (blur_amount*2+1, blur_amount*2+1), 0)
        
        # Threshold to maintain binary nature
        _, clean_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Convert back to original format
        if len(image_array.shape) == 3:
            result = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        else:
            result = clean_mask
        return result
    
    return image_array

def enhance_white_preservation(image_array, preservation_factor):
    """Enhance white garment area preservation - numpy array input/output"""
    # Identify white/light areas (garment regions)
    white_threshold = int(255 * (1 - preservation_factor))
    
    if len(image_array.shape) == 3:
        # Color image - check all channels
        white_mask = np.all(image_array > white_threshold, axis=2)
        enhanced = image_array.copy()
        enhanced[white_mask] = [255, 255, 255]  # Pure white for garment areas
    else:
        # Grayscale image
        white_mask = image_array > white_threshold
        enhanced = image_array.copy()
        enhanced[white_mask] = 255  # Pure white for garment areas
    
    return enhanced

def calculate_quality_score(image):
    """
    Calculate objective quality metrics for garment mask
    Focus: Mask clarity, artifact level, background separation
    """
    np_image = np.array(image.convert('L'))
    
    # Metrics for garment mask quality
    metrics = {}
    
    # 1. Background separation quality (should be mostly black/white)
    unique_values = len(np.unique(np_image))
    metrics['binary_clarity'] = max(0, 100 - unique_values)  # Fewer unique values = more binary
    
    # 2. Edge definition quality
    edges = cv2.Canny(np_image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    metrics['edge_definition'] = min(100, edge_density * 1000)  # More defined edges = better
    
    # 3. Artifact level (small isolated regions)
    contours, _ = cv2.findContours(np_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_artifacts = len([c for c in contours if cv2.contourArea(c) < 100])
    metrics['artifact_level'] = max(0, 100 - small_artifacts)  # Fewer artifacts = better
    
    # 4. White area preservation
    white_pixels = np.sum(np_image > 200)
    total_pixels = np_image.size
    white_ratio = white_pixels / total_pixels
    metrics['white_preservation'] = white_ratio * 100
    
    # Overall quality score (weighted average)
    overall_score = (
        metrics['binary_clarity'] * 0.3 +
        metrics['edge_definition'] * 0.2 +
        metrics['artifact_level'] * 0.3 +
        metrics['white_preservation'] * 0.2
    )
    
    return {
        'overall_score': round(overall_score, 2),
        'detailed_metrics': metrics
    }

def save_admin_test_result(results):
    """Save admin test results for analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ADMIN_RESULTS_DIR}/test_{timestamp}.json"
    
    # Save parameters and quality scores (not images)
    test_data = {
        'parameters': results['parameters'],
        'quality_score': results['quality_score'],
        'timestamp': results['test_timestamp']
    }
    
    with open(filename, 'w') as f:
        json.dump(test_data, f, indent=2)

@app.route('/test_history')
def test_history():
    """Get history of admin test results"""
    history = []
    
    for filename in sorted(os.listdir(ADMIN_RESULTS_DIR)):
        if filename.endswith('.json'):
            with open(os.path.join(ADMIN_RESULTS_DIR, filename), 'r') as f:
                try:
                    data = json.load(f)
                    history.append(data)
                except:
                    continue
    
    return jsonify(history[-20:])  # Last 20 tests

@app.route('/optimal_parameters')
def optimal_parameters():
    """Find optimal parameters from test history"""
    history = []
    
    for filename in os.listdir(ADMIN_RESULTS_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(ADMIN_RESULTS_DIR, filename), 'r') as f:
                try:
                    data = json.load(f)
                    history.append(data)
                except:
                    continue
    
    if not history:
        return jsonify({"message": "No test history available"})
    
    # Find best parameters by overall score
    best_test = max(history, key=lambda x: x['quality_score']['overall_score'])
    
    return jsonify({
        "optimal_parameters": best_test['parameters'],
        "best_score": best_test['quality_score']['overall_score'],
        "test_count": len(history)
    })

def array_to_pil(np_array):
    """Convert numpy array to PIL image"""
    if len(np_array.shape) == 3 and np_array.shape[2] == 3:
        # BGR to RGB conversion for color images
        return Image.fromarray(cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB))
    elif len(np_array.shape) == 2:
        # Grayscale image
        return Image.fromarray(np_array)
    else:
        # Already RGB format
        return Image.fromarray(np_array)

def image_to_base64(pil_image):
    """Convert PIL image to base64 string for web display"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

if __name__ == '__main__':
    print("🎯 HEXTRA Admin Testing Interface")
    print("Focus: Garment Mask Parameter Refinement")
    print("URL: http://localhost:8016/admin")
    print("Purpose: Test and optimize masking parameters")
    app.run(host='0.0.0.0', port=8016, debug=True)
