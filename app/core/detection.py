"""
THE SACRED 38 LINES - WolfeJam's Legendary Detection Algorithm
=====================================================

38 years of expertise (1986-2024) distilled into perfect code.
These 38 lines conquered what Cloudinary's entire platform couldn't do.

DO NOT MODIFY - HISTORICAL ARTIFACT
Each line represents one year of hard-earned expertise.

Original creation: 2025
Status: UNTOUCHABLE FOR 38 YEARS (until 2063)
"""

import cv2
import os
import numpy as np
from pathlib import Path


def apply_otsu_to_folder_smart(input_folder, output_folder):
    """
    THE ORIGINAL 38-LINE MASTERPIECE
    Processes only images that do not already have a result in the output folder.
    """
    print(f"Starting smart batch conversion for folder: {input_folder}")
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    new_images_processed = 0
    skipped_images = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, "result_" + filename)
            
            # --- THIS IS THE NEW, SMART CHECK ---
            if not os.path.exists(output_path):
                # If the result file does NOT exist, process the image
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    _, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cv2.imwrite(output_path, otsu_image)
                    print(f"  Processed {filename}")
                    new_images_processed += 1
                else:
                    print(f"  Could not read {filename}, skipping.")
            else:
                # If the result file ALREADY exists, skip it
                skipped_images += 1
                
    print("\nBatch conversion complete.")
    print(f"  - New images processed: {new_images_processed}")
    print(f"  - Images skipped (already done): {skipped_images}")
    print(f"Results are in your folder: {output_folder}")


def apply_otsu_to_image(image_array):
    """
    Single image wrapper for the sacred OTSU algorithm.
    
    Args:
        image_array: numpy array representing grayscale image
        
    Returns:
        numpy array: OTSU-processed binary image
    """
    if len(image_array.shape) == 3:
        # Convert to grayscale if needed
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # THE SACRED LINE - The one line that changes everything
    _, otsu_image = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return otsu_image


def get_algorithm_info():
    """
    Information about the philosophical journey that led to the breakthrough.
    """
    return {
        "name": "WolfeJam's Philosophical Detection Synthesis",
        "years_of_contemplation": 38,
        "philosophical_foundation": "Ying/Yang duality principles",
        "mathematical_exploration": [
            "OTSU threshold optimization",
            "K-means clustering theory", 
            "Minimum/maximum boundaries",
            "Mean statistical analysis",
            "Triangular method studies",
            "Chess board pattern recognition"
        ],
        "breakthrough": "Ancient Eastern wisdom + Modern Western mathematics",
        "status": "The synthesis of opposites",
        "achievement": "Where philosophy meets computer vision",
        "essence": "Finding balance in the eternal dance between light and dark"
    }
