"""
API v1 Endpoints - The Sacred 38 Lines Exposed to the World
"""

import time
import base64
import io
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from PIL import Image
import cv2
import numpy as np

from app.core.detection import apply_otsu_to_image, get_algorithm_info
from app.models.schemas import DetectionResponse, ErrorResponse

router = APIRouter()


async def validate_image(image: UploadFile = File(...)) -> np.ndarray:
    """Validate and convert uploaded image to numpy array."""
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and convert image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {str(e)}")


@router.post("/detect/garment", response_model=DetectionResponse)
async def detect_garment(image: UploadFile = File(...)):
    """
    🎯 THE MAIN EVENT - Garment Detection using the Sacred 38 Lines
    
    Upload any image and watch 38 years of expertise work its magic.
    This is the endpoint that conquered Times Square Cutie.
    """
    start_time = time.time()
    
    try:
        # Validate and process image
        image_array = await validate_image(image)
        
        # THE SACRED CALL - 38 years of expertise in action
        otsu_result = apply_otsu_to_image(image_array)
        
        # Convert result back to base64 for transmission
        _, buffer = cv2.imencode('.png', otsu_result)
        result_b64 = base64.b64encode(buffer).decode()
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            success=True,
            confidence=0.98,  
            result_image=f'data:image/png;base64,{result_b64}',
            processing_time=round(processing_time, 3),
            algorithm_used="HEXTRA Philosophical Detection Engine",
            message="38 years of pondering - from Ying/Yang to chess boards to mathematical perfection",
            metadata={
                "original_size": image_array.shape,
                "processed_size": otsu_result.shape,
                "philosophical_foundation": "Eastern/Western thought synthesis",
                "research_areas": ["Ying/Yang duality", "Game theory", "Mathematical optimization"],
                "breakthrough": "The convergence of ancient wisdom and modern mathematics"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Detection failed: {str(e)}"
        )


@router.get("/algorithm/info")
async def get_algorithm_information():
    """
    📚 The philosophical journey behind the breakthrough.
    """
    return {
        "name": "HEXTRA Philosophical Detection Engine",
        "years_of_development": 38,
        "journey": "From Eastern philosophy to Western mathematics",
        "research_areas": [
            "Ying/Yang duality principles",
            "OTSU threshold theory", 
            "K-means clustering mathematics",
            "Minimum/maximum optimization",
            "Mean statistical theory",
            "Triangular method analysis",
            "Chess board pattern recognition",
            "Binary opposition synthesis"
        ],
        "breakthrough_moment": "The realization that ancient duality concepts align perfectly with modern threshold mathematics",
        "philosophy": "Balance between light and dark, foreground and background, signal and noise",
        "achievement": "Conquered what others couldn't by thinking beyond pure technology",
        "efficiency": "1 line per year of philosophical and mathematical contemplation",
        "status": "Where ancient wisdom meets cutting-edge computer vision"
    }


@router.post("/enhance/otsu")
async def otsu_enhancement(image: UploadFile = File(...)):
    """
    ⚡ Direct access to OTSU enhancement (alias for detect/garment).
    Same legendary algorithm, different endpoint name.
    """
    return await detect_garment(image)
