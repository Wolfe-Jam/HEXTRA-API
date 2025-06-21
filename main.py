"""
HEXTRA-API: Garment Masking Service
Combines sacred-38 processing with OpenCV garment isolation
"""
import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
import cv2
import numpy as np

# Import our custom services
from services.sacred38 import apply_sacred38
from services.garment_masker import GarmentMasker

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HEXTRA Garment Mask Generator",
    description="Sacred-38 processing + intelligent garment isolation for HEXTRA Color Catalogs",
    version="1.0.0",
)

# Configure CORS for HEXTRA-COLOR-CATALOGS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative port
        "https://hextra.io",      # Production
        "https://catalog.hextra.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize garment masker
masker = GarmentMasker()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple status page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HEXTRA-API - Garment Masking Service</title>
        <style>
            body { 
                font-family: 'League Spartan', Arial, sans-serif; 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 20px;
                background: #f5f5f5;
            }
            .status { 
                background: #2ecc71; 
                color: white; 
                padding: 10px 20px; 
                border-radius: 5px; 
                display: inline-block;
            }
            h1 { color: #333; }
            .endpoints { 
                background: white; 
                padding: 20px; 
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
            code { 
                background: #f0f0f0; 
                padding: 2px 6px; 
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <h1>🎯 HEXTRA-API</h1>
        <div class="status">✅ Service Running</div>
        
        <div class="endpoints">
            <h2>Available Endpoints:</h2>
            <ul>
                <li><code>POST /process-garment/</code> - Full pipeline (sacred-38 + masking)</li>
                <li><code>POST /sacred38/</code> - Sacred-38 processing only</li>
                <li><code>POST /mask-garment/</code> - Garment masking only</li>
                <li><code>GET /docs</code> - Interactive API documentation</li>
            </ul>
        </div>
        
        <div class="endpoints">
            <h2>Pipeline Process:</h2>
            <ol>
                <li>Original Image → Sacred-38 Processing</li>
                <li>Sacred-38 Result → Garment Detection</li>
                <li>Final Output → Clean White Garment Mask</li>
            </ol>
        </div>
    </body>
    </html>
    """

@app.post("/process-garment/")
async def process_garment_full_pipeline(
    file: UploadFile = File(...),
    return_intermediate: bool = False
):
    """
    Full pipeline: Original image → Sacred-38 → Garment mask
    
    Args:
        file: Image file to process
        return_intermediate: If True, also returns sacred-38 result
    
    Returns:
        JSON with base64 encoded images
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Step 1: Apply sacred-38 processing
        sacred38_result = apply_sacred38(original_img)
        
        # Step 2: Extract garment mask from sacred-38 result
        garment_mask = masker.extract_garment(sacred38_result)
        
        if garment_mask is None:
            raise HTTPException(
                status_code=500, 
                detail="Could not detect garment. Try adjusting the image."
            )
        
        # Encode results
        response = {}
        
        # Always include final mask
        _, mask_buffer = cv2.imencode(".png", garment_mask)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        response["garment_mask"] = f"data:image/png;base64,{mask_base64}"
        
        # Optionally include intermediate sacred-38 result
        if return_intermediate:
            _, sacred_buffer = cv2.imencode(".png", sacred38_result)
            sacred_base64 = base64.b64encode(sacred_buffer).decode('utf-8')
            response["sacred38_result"] = f"data:image/png;base64,{sacred_base64}"
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/sacred38/")
async def process_sacred38_only(file: UploadFile = File(...)):
    """Apply only sacred-38 processing to an image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Apply sacred-38
        result = apply_sacred38(original_img)
        
        # Encode result
        _, buffer = cv2.imencode(".png", result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "sacred38_result": f"data:image/png;base64,{result_base64}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/mask-garment/")
async def mask_garment_only(file: UploadFile = File(...)):
    """Extract garment mask from an already processed image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Extract garment mask
        garment_mask = masker.extract_garment(img)
        
        if garment_mask is None:
            raise HTTPException(
                status_code=500, 
                detail="Could not detect garment"
            )
        
        # Encode result
        _, buffer = cv2.imencode(".png", garment_mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "garment_mask": f"data:image/png;base64,{mask_base64}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
