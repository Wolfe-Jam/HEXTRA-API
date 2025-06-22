from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import os

# Import your existing classes
from garment_masker import GarmentMasker
from focus_detector import FocusDetector

# Import new services for two-button approach
from services.quick_mask import QuickMaskProcessor
from services.sacred38_pro import Sacred38ProProcessor

app = FastAPI(title="HEXTRA Masking API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (including our frontend demo)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize your existing services
masker = GarmentMasker()
focus_detector = FocusDetector()

# Initialize new two-button processors
quick_mask_processor = QuickMaskProcessor(focus_detector)
sacred38_pro_processor = Sacred38ProProcessor(masker, focus_detector)

@app.get("/")
async def root():
    return {"message": "HEXTRA Masking API v2.0 - Two-Button System Ready"}

@app.get("/demo", response_class=HTMLResponse)
async def demo():
    """Serve the demo frontend"""
    with open("frontend_demo.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/sacred38-pro/")
async def sacred38_pro_analysis(
    file: UploadFile = File(...),
    return_debug: bool = False
):
    """
    Red-38 Button: Sacred-38 Pro B-Plus pipeline
    Professional multi-pass garment analysis with maximum accuracy
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process with Sacred-38 Pro
        result = sacred38_pro_processor.process(file_path, return_debug)
        
        # Clean up
        os.unlink(file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sacred-38 Pro processing error: {str(e)}")

@app.post("/quick-mask/")
async def quick_mask_analysis(
    file: UploadFile = File(...),
    return_debug: bool = False
):
    """
    Green-1 Button: Quick Mask pipeline
    Fast garment detection for simple images
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process with Quick Mask
        result = quick_mask_processor.process(file_path, return_debug)
        
        # Clean up
        os.unlink(file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick Mask processing error: {str(e)}")

# Keep your existing endpoint for backward compatibility
@app.post("/extract-garment/")
async def extract_garment(file: UploadFile = File(...)):
    """Legacy endpoint - preserved for backward compatibility"""
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
