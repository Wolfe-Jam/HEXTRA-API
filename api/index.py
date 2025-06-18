"""
HEXTRA API - The 38-Line Revolution
FastAPI application serving 38 years of computer vision expertise
"""

import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

from app.api.v1.endpoints import router as api_v1_router
from app.core.detection import get_algorithm_info
from app.models.schemas import HealthResponse

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="HEXTRA Detection API",
    description="""
    The 38-Line Revolution
    
    38 years of computer vision expertise (1986-2024) distilled into the most efficient 
    garment detection API on the planet. 
    
    What took Cloudinary's entire platform to fail at, we accomplish with 38 lines of code.
    Each line represents one year of hard-earned expertise.
    
    **Features:**
    - Sub-second processing on any image
    - Mathematical OTSU precision  
    - Industry-leading accuracy
    - Conquered Times Square Cutie (the impossible test case)
    
    **The Legend:** One line of code per year of experience.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "HEXTRA API Support",
        "url": "https://hextra.io",
    },
    license_info={
        "name": "Proprietary - The 38-Line Revolution",
        "url": "https://hextra.io/license",
    }
)

# Store startup time for uptime calculation
startup_time = time.time()

# CORS middleware - allow HEXTRA frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3005", 
        "http://localhost:3007",
        "http://localhost:3009",  # Added for current frontend port
        "https://catalog.hextra.io",
        "https://hextra-color-system-2.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_v1_router, prefix="/api/v1", tags=["Detection"])


@app.get("/", response_model=dict)
async def root():
    """
    Welcome to the HEXTRA API Empire
    """
    return {
        "message": "HEXTRA Detection API - The 38-Line Revolution",
        "version": "1.0.0",
        "tagline": "38 years. 38 lines. Infinite possibilities.",
        "documentation": "/docs",
        "legend": "Each line of code represents one year of expertise",
        "achievement": "Conquered what Cloudinary couldn't do",
        "status": "Revolutionary"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for monitoring
    """
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy",
        version="1.0.0", 
        uptime=round(uptime, 2),
        algorithm_info=get_algorithm_info()
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time to response headers
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for graceful error responses
    """
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "The 38-line algorithm encountered an unexpected situation",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.index:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )
