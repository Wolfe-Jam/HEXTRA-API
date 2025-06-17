"""
Pydantic models for HEXTRA API
Like TypeScript interfaces but for Python
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DetectionResponse(BaseModel):
    """Response model for garment detection."""
    success: bool = Field(description="Whether detection was successful")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence (0-1)")
    result_image: str = Field(description="Base64 encoded processed image")
    processing_time: float = Field(description="Processing time in seconds")
    algorithm_used: str = Field(description="Detection algorithm used")
    message: str = Field(description="Human-readable result message")
    metadata: Optional[dict] = Field(None, description="Additional processing metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    uptime: float = Field(description="Uptime in seconds")
    algorithm_info: dict = Field(description="Information about the core algorithm")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code for debugging")
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchProcessingRequest(BaseModel):
    """Request for batch processing (future feature)."""
    images: List[str] = Field(description="List of image URLs or base64 data")
    options: Optional[dict] = Field(None, description="Processing options")


class APIKeyInfo(BaseModel):
    """API key information (future feature)."""
    key_type: str = Field(description="test or live")
    requests_remaining: Optional[int] = Field(None, description="Requests remaining this month")
    rate_limit: dict = Field(description="Rate limiting information")
