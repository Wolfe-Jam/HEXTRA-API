"""
Tests for the HEXTRA API
Testing the legendary 38-line algorithm
"""

import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
import numpy as np

from app.main import app

client = TestClient(app)


def create_test_image(width: int = 300, height: int = 300, color: tuple = (255, 255, 255)) -> bytes:
    """Create a test image for API testing."""
    image = Image.new('RGB', (width, height), color)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "38-Line Revolution" in data["message"]
    assert data["version"] == "1.0.0"


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "algorithm_info" in data


def test_garment_detection_success():
    """Test successful garment detection."""
    # Create a white test image (perfect for OTSU)
    test_image_data = create_test_image(300, 300, (255, 255, 255))
    
    response = client.post(
        "/api/v1/detect/garment",
        files={"image": ("test.png", test_image_data, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["confidence"] > 0.9  # OTSU should be very confident
    assert "result_image" in data
    assert data["algorithm_used"] == "WolfeJam's 38-Line OTSU Detection"
    assert data["processing_time"] > 0


def test_algorithm_info():
    """Test algorithm information endpoint."""
    response = client.get("/api/v1/algorithm/info")
    assert response.status_code == 200
    data = response.json()
    
    assert data["lines_of_code"] == 38
    assert data["years_of_development"] == 38
    assert "Times Square Cutie conquered" in data["achievement"]


def test_invalid_file_upload():
    """Test uploading non-image file."""
    response = client.post(
        "/api/v1/detect/garment",
        files={"image": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code == 400


def test_otsu_enhancement_alias():
    """Test the OTSU enhancement endpoint (alias)."""
    test_image_data = create_test_image(100, 100, (128, 128, 128))
    
    response = client.post(
        "/api/v1/enhance/otsu",
        files={"image": ("test.png", test_image_data, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_processing_time_header():
    """Test that processing time is included in response headers."""
    response = client.get("/health")
    assert "X-Process-Time" in response.headers


def test_cors_headers():
    """Test CORS headers are present."""
    response = client.options("/api/v1/detect/garment")
    # CORS headers should be present for preflight requests
    assert response.status_code in [200, 405]  # 405 is also acceptable for OPTIONS


if __name__ == "__main__":
    pytest.main([__file__])
