#!/bin/bash

# HEXTRA API - Working Setup Script
echo "🎯 HEXTRA API - The 38-Line Revolution Setup"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the hextra-api directory"
    exit 1
fi

echo "📦 Setting up Python virtual environment..."
python3 -m venv venv

echo "⚡ Activating virtual environment..."
source venv/bin/activate

echo "🔧 Upgrading pip..."
pip install --upgrade pip

echo "📚 Installing OpenCV (headless version for better compatibility)..."
pip install opencv-python-headless

echo "🚀 Installing FastAPI and dependencies..."
pip install fastapi uvicorn pillow python-multipart python-dotenv pytest

echo "✅ Testing installation..."
python -c "
try:
    import cv2, fastapi, uvicorn
    print('🎉 All packages installed successfully!')
    print('📚 OpenCV version:', cv2.__version__)
    print('⚡ FastAPI version:', fastapi.__version__)
except ImportError as e:
    print('❌ Import error:', e)
    exit(1)
"

echo ""
echo "🧪 Running API tests..."
python -m pytest tests/ -v

echo ""
echo "🚀 Starting development server..."
echo "✨ Visit http://localhost:8000/docs for interactive API documentation"
echo "🎯 The Sacred 38 Lines are now serving the world!"
echo "📡 API endpoint: http://localhost:8000/api/v1/detect/garment"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn api.index:app --reload --port 8000