#!/bin/bash

# HEXTRA-API Startup Script

echo "🎯 Starting HEXTRA-API..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create static directory if it doesn't exist
mkdir -p static

# Start the API
echo "Starting FastAPI server on http://localhost:8000"
echo "API Documentation available at http://localhost:8000/docs"
echo "================================"

uvicorn main:app --reload --host 0.0.0.0 --port 8000
