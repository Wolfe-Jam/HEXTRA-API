#!/bin/bash

# HEXTRA Two-Button System Test Runner
echo "🎯 HEXTRA Two-Button System - Starting Test Environment"
echo "=================================================="

# Check if services exist
echo "📁 Checking service files..."

if [ ! -f "services/quick_mask.py" ]; then
    echo "❌ Missing services/quick_mask.py"
    exit 1
fi

if [ ! -f "services/sacred38_pro.py" ]; then
    echo "❌ Missing services/sacred38_pro.py"
    exit 1
fi

echo "✅ Service files found"

# Check if demo frontend exists
if [ ! -f "frontend_demo.html" ]; then
    echo "❌ Missing frontend_demo.html"
    exit 1
fi

echo "✅ Frontend demo found"

# Start the server
echo ""
echo "🚀 Starting HEXTRA API Server..."
echo "   Backend: http://localhost:8000"
echo "   Demo UI: http://localhost:8000/demo"
echo ""
echo "🔴 Red Button (38): Sacred-38 Pro - Maximum accuracy"
echo "🟢 Green Button (1): Quick Mask - Lightning fast"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================================="

# Run the server with the demo
python main_with_demo.py
