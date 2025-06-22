#!/bin/bash

# 🎯 HEXTRA Admin Testing App Launcher
# Focus: Garment Mask Parameter Refinement

echo "🎯 HEXTRA Admin Testing Interface"
echo "=================================="
echo ""
echo "🎯 Purpose: Garment mask parameter refinement"
echo "🚫 NO Face Processing - Focus on garment quality only"
echo "✅ Clean pipeline testing and optimization"
echo ""

# Check if we're in the right directory
if [ ! -f "admin_testing_app.py" ]; then
    echo "❌ Error: admin_testing_app.py not found"
    echo "Please run this script from the HEXTRA-API directory"
    exit 1
fi

# Setup virtual environment if it doesn't exist
if [ ! -d "admin_venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv admin_venv
    source admin_venv/bin/activate
    pip install flask opencv-python numpy pillow
else
    echo "🔧 Activating virtual environment..."
    source admin_venv/bin/activate
fi

# Create required directories
mkdir -p admin_test_results
mkdir -p templates
mkdir -p static

echo "✅ Environment ready"
echo ""

# Launch the admin testing interface
echo "🚀 Starting Admin Testing Interface..."
echo "📍 URL: http://localhost:8016/"
echo ""
echo "🎯 TESTING FOCUS:"
echo "  • Sacred-38 intensity optimization"
echo "  • Mask smoothing for artifact reduction"  
echo "  • Background separation threshold tuning"
echo "  • White area preservation enhancement"
echo "  • Edge refinement testing"
echo ""
echo "🚫 EXCLUDED FROM TESTING:"
echo "  • Face processing (completely isolated)"
echo "  • Complex multi-step face detection"
echo "  • Aggressive rectangular face cutting"
echo ""
echo "✅ WHAT TO EXPECT:"
echo "  • Clean hoodie shape recognition"
echo "  • No black rectangular holes"
echo "  • Background properly removed (black)"
echo "  • White garment areas preserved"
echo "  • Minimal artifacts (small dots only)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

python3 admin_testing_app.py