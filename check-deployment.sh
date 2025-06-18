#!/bin/bash
# Railway deployment diagnostic

echo "🔍 HEXTRA-API Deployment Check"
echo "=============================="

echo -e "\n📁 Project size (excluding venv):"
du -sh --exclude=venv --exclude=.git . 2>/dev/null || echo "Cannot calculate"

echo -e "\n📦 Python packages that will be installed:"
cat requirements.txt

echo -e "\n🔧 Key files for Railway:"
echo "- Procfile: $(cat Procfile)"
echo "- Runtime: $(cat runtime.txt)"
echo "- nixpacks.toml: $(if [ -f nixpacks.toml ]; then echo "Present"; else echo "Missing"; fi)"

echo -e "\n✅ Pre-deployment checklist:"
echo "- [ ] venv/ in .gitignore"
echo "- [ ] Procfile points to api.index:app"
echo "- [ ] requirements.txt under 250MB when installed"
echo "- [ ] Python 3.11.6 specified"

echo -e "\n💡 If OpenCV is the issue, consider:"
echo "1. Using requirements-minimal.txt (with scikit-image)"
echo "2. Multi-stage Docker build"
echo "3. Pre-built wheel for OpenCV"
