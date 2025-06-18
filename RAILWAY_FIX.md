# Railway Build Troubleshooting Guide

## 🚨 Common Railway Build Issues & Solutions

### 1. Missing main.py (FIXED ✅)
**Problem**: `ModuleNotFoundError: No module named 'app.main'`
**Solution**: Created app/main.py with FastAPI application

### 2. OpenCV Dependencies
**Problem**: OpenCV requires system libraries not available in basic containers
**Solution**: 
- Added nixpacks.toml with required system packages
- Using opencv-python-headless (no GUI dependencies)

### 3. Python Version Mismatch
**Current**: Python 3.11.6 specified in runtime.txt
**Railway Default**: May use different version
**Solution**: nixpacks.toml explicitly requests python311

### 4. Port Configuration
**Railway Requirement**: Must bind to $PORT environment variable
**Solution**: Procfile uses `--port $PORT`

## 🔧 Quick Fixes to Try

1. **Commit and Push All Changes**:
```bash
git add .
git commit -m "Fix Railway deployment - add missing main.py"
git push
```

2. **Environment Variables** (if needed):
- Set in Railway dashboard
- No .env file should be committed

3. **Build Command Override** (if nixpacks fails):
In Railway settings, try:
```
pip install --upgrade pip && pip install -r requirements.txt
```

## 🎯 Verification Steps

1. **Local Test First**:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. **Check Imports**:
```python
python -c "import app.main; print('✅ Import successful')"
```

3. **Railway Logs**:
- Check build logs for specific error
- Look for "Build failed" messages
- Check runtime logs after deploy

## 📝 Railway-Specific Files Created:
- ✅ app/main.py (was missing!)
- ✅ nixpacks.toml (system dependencies)
- ✅ railway.json (deployment config)
- ✅ Updated requirements.txt (added numpy, pydantic)
- ✅ Fixed app/api/v1/__init__.py imports

The main issue was the missing main.py file that Procfile was trying to run!
