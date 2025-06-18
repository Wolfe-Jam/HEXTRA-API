#!/usr/bin/env python3
"""
FastAPI Setup Verification Script
Tests that all components are properly configured
"""

import sys
import importlib

def check_import(module_name, component=None):
    """Check if a module/component can be imported"""
    try:
        module = importlib.import_module(module_name)
        if component:
            getattr(module, component)
        print(f"✅ {module_name}{f'.{component}' if component else ''}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}{f'.{component}' if component else ''}: {e}")
        return False
    except AttributeError as e:
        print(f"❌ {module_name}.{component}: Not found")
        return False

def main():
    print("🔍 HEXTRA FastAPI Setup Check")
    print("=" * 40)
    
    # Check main FastAPI app
    print("\n1. Main Application:")
    check_import("main", "app")
    
    # Check core dependencies
    print("\n2. Core Dependencies:")
    checks = [
        ("fastapi", None),
        ("uvicorn", None),
        ("cv2", None),  # OpenCV
        ("PIL", None),  # Pillow
        ("numpy", None),
        ("pydantic", None),
    ]
    
    for module, component in checks:
        check_import(module, component)
    
    # Check app structure
    print("\n3. App Structure:")
    app_checks = [
        ("app.api.v1.endpoints", "router"),
        ("app.core.detection", "apply_otsu_to_image"),
        ("app.core.detection", "get_algorithm_info"),
        ("app.models.schemas", "DetectionResponse"),
        ("app.models.schemas", "HealthResponse"),
    ]
    
    all_good = True
    for module, component in app_checks:
        if not check_import(module, component):
            all_good = False
    
    # Test FastAPI app configuration
    print("\n4. FastAPI Configuration:")
    try:
        from main import app
        print(f"✅ App title: {app.title}")
        print(f"✅ App version: {app.version}")
        print(f"✅ Docs URL: {app.docs_url}")
        print(f"✅ Number of routes: {len(app.routes)}")
    except Exception as e:
        print(f"❌ FastAPI configuration error: {e}")
        all_good = False
    
    print("\n" + "=" * 40)
    if all_good:
        print("✅ All checks passed! FastAPI is properly configured.")
        print("\n🚀 To run locally:")
        print("   python main.py")
        print("   OR")
        print("   uvicorn main:app --reload")
    else:
        print("❌ Some issues found. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
