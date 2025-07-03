#!/usr/bin/env python3
"""
Test script to verify all dependencies are available.
Run this before deployment to ensure everything is properly installed.
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        if package_name:
            importlib.import_module(package_name)
        else:
            importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    """Test all required dependencies."""
    print("Testing dependencies for ReadWave Quant Trading System...")
    print("=" * 60)
    
    # Core dependencies
    core_deps = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("yaml", "yaml"),
        ("ruamel.yaml", "ruamel.yaml"),
        ("requests", "requests"),
    ]
    
    # Trading dependencies
    trading_deps = [
        ("qlib", "qlib"),
        ("alpaca-py", "alpaca"),
        ("yfinance", "yfinance"),
        ("aiohttp", "aiohttp"),
        ("pytz", "pytz"),
    ]
    
    # ML dependencies
    ml_deps = [
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
    ]
    
    # Real-time dependencies
    realtime_deps = [
        ("websockets", "websockets"),
        ("asyncio-mqtt", "asyncio_mqtt"),
    ]
    
    # Monitoring dependencies
    monitoring_deps = [
        ("psutil", "psutil"),
        ("schedule", "schedule"),
        ("sqlalchemy", "sqlalchemy"),
        ("python-dotenv", "dotenv"),
        ("structlog", "structlog"),
    ]
    
    all_deps = [
        ("Core Dependencies", core_deps),
        ("Trading Dependencies", trading_deps),
        ("ML Dependencies", ml_deps),
        ("Real-time Dependencies", realtime_deps),
        ("Monitoring Dependencies", monitoring_deps),
    ]
    
    failed_imports = []
    
    for category, deps in all_deps:
        print(f"\n{category}:")
        print("-" * 30)
        for dep_name, import_name in deps:
            if not test_import(dep_name, import_name):
                failed_imports.append(dep_name)
    
    print("\n" + "=" * 60)
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
        print("Please install missing dependencies:")
        print(f"pip install {' '.join(failed_imports)}")
        return False
    else:
        print("✅ All dependencies are available!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 