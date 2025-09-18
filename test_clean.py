#!/usr/bin/env python3
"""
Test script for the clean whisper implementation
Tests core functionality without PyAV dependency
"""

import os
import subprocess
import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import faster_whisper
        print(f"✅ faster-whisper {faster_whisper.__version__}")
    except ImportError as e:
        print(f"❌ faster-whisper import failed: {e}")
        return False
    
    try:
        import ctranslate2
        print(f"✅ ctranslate2 {ctranslate2.__version__}")
    except ImportError as e:
        print(f"❌ ctranslate2 import failed: {e}")
        return False
    
    try:
        import rich
        print("✅ rich (UI library)")
    except ImportError as e:
        print(f"❌ rich import failed: {e}")
        return False
    
    # Test that PyAV is NOT imported (clean implementation)
    try:
        import av
        print("⚠️  Warning: PyAV is installed but should not be used")
    except ImportError:
        print("✅ PyAV correctly not available (clean implementation)")
    
    return True

def test_ffmpeg():
    """Test FFmpeg availability"""
    print("\n🔍 Testing FFmpeg...")
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ {version_line}")
            return True
        else:
            print(f"❌ FFmpeg failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ FFmpeg not available: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\n🔍 Testing CUDA support...")
    
    try:
        import ctranslate2
        
        # Try to create a CUDA device
        try:
            device = ctranslate2.get_cuda_device_count()
            if device > 0:
                print(f"✅ CUDA devices reported by driver: {device}")
                # Additional DLL diagnostics (common missing runtime libs)
                missing = []
                dlls = [
                    'cudart64_12.dll',
                    'cublas64_12.dll',
                    'cublasLt64_12.dll',
                    'cudnn_ops64_9.dll'
                ]
                for name in dlls:
                    # Use where.exe to probe PATH (fast, Windows native)
                    try:
                        r = subprocess.run(['where', name], capture_output=True, text=True)
                        if r.returncode != 0:
                            missing.append(name)
                    except Exception:
                        # If where fails just skip
                        pass
                if missing:
                    print("⚠️  GPU detected but missing runtime DLLs (CUDA toolkit / cuDNN not fully installed):")
                    for m in missing:
                        print(f"   - {m}")
                    print("   → Transcription will automatically fall back to CPU unless you install CUDA Toolkit 12.x + cuDNN 9.")
                    return False  # Mark as not fully usable
                return True
            else:
                print("⚠️  No CUDA devices found - will use CPU")
                return False
        except Exception as e:
            print(f"⚠️  CUDA not available: {e}")
            return False
    except ImportError:
        print("❌ ctranslate2 not available for CUDA test")
        return False

def test_clean_script():
    """Test the clean whisper script"""
    print("\n🔍 Testing whisper_clean.py script...")
    
    script_path = Path("whisper_clean.py")
    if not script_path.exists():
        print("❌ whisper_clean.py not found")
        return False
    
    # Test help command
    try:
        result = subprocess.run([sys.executable, str(script_path), '--help'],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and 'usage:' in result.stdout:
            print("✅ Script help system working")
            return True
        else:
            print(f"❌ Script help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Script test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Clean Whisper Implementation")
    print("=" * 50)
    
    tests = [
        ("Package imports", test_imports),
        ("FFmpeg binary", test_ffmpeg),
        ("CUDA support", test_cuda),
        ("Clean script", test_clean_script),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print("\n📊 Test Results:")
    print("=" * 30)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
    
    all_critical_passed = results["Package imports"] and results["FFmpeg binary"]
    
    if all_critical_passed:
        print(f"\n🎉 Ready to use! Critical tests passed.")
        if not results["CUDA support"]:
            print("💡 Tip: For best performance, ensure CUDA is available")
    else:
        print(f"\n❌ Setup incomplete. Please fix failing tests.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())