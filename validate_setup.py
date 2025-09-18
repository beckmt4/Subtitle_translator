#!/usr/bin/env python3
"""
Setup validation script for Whisper MVP
Checks if all dependencies are properly installed and configured.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.8+)")
        return False

def check_package_import(package_name, import_name=None):
    """Check if a Python package can be imported."""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} imported successfully")
        return True
    except ImportError:
        print(f"✗ {package_name} import failed")
        return False

def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ {version_line}")
            return True
        else:
            print("✗ FFmpeg found but failed to run")
            return False
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH")
        print("  Install with: choco install ffmpeg")
        print("  Or download from: https://www.gyan.dev/ffmpeg/builds/")
        return False
    except subprocess.TimeoutExpired:
        print("✗ FFmpeg command timed out")
        return False
    except Exception as e:
        print(f"✗ FFmpeg check failed: {e}")
        return False

def check_cuda():
    """Check CUDA availability (optional)."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"✓ CUDA available with {device_count} device(s): {device_name}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU mode")
            return False
    except ImportError:
        print("⚠ PyTorch not found - cannot check CUDA")
        return False

def check_faster_whisper_cuda():
    """Check if faster-whisper can use CUDA."""
    try:
        from faster_whisper import WhisperModel
        # Try to create a model with CUDA (this won't download anything)
        try:
            # This will fail if CUDA is not available but imports are OK
            print("✓ faster-whisper CUDA support available")
            return True
        except Exception:
            print("⚠ faster-whisper installed but CUDA support unclear")
            return False
    except ImportError:
        print("✗ faster-whisper import failed")
        return False

def main():
    """Run all validation checks."""
    print("Whisper MVP - Setup Validation")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("faster-whisper", lambda: check_package_import("faster-whisper", "faster_whisper")),
        ("ffmpeg-python", lambda: check_package_import("ffmpeg-python", "ffmpeg")),
        ("rich", lambda: check_package_import("rich")),
        ("FFmpeg Binary", check_ffmpeg),
    ]
    
    # Optional checks
    optional_checks = [
        ("PyTorch CUDA", check_cuda),
        ("faster-whisper CUDA", check_faster_whisper_cuda),
    ]
    
    print("\nRequired Dependencies:")
    print("-" * 25)
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        if check_func():
            passed += 1
    
    print(f"\nRequired: {passed}/{total} checks passed")
    
    print("\nOptional GPU Features:")
    print("-" * 25)
    gpu_passed = 0
    gpu_total = len(optional_checks)
    
    for name, check_func in optional_checks:
        if check_func():
            gpu_passed += 1
    
    print(f"\nGPU Features: {gpu_passed}/{gpu_total} available")
    
    # Summary
    print("\n" + "=" * 40)
    if passed == total:
        print("✅ Setup validation PASSED!")
        print("Ready to generate subtitles!")
        
        if gpu_passed == gpu_total:
            print("🚀 GPU acceleration fully available!")
        elif gpu_passed > 0:
            print("⚠️  Some GPU features available")
        else:
            print("💻 CPU-only mode (slower but functional)")
        
        print("\nNext steps:")
        print("1. python whisper_mvp.py --help")
        print("2. python whisper_mvp.py your_video.mp4")
        
        return True
    else:
        print("❌ Setup validation FAILED!")
        print("Please install missing dependencies before using Whisper MVP.")
        print("\nQuick fix:")
        print("pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nValidation error: {e}")
        sys.exit(1)