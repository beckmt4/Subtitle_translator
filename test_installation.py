#!/usr/bin/env python3
"""
Test script to verify the installation instructions in README.md work correctly.
This simulates a fresh installation following the README steps.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description, check_output=False):
    """Run a command and report success/failure."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        if check_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ Success!")
                if result.stdout:
                    print(f"Output: {result.stdout[:200]}...")
                return True
            else:
                print(f"❌ Failed with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                return False
        else:
            result = subprocess.run(command, shell=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ Success!")
                return True
            else:
                print(f"❌ Failed with code {result.returncode}")
                return False
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ffmpeg_availability():
    """Test if FFmpeg is available and working."""
    return run_command("ffmpeg -version", "Testing FFmpeg availability", check_output=True)

def test_python_environment():
    """Test Python environment and package imports."""
    commands = [
        ("python --version", "Checking Python version"),
        ("python -c \"import faster_whisper; print('faster-whisper OK')\"", "Testing faster-whisper import"),
        ("python -c \"import ffmpeg; print('ffmpeg-python OK')\"", "Testing ffmpeg-python import"),
        ("python -c \"import rich; print('rich OK')\"", "Testing rich import"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description, check_output=True):
            success_count += 1
    
    return success_count == len(commands)

def test_validation_script():
    """Test the validation script."""
    return run_command("python validate_setup.py", "Running validation script", check_output=True)

def test_main_application():
    """Test the main application help."""
    return run_command("python whisper_mvp.py --help", "Testing main application", check_output=True)

def main():
    """Run installation verification tests."""
    print("Installation Instructions Verification")
    print("=" * 50)
    
    tests = [
        ("FFmpeg Installation", test_ffmpeg_availability),
        ("Python Environment", test_python_environment),
        ("Validation Script", test_validation_script),
        ("Main Application", test_main_application),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "="*50)
    print(f"Installation Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All installation instructions work correctly!")
        print("\nThe README.md installation steps are verified and functional.")
        return True
    else:
        print("⚠️  Some installation steps may need updating in README.md")
        print("\nPlease review failed tests and update documentation accordingly.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest error: {e}")
        sys.exit(1)