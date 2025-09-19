#!/usr/bin/env python
"""
Test script to validate the subtitle-translator package installation.
"""
import sys
import importlib.util
from pathlib import Path

# Define required modules
required_modules = [
    "subtitle_translator",
    "subtitle_translator.asr",
    "subtitle_translator.mt",
    "subtitle_translator.quality",
    "subtitle_translator.media",
    "subtitle_translator.remux",
    "subtitle_translator.srt_io",
    "subtitle_translator.profiles",
    "subtitle_translator.cli",
]

# Define required classes
required_classes = [
    ("subtitle_translator.asr", "WhisperASR"),
    ("subtitle_translator.asr", "Segment"),
    ("subtitle_translator.mt", "TranslationEngine"),
    ("subtitle_translator.quality", "SubtitleQualityShaper"),
]

# Define required functions
required_functions = [
    ("subtitle_translator.media", "extract_audio"),
    ("subtitle_translator.media", "check_ffmpeg_available"),
    ("subtitle_translator.remux", "remux_subtitles"),
    ("subtitle_translator.srt_io", "read_srt"),
    ("subtitle_translator.srt_io", "write_srt"),
    ("subtitle_translator.profiles", "get_profile"),
    ("subtitle_translator.profiles", "list_profiles"),
]

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        return False, str(e)

def check_attribute(module_name, attr_name):
    """Check if a module has a specific attribute (class or function)."""
    try:
        module = importlib.import_module(module_name)
        return hasattr(module, attr_name)
    except ImportError:
        return False

def check_ffmpeg():
    """Check if FFmpeg is available."""
    try:
        from subtitle_translator.media import check_ffmpeg_available
        return check_ffmpeg_available()
    except ImportError:
        import subprocess
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

def check_faster_whisper():
    """Check if faster-whisper is available."""
    try:
        import faster_whisper
        return True
    except ImportError:
        return False

def check_transformers():
    """Check if transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False

def check_torch_cuda():
    """Check if PyTorch with CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def main():
    """Run all validation checks."""
    print("Subtitle Translator - Package Validation")
    print("=======================================")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python Version: {python_version}")
    
    # Check package modules
    print("\nChecking Package Modules:")
    all_modules_ok = True
    for module in required_modules:
        result = check_module(module)
        if result is True:
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}: {result[1]}")
            all_modules_ok = False
    
    if not all_modules_ok:
        print("\n⚠️  Some package modules are missing. Please reinstall the package.")
        return False
    
    # Check required classes
    print("\nChecking Required Classes:")
    all_classes_ok = True
    for module_name, class_name in required_classes:
        if check_attribute(module_name, class_name):
            print(f"  ✓ {module_name}.{class_name}")
        else:
            print(f"  ✗ {module_name}.{class_name}")
            all_classes_ok = False
    
    # Check required functions
    print("\nChecking Required Functions:")
    all_functions_ok = True
    for module_name, func_name in required_functions:
        if check_attribute(module_name, func_name):
            print(f"  ✓ {module_name}.{func_name}")
        else:
            print(f"  ✗ {module_name}.{func_name}")
            all_functions_ok = False
    
    # Check dependencies
    print("\nChecking Dependencies:")
    
    # FFmpeg
    if check_ffmpeg():
        print("  ✓ FFmpeg")
    else:
        print("  ✗ FFmpeg - Not found. Please install FFmpeg.")
    
    # faster-whisper
    if check_faster_whisper():
        print("  ✓ faster-whisper")
    else:
        print("  ✗ faster-whisper - Not installed.")
    
    # transformers (optional)
    if check_transformers():
        print("  ✓ transformers")
    else:
        print("  ⚠️  transformers - Not installed (optional for MT).")
    
    # PyTorch CUDA
    if check_torch_cuda():
        print("  ✓ PyTorch with CUDA")
    else:
        print("  ⚠️  PyTorch with CUDA not available (CPU mode will be used).")
    
    # Overall result
    print("\nValidation Summary:")
    if all_modules_ok and all_classes_ok and all_functions_ok:
        print("✅ Package structure is valid.")
        return True
    else:
        print("❌ Package structure has issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)