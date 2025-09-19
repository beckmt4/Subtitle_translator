#!/usr/bin/env python
"""
Build script to create a standalone executable for the subtitle translator.
"""
import os
import subprocess
import sys
from pathlib import Path

def build_binary():
    """Build a standalone executable for the subtitle translator."""
    print("Building subtitle-translator binary...")
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Determine the script to build
    script_path = Path(__file__).parent / "src" / "subtitle_translator" / "cli.py"
    
    # Build the executable
    cmd = [
        "pyinstaller",
        "--name=subtitle-translator",
        "--onefile",
        "--console",
        # Add icon if available
        # "--icon=icon.ico",
        # Add version info
        "--version-file=version.txt",
        # Add hidden imports
        "--hidden-import=faster_whisper",
        "--hidden-import=ctranslate2",
        "--hidden-import=transformers",
        "--hidden-import=rich",
        str(script_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("Build complete!")
    print(f"Executable created: {Path('dist') / 'subtitle-translator.exe'}")

if __name__ == "__main__":
    build_binary()