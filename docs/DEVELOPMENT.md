# Development Guide - Subtitle Translator

This guide provides key information for developers working on the Subtitle Translator project. It covers development workflows, code structure, and best practices.

## Development Environment Setup

### Prerequisites
- Python 3.9+ (3.9 or 3.10 recommended for compatibility)
- Windows 10/11 (primary target platform)
- NVIDIA GPU with CUDA support (for acceleration)
- FFmpeg installed and available in PATH

### Initial Setup

1. **Clone the repository**

2. **Set up PowerShell execution policy**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Initialize the environment**
   ```powershell
   # Create and configure virtual environment with pinned versions
   pwsh -File .\setup.ps1 -Force -Model medium
   
   # If you need a specific Python version
   pwsh -File .\setup.ps1 -Python "C:\Python310\python.exe" -Force
   ```

4. **Verify the setup**
   ```powershell
   .\.venv\Scripts\python.exe test_clean.py
   ```

### GPU Acceleration Setup

For GPU acceleration, ensure you have:
1. NVIDIA GPU drivers installed
2. CUDA Toolkit 12.x
3. cuDNN 9.x for CUDA 12

Verification:
```powershell
# Check GPU detection
pwsh -File .\scripts\check_cuda.ps1

# Verify DLL availability
where cudnn_ops64_9.dll
where cublas64_12.dll
```

## Project Structure

### Core Scripts
- **`whisper_clean.py`**: Primary implementation (Windows-compatible)
- **`asr_translate_srt.py`**: Two-pass ASR+MT implementation
- **`whisper_mvp.py`**: Legacy implementation with PyAV dependency

### Support Scripts
- **`setup.ps1`**: Environment setup automation
- **`test_clean.py`**: Environment validation
- **`check_cuda.ps1`**: GPU/CUDA diagnostics

### Documentation
- **`README.md`**: User-facing documentation
- **`docs/TROUBLESHOOTING.md`**: Issue resolution guide
- **`docs/CODEBASE.md`**: Codebase documentation
- **`docs/ARCHITECTURE.md`**: Architecture diagrams

## Development Workflow

### Adding New Features

1. **Fork from whisper_clean.py**: Use as the base for new implementations
2. **Import pattern**: Follow the existing import structure with graceful degradation
   ```python
   try:
       from faster_whisper import WhisperModel
   except ImportError:
       console.print("[red]faster-whisper not installed.[/red]")
       sys.exit(1)
   ```

3. **Error handling pattern**: Implement GPU fallback for robustness
   ```python
   try:
       # GPU attempt
       model = WhisperModel(model_name, device="cuda", compute_type="float16")
   except Exception as e:
       if not no_fallback:
           console.print(f"[yellow]GPU failed, attempting CPU fallback: {e}[/yellow]")
           model = WhisperModel(model_name, device="cpu", compute_type="int8")
       else:
           raise
   ```

### Testing Changes

1. **Test both paths**: GPU and CPU execution
   ```powershell
   # Test GPU path (normal)
   .\.venv\Scripts\python.exe whisper_clean.py test_video.mp4
   
   # Test CPU fallback
   .\.venv\Scripts\python.exe whisper_clean.py test_video.mp4 --device cpu
   
   # Test no-fallback behavior
   .\.venv\Scripts\python.exe whisper_clean.py test_video.mp4 --no-fallback
   ```

2. **Test with different model sizes**
   ```powershell
   # Test with tiny model (fast)
   .\.venv\Scripts\python.exe whisper_clean.py test_video.mp4 --model tiny
   
   # Test with medium model (balanced)
   .\.venv\Scripts\python.exe whisper_clean.py test_video.mp4 --model medium
   ```

3. **Validate across media formats**
   ```powershell
   # Test different media formats
   .\.venv\Scripts\python.exe whisper_clean.py test_video.mp4
   .\.venv\Scripts\python.exe whisper_clean.py test_audio.mp3
   .\.venv\Scripts\python.exe whisper_clean.py test_video.mkv
   ```

## Code Style Guide

### General Patterns

1. **Use Rich for UI**: Consistent progress bars and feedback
   ```python
   from rich.console import Console
   from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

   console = Console()
   with Progress(...) as progress:
       task = progress.add_task("[cyan]Processing...", total=100)
       # Work here
   ```

2. **Type Hints**: Include for maintainability
   ```python
   from typing import List, Optional, Dict
   
   def process_file(file_path: str, language: Optional[str] = None) -> Dict:
       # Implementation
   ```

3. **Error handling with fallback**: Always consider degraded operation
   ```python
   try:
       # Optimal path
   except Exception as e:
       console.print(f"[yellow]Warning: {e}. Using fallback.[/yellow]")
       # Fallback implementation
   ```

4. **Use pathlib**: Modern path handling
   ```python
   from pathlib import Path
   
   input_path = Path(input_str)
   if input_path.is_dir():
       # Process directory
   ```

### Windows Compatibility

1. **Avoid PyAV**: Use FFmpeg directly via subprocess
   ```python
   subprocess.run(['ffmpeg', '-i', video_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-y', audio_output], 
                 capture_output=True, text=True)
   ```

2. **Path handling**: Account for Windows backslashes
   ```python
   # Correct:
   path_str = str(Path(input_path))
   
   # Avoid:
   path_str = input_path  # Might have mixed slashes on Windows
   ```

3. **Temporary files**: Use the tempfile module
   ```python
   with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
       temp_audio_path = temp_file.name
   
   try:
       # Use temp_audio_path
   finally:
       # Clean up
       os.unlink(temp_audio_path)
   ```

## Dependency Management

### Primary Dependencies

Core functionality:
- faster-whisper: Main Whisper implementation
- ctranslate2: Inference engine
- rich: Terminal UI

Extended functionality:
- transformers: External MT models
- sentencepiece: Tokenization for MT
- torch: GPU acceleration for transformers

### Installation Pattern

```powershell
# Clean install (no PyAV)
.\.venv\Scripts\pip.exe install faster-whisper --no-deps
.\.venv\Scripts\pip.exe install ctranslate2 tokenizers transformers rich onnxruntime
```

### Managing Optional Dependencies

For two-pass translation:
```powershell
# Add MT capabilities
.\.venv\Scripts\pip.exe install transformers sentencepiece torch
```

With GPU support for torch:
```powershell
# Install torch with CUDA
.\.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu121
```

## Common Development Tasks

### Adding a New Model Format
1. Add model format to supported formats
2. Implement loading logic
3. Test with fallback pattern

### Improving Translation Quality
1. Modify `asr_translate_srt.py` parameters
2. Tune quality constraints:
   - CPS (characters per second)
   - Line breaks and wrapping
   - Duration constraints

### Optimizing Performance
1. Adjust compute types (`float16`, `int8_float16`, `int8`)
2. Balance model size vs. accuracy
3. Consider beam size for search efficiency

## Troubleshooting Development Issues

### GPU Not Detected
1. Run diagnostic script: `pwsh -File .\scripts\check_cuda.ps1`
2. Verify CUDA Toolkit and cuDNN installation
3. Check DLL availability: `where cudnn_ops64_9.dll`

### Import Errors
1. For PyAV warnings: Ignore if using `whisper_clean.py` (intentional)
2. For other imports: Check pinned versions match
3. Use `--no-deps` install pattern for faster-whisper

### FFmpeg Issues
1. Verify FFmpeg in PATH: `where ffmpeg`
2. Check FFmpeg version: `ffmpeg -version`
3. Test direct FFmpeg command: 
   ```powershell
   ffmpeg -i test_video.mp4 -ar 16000 -ac 1 -c:a pcm_s16le -y test_audio.wav
   ```

## Release Process

1. **Testing**: Validate across multiple platforms and GPU configurations
2. **Documentation**: Update README.md with new features
3. **Version Bump**: Update version numbers in script headers
4. **Release Notes**: Document changes, fixes, and improvements
5. **Distribution**: Tag release in repository