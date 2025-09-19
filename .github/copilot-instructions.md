# AI Agent Instructions for Subtitle Translator

## Project Overview
This is a **Windows-first** GPU-accelerated subtitle generation tool using OpenAI's Whisper via `faster-whisper`. The project solves **PyAV compatibility issues** on Windows with a clean FFmpeg-based approach and provides automatic CUDA→CPU fallback for robustness.

## Architecture & Core Components

### Three Main Scripts (Execution Priority)
1. **`whisper_clean.py`** - Primary tool, no PyAV dependencies, robust fallback
2. **`asr_translate_srt.py`** - Two-pass pipeline: ASR + external MT (NLLB) with subtitle quality shaping  
3. **`whisper_mvp.py`** - Legacy version with PyAV (Windows compatibility issues)

### Key Design Patterns
- **Graceful degradation**: GPU→CPU fallback with `--no-fallback` override
- **Dependency isolation**: Core functionality works without heavy ML deps
- **Windows-centric**: PowerShell automation, CUDA Toolkit integration, PATH management
- **Rich UI**: Extensive use of `rich` library for progress bars and formatted output
- **FFmpeg-first**: Direct subprocess calls avoid PyAV Windows compatibility issues

## Critical Windows-Specific Knowledge

### PowerShell Execution Policy
Windows restricts script execution by default. Enable before first use:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Environment Setup Pattern
```powershell
# Always use venv Python explicitly to avoid global conflicts
.\.venv\Scripts\python.exe script_name.py [args]
```

### CUDA Dependency Chain
```
NVIDIA Driver → CUDA Toolkit 12.x → cuDNN 9 → faster-whisper GPU acceleration
```
Missing any link triggers automatic CPU fallback in `whisper_clean.py`.

### FFmpeg Integration
- Direct `subprocess` calls to avoid PyAV Windows issues
- Audio extraction: `ffmpeg -i video.ext -ar 16000 -ac 1 -vn temp.wav`
- Remux workflow: Stream copy (`-c copy`) to embed SRT without re-encoding

## Development Workflows

### Setup & Testing
```powershell
# CRITICAL: Enable PowerShell script execution (first-time setup)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Automated setup with pinned versions
pwsh -File .\setup.ps1 -Force -Model medium

# Manual dependency install (if setup.ps1 fails)
.\.venv\Scripts\pip.exe install faster-whisper --no-deps
.\.venv\Scripts\pip.exe install ctranslate2 huggingface-hub tokenizers tqdm onnxruntime rich

# Core validation (imports, FFmpeg, CUDA detection)
.\.venv\Scripts\python.exe test_clean.py

# Test primary script directly (more reliable)
.\.venv\Scripts\python.exe whisper_clean.py --help
```

### Dependency Management Philosophy
- **No PyAV**: Uses direct FFmpeg subprocess calls to avoid Windows PyAV compatibility issues
- **No-deps install**: `pip install faster-whisper --no-deps` then explicit deps to avoid PyAV
- **FFmpeg direct**: Audio extraction via subprocess, not PyAV bindings
- **Optional external MT**: transformers/torch only needed for `asr_translate_srt.py`
- **PyAV bypass**: `whisper_clean.py` works without PyAV, uses direct FFmpeg calls for Windows compatibility
- **Expected dependency warnings**: `faster-whisper` will show "requires av>=11" but script functions correctly

## Project-Specific Conventions

### Error Handling Pattern
```python
# Standard GPU fallback implementation
try:
    # GPU transcription attempt
    result = model.transcribe(audio_path, **gpu_kwargs)
except Exception as e:
    if not no_fallback:
        console.print(f"[yellow]GPU failed, attempting CPU fallback: {e}[/yellow]")
        # Retry with CPU device and compatible compute type
```

### File Processing Patterns
- **Batch folder processing**: Recursively find `SUPPORTED_FORMATS`
- **Output naming**: `video.mp4` → `video.srt`, remux → `video.subbed.mkv`
- **Temp file management**: `tempfile.NamedTemporaryFile` for audio extraction

### Configuration Patterns
```python
# Device cascade: try cuda, fallback to cpu with adjusted compute type
if device == "cuda" and compute_type == "float16":
    # CPU doesn't support pure float16, auto-switch to int8_float16
```

## Integration Points

### External Dependencies
- **FFmpeg**: Core audio extraction, must be on PATH
- **CUDA Runtime**: Optional, triggers fallback if missing
- **Hugging Face Hub**: Model caching in `%USERPROFILE%\.cache\huggingface\hub`

### Cross-Component Communication
- **Model loading**: Lazy initialization, cached instances
- **Progress reporting**: Rich Progress context managers throughout
- **Subtitle formatting**: Standard SubRip (.srt) output with quality constraints in two-pass mode

## Common Development Tasks

### Adding New Features
- Follow the `whisper_clean.py` pattern for robustness
- Implement fallback mechanisms for GPU/dependency failures
- Use Rich for user feedback and progress indication

### Debugging GPU Issues
- Check `test_clean.py` output for import/CUDA detection
- Use `--diag` flag for device/compute type confirmation
- Verify DLL availability: `where cudnn_ops64_9.dll`

### Testing Changes
- Always test both GPU and CPU paths
- Verify fallback behavior with `--no-fallback`
- Test with different model sizes and compute types
- **Ignore PyAV dependency warnings**: `whisper_clean.py` bypasses PyAV entirely

## Key Files for Context
- `whisper_clean.py`: Primary implementation, fallback logic
- `setup.ps1`: Environment creation, pinned dependencies
- `requirements.txt`: Dependency documentation with install strategies
- `docs/TROUBLESHOOTING.md`: Windows-specific CUDA/cuDNN resolution
- `test_clean.py`: Environment validation and diagnostics