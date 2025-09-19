# Subtitle Translator Codebase Documentation

This document provides a comprehensive overview of the Subtitle Translator codebase, explaining its architecture, components, and usage patterns.

## Project Overview

Subtitle Translator is a Windows-first GPU-accelerated subtitle generation tool using OpenAI's Whisper model via the `faster-whisper` library. The project solves PyAV compatibility issues on Windows with a clean FFmpeg-based approach and provides automatic CUDA→CPU fallback for robustness.

### Purpose

To provide an efficient, reliable tool for:
- Generating subtitles from video files
- Transcribing audio in the original language
- Translating speech to English
- Processing batches of media files
- Creating high-quality subtitles with configurable formatting

### Core Value Proposition

1. **Windows Compatibility**: Direct FFmpeg integration bypasses PyAV issues
2. **Performance**: GPU acceleration with graceful fallback to CPU
3. **Usability**: Rich UI, batch processing, and quality constraints
4. **Robustness**: Handles dependency and hardware limitations gracefully

## System Architecture

### Major Components

1. **`whisper_clean.py`** - Primary implementation with Windows compatibility
   - Uses FFmpeg directly for audio extraction
   - No PyAV dependency, robust GPU→CPU fallback
   - Rich UI with progress tracking

2. **`asr_translate_srt.py`** - Two-pass pipeline for high-quality translations
   - ASR in original language with faster-whisper
   - External MT using Hugging Face models (e.g., NLLB)
   - Subtitle quality shaping (line wrapping, max CPS, etc.)
   - Fallback to Whisper's internal translation

3. **`whisper_mvp.py`** - Legacy implementation with PyAV
   - Windows compatibility issues due to PyAV dependency
   - Maintained for reference but not recommended for Windows users

4. **Setup & Diagnostics**
   - `setup.ps1`: PowerShell environment creation script
   - `test_clean.py`: Environment validation
   - `scripts/check_cuda.ps1`: CUDA/cuDNN diagnostics

### Dependency Structure

```
Core Dependencies:
  faster-whisper <- ctranslate2, tokenizers
  rich (UI library)
  FFmpeg (external binary)

Optional Extensions:
  transformers, sentencepiece <- external MT in asr_translate_srt.py
  torch <- GPU acceleration for transformers
  
Hardware Dependencies:
  NVIDIA Driver -> CUDA Toolkit -> cuDNN -> GPU Acceleration
```

## Key Implementation Details

### Audio Processing Workflow

1. **Input Validation**: Check for supported formats
2. **Audio Extraction**: 
   - Direct FFmpeg subprocess calls
   - Convert to 16kHz mono WAV
   - Use temp files for intermediate processing
3. **Transcription/Translation**:
   - Load faster-whisper model with appropriate device/compute type
   - Try GPU first with fallback to CPU if needed
   - Process audio through Whisper model
4. **Output Generation**:
   - Format as SRT subtitle file
   - Optional subtitle shaping in two-pass mode
   - Optional remuxing to embed subtitles in video

### GPU Acceleration & Fallback System

```python
# Pseudo-code representation
try:
    # Attempt GPU transcription
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    result = model.transcribe(audio_path)
except Exception:
    if not no_fallback:
        # Fallback to CPU with compatible compute type
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        result = model.transcribe(audio_path)
```

### Two-Pass Translation Pipeline

1. **ASR Step**: Transcribe audio in original language
2. **MT Step**: Translate transcription using external model
   - Load seq2seq model (NLLB, M2M100) with appropriate device
   - Tokenize and translate segments
   - Apply quality constraints (formatting, timing)
3. **Fallback**: Use Whisper's internal translation if external MT fails

### Windows Compatibility Strategy

- Direct FFmpeg subprocess calls instead of PyAV bindings
- PowerShell scripts for environment setup
- CUDA detection and path management
- Detailed troubleshooting documentation

## Code Components in Detail

### 1. `whisper_clean.py`

The primary implementation with direct FFmpeg integration for Windows compatibility.

**Key Classes & Functions:**

- `WhisperMVPClean`: Main class encapsulating the transcription functionality
  - `check_ffmpeg()`: Validates FFmpeg availability
  - `load_model()`: Initializes the Whisper model with appropriate device/compute
  - `extract_audio()`: Uses FFmpeg to extract audio from video files
  - `transcribe()`: Processes audio through the model with fallback support
  - `generate_srt()`: Formats transcription results as SRT subtitles

**Usage Patterns:**

```python
whisper = WhisperMVPClean()
whisper.load_model("medium", device="cuda", compute_type="float16")
result = whisper.transcribe("video.mp4", language="ja", translate=True)
whisper.generate_srt(result, "video.srt")
```

### 2. `asr_translate_srt.py`

Two-pass pipeline for high-quality translations with subtitle formatting.

**Key Functions:**

- `load_mt_model()`: Initializes the translation model
- `translate_segments()`: Processes transcription segments through MT
- `generate_srt_with_constraints()`: Creates SRT with quality constraints
- `process_file()`: Orchestrates the full pipeline

**Quality Constraints:**

- Maximum characters per line
- Maximum lines per segment
- Maximum characters per second (CPS)
- Minimum duration
- Inter-segment gaps

### 3. `setup.ps1`

PowerShell script for environment setup and dependency management.

**Key Features:**

- Virtual environment creation
- Pinned dependency installation
- FFmpeg validation
- Optional model warmup
- Environment testing

### 4. Testing & Diagnostics

- `test_clean.py`: Validates environment and dependencies
- `check_cuda.ps1`: Diagnoses CUDA/cuDNN setup and availability

## Usage Examples

### Basic Transcription

```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Transcribe in original language
python whisper_clean.py video.mp4

# Translate to English
python whisper_clean.py video.mp4 --translate

# Specify language for better accuracy
python whisper_clean.py video.mp4 --language ja
```

### Batch Processing

```powershell
# Process all videos in a folder
python whisper_clean.py C:\Videos\anime --language ja --translate

# Process with larger model for accuracy
python whisper_clean.py C:\Videos\lectures --model large-v3
```

### Two-Pass Translation

```powershell
# ASR + external MT with quality shaping
python asr_translate_srt.py video.mp4 --language ja --mt-model facebook/nllb-200-distilled-600M

# With custom constraints
python asr_translate_srt.py video.mp4 --language ja --max-cps 20 --max-line-length 42
```

## Error Handling & Diagnostics

### Common Error Patterns

1. **GPU Acceleration Failures**:
   - CUDA/cuDNN missing or incompatible
   - GPU memory limitations
   - Driver issues

2. **Dependency Issues**:
   - PyAV compatibility problems on Windows
   - Version conflicts between faster-whisper and ctranslate2
   - Missing optional dependencies for advanced features

### Diagnostic Approaches

1. **Environment Validation**:
   ```powershell
   .\.venv\Scripts\python.exe test_clean.py
   ```

2. **GPU Setup Verification**:
   ```powershell
   pwsh -File .\scripts\check_cuda.ps1
   ```

3. **Direct DLL Validation**:
   ```powershell
   where cudnn_ops64_9.dll
   ```

## Development Guidelines

### Adding Features

1. Prefer extending `whisper_clean.py` over legacy implementations
2. Implement robust fallback mechanisms for all new features
3. Use the Rich library for consistent UI experiences
4. Maintain Windows compatibility as a primary concern

### Testing Strategy

1. Test both GPU and CPU execution paths
2. Verify behavior with different model sizes
3. Test with and without fallback (`--no-fallback`)
4. Validate against various media formats

### Code Style

- Follow PEP 8 conventions
- Use type hints for better maintainability
- Provide rich user feedback and progress indication
- Implement graceful error handling and diagnostics

## Appendix

### Performance Considerations

- GPU memory usage increases with model size:
  - tiny: ~1GB VRAM
  - base: ~1GB VRAM
  - small: ~2GB VRAM
  - medium: ~5GB VRAM
  - large-v3: ~10GB VRAM

- Compute types affect accuracy and performance:
  - float16: Best accuracy, requires modern GPU
  - int8_float16: Good balance, compatible with most GPUs
  - int8: Fastest, slight accuracy loss

### Future Enhancements

- Multi-language subtitle generation
- Custom subtitle styling
- Integration with video editing workflows
- Web UI for non-technical users
- Enhanced batch processing capabilities