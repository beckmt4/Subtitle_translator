# Subtitle Translator: GPU-Accelerated Subtitle Generation

A high-performance Python package for generating subtitles from video files, with support for GPU acceleration using NVIDIA CUDA and the faster-whisper library. Perfect for creating subtitles for movies, anime, lectures, and any video content.

## ✨ Features

- **🚀 GPU Acceleration**: Uses NVIDIA CUDA for fast transcription with faster-whisper
- **🎬 Multiple Formats**: Supports all common video/audio formats (MP4, MKV, AVI, MP3, etc.)
- **🌍 Language Support**: Transcribe in original language or translate to English
- **📁 Batch Processing**: Process entire folders of videos automatically
- **⚡ Smart Caching**: Models are cached locally for faster subsequent runs
- **🎯 Configurable**: Choose model size, beam width, and compute precision
- **📊 Rich UI**: Beautiful terminal interface with progress bars and summaries
- **🧠 Two-Pass Translation**: High-quality ASR + MT pipeline with subtitle quality shaping (CPS, line wrapping, min gaps)
- **🧩 Modular Design**: Structured Python package for easy integration into other projects

## 📋 Requirements

### System Requirements
- **Windows 10/11** (64-bit) - Primary focus, but also works on Linux/macOS
- **NVIDIA GPU** with CUDA support recommended (automatic CPU fallback available)
- **8GB RAM minimum** (16GB recommended for large models)
- **2-10GB storage** (for model caching)

### Software Dependencies
- **Python 3.8+** (Python 3.10+ recommended)
- **CUDA Toolkit** (installed automatically with PyTorch - optional for GPU acceleration)
- **FFmpeg** (for audio extraction and remuxing)

## 🚀 Quick Start

### 1. Installation

**Option A: Quick Install Script (Windows PowerShell)**
```powershell
# Enable script execution if needed (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run the installer
.\Install-SubtitleTranslator.ps1 -Model medium
```

**Option B: Manual Installation**
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install package
pip install -e .
```

### 2. Verify FFmpeg Installation

FFmpeg is required for audio extraction and remuxing. Make sure it's installed and available in your PATH.

```bash
# Check if FFmpeg is installed
ffmpeg -version
```

### 3. Basic Usage

**Transcribe a Video File**
```bash
# Transcribe in original language
subtitle-translator transcribe video.mp4

# Specify language and model size
subtitle-translator transcribe video.mp4 --language ja --model medium
```

**Translate Subtitles**
```bash
# Two-pass ASR + MT with quality controls
subtitle-translator translate video.mp4 --target-language eng_Latn

# With additional options
subtitle-translator translate video.mp4 --target-language eng_Latn --asr-model medium --max-cps 20 --remux
```

**List Available Quality Profiles**
```bash
subtitle-translator profiles
```

## 📖 Command Reference

### Core Commands

**Transcribe**: Generate subtitles in original language
```bash
subtitle-translator transcribe VIDEO_PATH [OPTIONS]
```

**Translate**: Generate translated subtitles with quality controls
```bash
subtitle-translator translate VIDEO_PATH [OPTIONS]
```

**Profiles**: List available subtitle quality profiles
```bash
subtitle-translator profiles
```

### Common Options

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output SRT path (default: auto-generated) |
| `--language` | Source language code (default: auto-detect) |
| `--model` | Whisper model size: tiny, base, small, medium, large-v3 (default: medium) |
| `--device` | Device to use: cuda or cpu (default: cuda) |
| `--compute-type` | Compute type: float16, int8_float16, int8 (default: auto) |
| `--target-language` | Target language for translation (default: eng_Latn) |
| `--remux` | Remux subtitles into video file (default: False) |
| `--profile` | Use predefined quality profile (default: None) |

### Translation Quality Options

| Option | Description |
|--------|-------------|
| `--max-line-chars` | Maximum characters per line (default: 42) |
| `--max-lines` | Maximum lines per subtitle (default: 2) |
| `--max-cps` | Maximum characters per second (default: 20.0) |
| `--min-duration` | Minimum subtitle duration in seconds (default: 0.5) |
| `--min-gap` | Minimum gap between subtitles in seconds (default: 0.2) |

## 📚 API Reference

### Core Classes

- **WhisperASR**: Wrapper for faster-whisper ASR model
- **TranslationEngine**: Machine Translation engine using transformer models
- **SubtitleQualityShaper**: Applies readability constraints to subtitles

### Key Functions

- **extract_audio**: Extract audio from video files
- **remux_subtitles**: Embed subtitles into video files
- **read_srt/write_srt**: SRT file I/O operations
- **get_profile**: Load subtitle quality profiles

## 🧪 Development

### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/subtitle-translator.git
cd subtitle-translator

# Install in development mode
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest
```

### Build Binary
```bash
python build_binary.py
```

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Base ASR model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2-based Whisper implementation
- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - No Language Left Behind translation models