# Whisper MVP - GPU-Accelerated Subtitle Generation

A high-performance command-line tool that generates subtitles from video files using NVIDIA GPU acceleration and the faster-whisper library. Perfect for creating subtitles for movies, anime, lectures, and any video content.

## ✨ Features

- **🚀 GPU Acceleration**: Uses NVIDIA CUDA for fast transcription with faster-whisper
- **🎬 Multiple Formats**: Supports all common video/audio formats (MP4, MKV, AVI, MP3, etc.)
- **🌍 Language Support**: Transcribe in original language or translate to English
- **📁 Batch Processing**: Process entire folders of videos automatically
- **⚡ Smart Caching**: Models are cached locally for faster subsequent runs
- **🎯 Configurable**: Choose model size, beam width, and compute precision
- **📊 Rich UI**: Beautiful terminal interface with progress bars and summaries

## 📋 Requirements

### System Requirements
- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with CUDA support (GTX 1060 or newer recommended)
- **8GB RAM minimum** (16GB recommended for large models)
- **2-10GB storage** (for model caching)

### Software Dependencies
- **Python 3.8+** (Python 3.9+ recommended)
- **CUDA Toolkit** (installed automatically with PyTorch)
- **FFmpeg** (for audio extraction)

## 🚀 Quick Start

### 1. Install FFmpeg

**Option A: Automatic PowerShell Installation (Recommended)**
```powershell
# Create FFmpeg directory and download/extract in one go
New-Item -ItemType Directory -Path "C:\ffmpeg" -Force
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$zipPath = "$env:TEMP\ffmpeg.zip"
Invoke-WebRequest -Uri $ffmpegUrl -OutFile $zipPath -UseBasicParsing
Expand-Archive -Path $zipPath -DestinationPath "C:\ffmpeg" -Force

# Add to PATH permanently
$ffmpegDir = Get-ChildItem "C:\ffmpeg" -Directory | Select-Object -First 1
$binPath = "$($ffmpegDir.FullName)\bin"
[Environment]::SetEnvironmentVariable("PATH", [Environment]::GetEnvironmentVariable("PATH", "User") + ";$binPath", "User")
$env:PATH = "$binPath;$env:PATH"

# Verify installation
ffmpeg -version
```

**Option B: Using Chocolatey (If Available)**
```powershell
# Install FFmpeg via Chocolatey (requires admin privileges)
choco install ffmpeg
```

**Option C: Manual Installation**
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg\ffmpeg-x.x-essentials_build\`
3. Add `C:\ffmpeg\ffmpeg-x.x-essentials_build\bin` to your Windows PATH

### 2. Setup Python Environment

```powershell
# Navigate to project directory
cd Subtitle_translator

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Setup

```powershell
# Run the validation script to check everything is working
.\.venv\Scripts\python.exe validate_setup.py
```

The validation script will check:
- ✅ Python version compatibility
- ✅ All required packages (faster-whisper, ffmpeg-python, rich)
- ✅ FFmpeg binary availability
- ✅ CUDA support (optional but recommended)

## 📖 Usage

### Basic Commands

```powershell
# Basic transcription (same language subtitles)
.\.venv\Scripts\python.exe whisper_mvp.py video.mp4

# Translate to English
.\.venv\Scripts\python.exe whisper_mvp.py foreign_video.mkv --translate

# Specify source language for better accuracy
.\.venv\Scripts\python.exe whisper_mvp.py anime.mkv --translate --lang ja

# Use different model size
.\.venv\Scripts\python.exe whisper_mvp.py video.mp4 --model large-v3

# Custom output path
.\.venv\Scripts\python.exe whisper_mvp.py video.mp4 --output "subtitles/video_subs.srt"
```

**Note**: Always use `.\.venv\Scripts\python.exe` to ensure you're using the virtual environment with all the required packages installed.

### Advanced Options

```powershell
# High-quality processing
.\.venv\Scripts\python.exe whisper_mvp.py video.mp4 --model large-v3 --beam 10 --compute float16

# Batch process entire folder
.\.venv\Scripts\python.exe whisper_mvp.py "C:\Videos\Movies" --model medium --translate

# Memory-optimized for smaller GPUs
.\.venv\Scripts\python.exe whisper_mvp.py video.mp4 --model small --compute int8_float16
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Input video file or folder path | Required |
| `--output` `-o` | Output SRT file path | `{input_name}.srt` |
| `--model` `-m` | Whisper model size | `medium` |
| `--lang` | Source language code (e.g., `ja`, `fr`, `de`) | Auto-detect |
| `--translate` | Translate to English instead of transcribe | False |
| `--beam` | Beam size for decoding (1-10) | 5 |
| `--compute` | Compute precision (`float16`, `int8_float16`) | `float16` |
| `--device` | Processing device | `cuda` |

## 🎯 Model Selection Guide

| Model | Size | VRAM | Speed | Quality | Best For |
|-------|------|------|-------|---------|----------|
| `tiny` | ~39MB | 1GB | Fastest | Basic | Quick drafts, low-resource systems |
| `base` | ~74MB | 1GB | Fast | Good | General use, older GPUs |
| `small` | ~244MB | 2GB | Moderate | Better | Balanced performance |
| `medium` | ~769MB | 5GB | Slower | Great | **Recommended default** |
| `large-v3` | ~1550MB | 10GB | Slowest | Best | Professional/production use |

## 🌏 Language Support

### Common Language Codes
- `en` - English
- `ja` - Japanese  
- `zh` - Chinese
- `ko` - Korean
- `fr` - French
- `de` - German
- `es` - Spanish
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ar` - Arabic
- `hi` - Hindi

### Usage Examples

```powershell
# Japanese anime to English subtitles
python whisper_mvp.py "Anime Episode.mkv" --translate --lang ja --model medium

# French movie with original French subtitles
python whisper_mvp.py "French_Film.mp4" --lang fr --model large-v3

# Auto-detect language and translate to English
python whisper_mvp.py "Foreign_Video.avi" --translate
```

## 🔧 Troubleshooting

### Common Issues

**1. "FFmpeg not found" Error**
```powershell
# First, check if FFmpeg is installed
ffmpeg -version

# If not found, use the automatic installation:
New-Item -ItemType Directory -Path "C:\ffmpeg" -Force
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$zipPath = "$env:TEMP\ffmpeg.zip"
Invoke-WebRequest -Uri $ffmpegUrl -OutFile $zipPath -UseBasicParsing
Expand-Archive -Path $zipPath -DestinationPath "C:\ffmpeg" -Force
$ffmpegDir = Get-ChildItem "C:\ffmpeg" -Directory | Select-Object -First 1
$binPath = "$($ffmpegDir.FullName)\bin"
[Environment]::SetEnvironmentVariable("PATH", [Environment]::GetEnvironmentVariable("PATH", "User") + ";$binPath", "User")
$env:PATH = "$binPath;$env:PATH"
```

**2. "choco command not found" Error**
- Chocolatey may not be properly installed or PATH not refreshed
- Use the automatic PowerShell FFmpeg installation instead (Option A above)
- Or restart PowerShell/Command Prompt after installing Chocolatey

**3. "CUDA not available" Warning**
- This is often okay - faster-whisper has built-in CUDA support
- GPU acceleration will still work if you have NVIDIA drivers installed
- Only install PyTorch separately if you specifically need it for other purposes

**4. "Out of memory" Error**
- Use smaller model: `--model small` or `--model base`
- Use lower precision: `--compute int8_float16`
- Close other GPU-intensive applications

**5. "Package installation failed" Error**
```powershell
# Upgrade pip and try again
python -m pip install --upgrade pip
pip install -r requirements.txt

# If still failing, install packages individually:
pip install faster-whisper
pip install ffmpeg-python
pip install rich
```

**6. "Model download failed" Error**
- Check internet connection
- Try downloading with CPU first to test:
```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu')"
```

### Performance Optimization

**For High-End GPUs (RTX 3080+):**
```powershell
python whisper_mvp.py video.mp4 --model large-v3 --compute float16 --beam 10
```

**For Mid-Range GPUs (GTX 1660, RTX 2060):**
```powershell
python whisper_mvp.py video.mp4 --model medium --compute float16 --beam 5
```

**For Low-End GPUs (GTX 1060, GTX 1650):**
```powershell
python whisper_mvp.py video.mp4 --model small --compute int8_float16 --beam 3
```

## 📁 Example Workflows

### 1. Processing Japanese Anime Collection
```powershell
# Process entire anime folder with Japanese to English translation
python whisper_mvp.py "D:\Anime\Season 1" --translate --lang ja --model medium

# Result: Creates .srt files for all video files in the folder
```

### 2. Creating Subtitles for Lectures
```powershell
# High-quality English transcription for educational content
python whisper_mvp.py "Lecture_Recording.mp4" --model large-v3 --beam 8 --lang en
```

### 3. Batch Processing Movie Collection
```powershell
# Process all movies with automatic language detection and translation
python whisper_mvp.py "C:\Movies\Foreign Films" --translate --model medium
```

## 🔄 Output Format

Generated `.srt` files follow the standard SubRip format:

```
1
00:00:00,000 --> 00:00:04,560
This is the first subtitle line.

2
00:00:04,560 --> 00:00:08,920
This is the second subtitle line
that can span multiple lines.

3
00:00:08,920 --> 00:00:12,480
Final subtitle segment.
```

## 🎛️ Advanced Configuration

### Model Caching Location
Models are cached in the default Hugging Face cache directory:
- Windows: `%USERPROFILE%\.cache\huggingface\hub`

### Custom Cache Directory
```python
# Set custom cache directory
export HF_HOME="D:\AI_Models"
```

### GPU Memory Management
```python
# For systems with multiple GPUs
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - High-performance Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal interface
- [FFmpeg](https://ffmpeg.org/) - Multimedia processing

## 📞 Support

- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Open an issue with the "enhancement" label  
- 📧 **Questions**: Check existing issues or open a new discussion

---

**Happy subtitling!** 🎬✨