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

**Option A: Using Chocolatey (Recommended)**
```powershell
# Install Chocolatey if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install FFmpeg
choco install ffmpeg
```

**Option B: Manual Installation**
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your Windows PATH

### 2. Clone and Setup

```powershell
# Clone the repository
git clone <your-repo-url>
cd Subtitle_translator

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify CUDA Setup

```powershell
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
```

If CUDA is not available, install PyTorch with CUDA support:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Usage

### Basic Commands

```powershell
# Basic transcription (same language subtitles)
python whisper_mvp.py video.mp4

# Translate to English
python whisper_mvp.py foreign_video.mkv --translate

# Specify source language for better accuracy
python whisper_mvp.py anime.mkv --translate --lang ja

# Use different model size
python whisper_mvp.py video.mp4 --model large-v3

# Custom output path
python whisper_mvp.py video.mp4 --output "subtitles/video_subs.srt"
```

### Advanced Options

```powershell
# High-quality processing
python whisper_mvp.py video.mp4 --model large-v3 --beam 10 --compute float16

# Batch process entire folder
python whisper_mvp.py "C:\Videos\Movies" --model medium --translate

# Memory-optimized for smaller GPUs
python whisper_mvp.py video.mp4 --model small --compute int8_float16
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

**1. "CUDA not available" Error**
```powershell
# Install CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. "FFmpeg not found" Error**
```powershell
# Check if FFmpeg is in PATH
ffmpeg -version

# If not found, reinstall FFmpeg and add to PATH
```

**3. "Out of memory" Error**
- Use smaller model: `--model small` or `--model base`
- Use lower precision: `--compute int8_float16`
- Close other GPU-intensive applications

**4. "Model download failed" Error**
- Check internet connection
- Try downloading manually:
```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu')"
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