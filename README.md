# Whisper MVP - GPU-Accelerated Subtitle Generation

A high-performance command-line tool that generates subtitles from video files using NVIDIA GPU acceleration and the faster-whisper library. Perfect for creating subtitles for movies, anime, lectures, and any video content.

> **🎯 Quick Tip**: Use `whisper_clean.py` for the most reliable experience. It bypasses Windows PyAV compatibility issues.

## ✨ Features

- **🚀 GPU Acceleration**: Uses NVIDIA CUDA for fast transcription with faster-whisper
- **🎬 Multiple Formats**: Supports all common video/audio formats (MP4, MKV, AVI, MP3, etc.)
- **🌍 Language Support**: Transcribe in original language or translate to English
- **📁 Batch Processing**: Process entire folders of videos automatically
- **⚡ Smart Caching**: Models are cached locally for faster subsequent runs
- **🎯 Configurable**: Choose model size, beam width, and compute precision
- **📊 Rich UI**: Beautiful terminal interface with progress bars and summaries
- **🧠 Two-Pass Translation (Optional)**: New `asr_translate_srt.py` script performs ASR first, then high-quality Machine Translation (MT) with quality shaping (CPS, line wrapping, min gaps) and fallback to Whisper internal translation.

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

You have two options: the automated PowerShell script (recommended) or manual steps.

#### Option A: Automated Script (Recommended)

`setup.ps1` creates (or refreshes) the virtual environment, installs a **known-good pinned set** of packages, optionally pre-downloads a model, and can run validation.

Basic usage:
```powershell
cd ~\Projects\Subtitle_translator
pwsh -File .\setup.ps1
```

Warm up the medium model (downloads it once so first real run is faster):
```powershell
pwsh -File .\setup.ps1 -Model medium
```

Force a clean rebuild of the virtual environment:
```powershell
## 🔧 Troubleshooting

The README now contains only a concise summary. Full details moved to [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

Quick reference:
- PyAV import error → Use `whisper_clean.py`.
- FFmpeg missing → Install via script or Chocolatey.
- CUDA warning / slow run → Install CUDA Toolkit + cuDNN or rely on CPU fallback.
- Missing cuDNN DLLs → Automatic CPU fallback (see doc for permanent fix).
- Version mismatch (StorageView) → Reinstall `ctranslate2==4.5.0` + `faster-whisper==1.2.0`.
- Force clean rebuild → Remove `.venv` and reinstall pinned deps.

Device fallback behavior:
| Scenario | Action |
|----------|--------|
| Run with `--device cuda` and CUDA incomplete | Auto fallback to CPU (compute cascade) |
| Run with `--device-order cuda,igpu,cpu` | Try each in order (internal fallback only on last) |
| Use `--no-fallback` | Abort on first device failure |

See the full guide for commands, diagnostics, and DLL remediation.

## � Additional Troubleshooting (Advanced)
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

---

## 🧠 Advanced Two-Pass ASR → MT Pipeline (`asr_translate_srt.py`)

If you need higher quality English subtitles from (e.g.) Japanese audio, use the new two-pass pipeline:

1. Pass 1: Whisper ASR (source language transcription)
2. Pass 2: External MT model (e.g. NLLB) translates clean source text
3. Shaping: Applies constraints: max chars per line, max lines, max CPS (characters per second), min duration, min gap between subtitles
4. Fallback: If MT model or dependencies unavailable, it automatically performs a second Whisper translation pass

### Why Two-Pass?
Whisper’s built-in translate mode is convenient but sometimes:
- Produces looser English phrasing
- Loses nuance for Japanese honorifics or compound verbs
- Can be harder to subtitle cleanly (long lines)

The two-pass approach improves segmentation fidelity and gives you deterministic MT constraints.

### Additional Dependencies (Optional Feature)
These are only required if you use `asr_translate_srt.py` with external MT:
```
transformers
sentencepiece
torch            # (GPU strongly recommended for large MT models)
```
They are already listed in `requirements.txt`, but you can skip installing them if you only use `whisper_clean.py`.

### Usage Examples
```powershell
# Japanese → English using NLLB distilled 600M (default)
.\.venv\Scripts\python.exe asr_translate_srt.py "anime_episode.mkv" --language ja

# Force CPU (slow) and skip external MT (use Whisper internal translation)
.\.venv\Scripts\python.exe asr_translate_srt.py "anime.mkv" --language ja --no-mt --task translate --device cpu

# Use larger MT model (needs more VRAM)
.\.venv\Scripts\python.exe asr_translate_srt.py "movie.mkv" --language ja --mt-model facebook/nllb-200-1.3B

# Tune quality constraints
.\.venv\Scripts\python.exe asr_translate_srt.py "movie.mkv" --language ja --max-line-chars 40 --max-lines 2 --max-cps 16 --min-gap 0.12

# Dry run (no file written)
.\.venv\Scripts\python.exe asr_translate_srt.py "movie.mkv" --language ja --dry-run
```

### Key Options (Two-Pass Script)
| Flag | Purpose | Default |
|------|---------|---------|
| `--language` | Source language (e.g. `ja`) | required/auto |
| `--asr-model` | Whisper model name/path | `medium` |
| `--mt-model` | Hugging Face MT model | `facebook/nllb-200-distilled-600M` |
| `--no-mt` | Disable external MT (use Whisper translation) | off |
| `--task` | If `translate` & `--no-mt`, first pass translates directly | `transcribe` |
| `--beam-size` | Whisper beam size | 5 |
| `--mt-beams` | MT beam size | 4 |
| `--batch-size` | MT batch size | 8 |
| `--max-new-tokens` | MT generation cap | 200 |
| `--max-line-chars` | Max chars per subtitle line | 42 |
| `--max-lines` | Max lines per subtitle block | 2 |
| `--max-cps` | Max characters per second | 17.0 |
| `--min-duration` | Min subtitle duration (s) | 1.0 |
| `--min-gap` | Min gap between subtitles (s) | 0.09 |
| `--vad-filter` | Enable whisper VAD | off |
| `--dry-run` | Process but do not write SRT | off |
| `--model-progress` | Show heuristic model download progress while loading | off |

### Performance Notes
- NLLB 600M can run on mid-range GPUs; larger variants (1.3B / 3.3B) need substantial VRAM.
- For constrained GPUs, lower batch size (`--batch-size 2` or `4`).
- CPU MT is functional but slow for long content.
- If MT fails mid-run, you’ll see a yellow fallback message and the pipeline will still complete with Whisper translations.
 - First-time model load can appear static; use `--model-progress` to display approximate cache download percentage.

### When to Prefer `whisper_clean.py`
| Scenario | Tool |
|----------|------|
| Fast one-shot transcription/translation | `whisper_clean.py` |
| Need subtitle shaping + CPS control | `asr_translate_srt.py` |
| No MT dependencies installed | `whisper_clean.py` |
| Strict accuracy for JP→EN phrasing | `asr_translate_srt.py` |

---

## � Remuxing Subtitles into a New Video File

Both `whisper_clean.py` and `asr_translate_srt.py` now support embedding the generated SRT into a new media file without re-encoding (stream copy). The new file name appends `.subbed` before the original extension.

### Basic Usage
```powershell
# Generate SRT and create remuxed MKV with embedded subs
.\.venv\Scripts\python.exe whisper_clean.py movie.mkv --translate --lang ja --remux

# Two-pass pipeline with remux
.\.venv\Scripts\python.exe asr_translate_srt.py movie.mkv --language ja --remux
```

### Output Naming
```
movie.mkv        -> movie.srt         (subtitle file)
				 -> movie.subbed.mkv  (remuxed container with embedded subtitle track)
```

### Flags
| Flag | Script(s) | Description |
|------|-----------|-------------|
| `--remux` | both | Enable remuxing after SRT creation |
| `--remux-language <code>` | both | Set language metadata for the new subtitle track (default: `en` if translated else source or `und`) |
| `--remux-overwrite` | both | Overwrite existing `.subbed` file if present |

### Notes
- Streams are copied (`-c copy`), so this is fast and lossless.
- Existing subtitle tracks are preserved; the new one is appended.
- If the source already has a similar subtitle track, you can still differentiate externally (e.g., track ordering).
- Use `--remux-language ja` to tag Japanese transcription or leave default when translating to English.

### Troubleshooting Remux
| Issue | Cause | Fix |
|-------|-------|-----|
| Remux failed: `ffmpeg error` | FFmpeg missing or unsupported container + SRT | Install FFmpeg / try MKV output |
| No new file created | Existing `.subbed` file and no `--remux-overwrite` | Add `--remux-overwrite` |
| Media player not showing subs | Player filtering tracks | Manually enable subtitle track in player |

---

---

## �🎯 Model Selection Guide

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
.\.venv\Scripts\python.exe whisper_clean.py "Anime Episode.mkv" --translate --lang ja --model medium

# French movie with original French subtitles
.\.venv\Scripts\python.exe whisper_clean.py "French_Film.mp4" --lang fr --model large-v3

# Auto-detect language and translate to English
.\.venv\Scripts\python.exe whisper_clean.py "Foreign_Video.avi" --translate
```

## 🔧 Troubleshooting

### Common Issues

**1. "No module named 'av._core'" Error (Most Common on Windows)**

This is a known issue with the PyAV package on Windows. The clean implementation (`whisper_clean.py`) avoids this entirely by using direct FFmpeg calls.

**2. "FFmpeg not found" Error**
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

# Clean installation method (recommended for Windows):
pip install faster-whisper --no-deps
pip install ctranslate2 tokenizers transformers rich onnxruntime av==11.0.0
```

**6. Missing CUDA / cuDNN DLLs (e.g., `cudnn_ops64_9.dll`, `cublas64_12.dll`)**

Symptoms:
```
Could not locate cudnn_ops64_9.dll
... or errors mentioning cublas / cudart / cudnnCreateTensorDescriptor
```

What’s happening:
- The NVIDIA driver lets `ctranslate2` detect a CUDA-capable GPU (so it reports a device), **but** the user‑space runtime DLLs (CUDA Toolkit + cuDNN) are not present in your PATH.
- When transcription starts, faster-whisper tries to initialize cuDNN and fails.

Automatic Fallback:
- `whisper_clean.py` now automatically retries on CPU if GPU initialization/transcription fails.
- You will see: `Attempting automatic fallback to CPU...` or `CUDA transcription failed. Retrying on CPU (fallback).`
- Disable this behavior with `--no-fallback` if you prefer a hard failure (useful in CI to enforce GPU availability).

Permanent Fix (Recommended to restore GPU speed):
1. Install the latest **CUDA Toolkit 12.x** from NVIDIA (https://developer.nvidia.com/cuda-downloads) – choose Express install.
2. Install **cuDNN 9** for Windows (requires free NVIDIA developer login):
	 - Download cuDNN (matching CUDA 12) from https://developer.nvidia.com/cudnn
	 - Extract and copy the `bin/*.dll` files into either:
		 - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.X\bin` (preferred), or
		 - A directory added to your User PATH.
3. Open a **new** PowerShell window (PATH changes require a new session).
4. Verify DLL availability:
```powershell
Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Recurse -Filter cudnn*64_9.dll
```
5. Re-run validation:
```powershell
.\.venv\Scripts\python.exe test_clean.py
```

Quick Diagnostic (optional):
```powershell
$dlls = 'cudart64_12.dll','cublas64_12.dll','cublasLt64_12.dll','cudnn_ops64_9.dll'
foreach ($d in $dlls) { "$d => " + ([bool](Get-Command $d -ErrorAction SilentlyContinue)) }
```

If GPU still fails, the script will continue on CPU (slower but functional). To force investigation instead of fallback, run with:
```powershell
.\.venv\Scripts\python.exe whisper_clean.py video.mp4 --device cuda --no-fallback
```

Edge Case – CPU float16:
- If you supplied `--compute float16` but end up on CPU (manual or fallback), the tool automatically switches to `int8_float16` because pure float16 isn’t efficient/available on CPU.

Summary Table:
| Situation | Behavior |
|-----------|----------|
| Missing cuDNN DLLs | Automatic CPU retry (unless `--no-fallback`) |
| Driver only (no toolkit) | Fallback triggers at first transcription attempt |
| Proper CUDA + cuDNN installed | Full GPU speed maintained |
| `--no-fallback` set | Program exits on CUDA failure |

Flag Documentation Addition:
| Option | Description | Default |
|--------|-------------|---------|
| `--no-fallback` | Disable automatic CUDA→CPU fallback | Disabled (fallback enabled) |

## 🛠 Additional Troubleshooting (Advanced)

### A. NumPy "source directory" Import Error
If you see:
```
Error importing numpy: you should not try to import numpy from its source directory
```
It usually means the interpreter is resolving NumPy from a global install instead of your virtual environment.

Fix:
```powershell
# Ensure you're in the project root
cd ~\Projects\Subtitle_translator

# Explicitly use venv Python for installs
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall numpy

# Verify it points inside .venv
.\.venv\Scripts\python.exe -c "import numpy, sys; print(numpy.__file__)"

# You should see a path like:
# ...\Subtitle_translator\.venv\Lib\site-packages\numpy\__init__.py

# If it still points to a global path, reactivate the environment:
.\.venv\Scripts\Activate.ps1
```

### B. ctranslate2 AttributeError (StorageView)
If faster-whisper import fails with something like:
```
AttributeError: module 'ctranslate2' has no attribute 'StorageView'
```
This is a version mismatch between `faster-whisper` and `ctranslate2`.

Resolution (working combo):
```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall "ctranslate2==4.5.0" "faster-whisper==1.2.0"
```
Re-run validation:
```powershell
.\.venv\Scripts\python.exe test_clean.py
```

### C. Clean Rebuild (Last Resort)
```powershell
# From project root
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install faster-whisper --no-deps
pip install ctranslate2==4.5.0 tokenizers transformers rich onnxruntime av==11.0.0
.\.venv\Scripts\python.exe test_clean.py
```

### Performance Optimization

**For High-End GPUs (RTX 3080+):**
```powershell
.\.venv\Scripts\python.exe whisper_clean.py video.mp4 --model large-v3 --compute float16 --beam 10
```

**For Mid-Range GPUs (GTX 1660, RTX 2060):**
```powershell
.\.venv\Scripts\python.exe whisper_clean.py video.mp4 --model medium --compute float16 --beam 5
```

**For Low-End GPUs (GTX 1060, GTX 1650):**
```powershell
.\.venv\Scripts\python.exe whisper_clean.py video.mp4 --model small --compute int8_float16 --beam 3
```

## 📁 Example Workflows

### 1. Processing Japanese Anime Collection
```powershell
# Process entire anime folder with Japanese to English translation
.\.venv\Scripts\python.exe whisper_clean.py "D:\Anime\Season 1" --translate --lang ja --model medium

# Result: Creates .srt files for all video files in the folder
```

### 2. Creating Subtitles for Lectures
```powershell
# High-quality English transcription for educational content
.\.venv\Scripts\python.exe whisper_clean.py "Lecture_Recording.mp4" --model large-v3 --beam 8 --lang en
```

### 3. Batch Processing Movie Collection
```powershell
# Process all movies with automatic language detection and translation
.\.venv\Scripts\python.exe whisper_clean.py "C:\Movies\Foreign Films" --translate --model medium
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
```bash
# Set custom cache directory
export HF_HOME="D:\AI_Models"
```

### GPU Memory Management
```bash
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