# API Reference - Subtitle Translator

This document provides a detailed API reference for the key modules and classes in the Subtitle Translator project.

## 1. whisper_clean.py

### `WhisperMVPClean` Class

Primary implementation with direct FFmpeg integration for Windows compatibility.

#### Initialization

```python
whisper = WhisperMVPClean()
```

#### Methods

##### `check_ffmpeg() -> bool`
Checks if FFmpeg is available in PATH.

- **Returns**: `bool` - `True` if FFmpeg is available, `False` otherwise

##### `load_model(model_name: str, device: str = "cuda", compute_type: str = "float16", asr_options: dict = None) -> None`
Loads a Whisper model for transcription/translation.

- **Parameters**:
  - `model_name`: Model size to use (tiny, base, small, medium, large-v3)
  - `device`: Device to use ("cuda" or "cpu")
  - `compute_type`: Computation type ("float16", "int8_float16", or "int8")
  - `asr_options`: Additional options for the model
- **Returns**: `None`

##### `extract_audio(video_path: str, output_path: str = None) -> str`
Extracts audio from a video file using FFmpeg.

- **Parameters**:
  - `video_path`: Path to the video file
  - `output_path`: Optional path for the extracted audio
- **Returns**: `str` - Path to the extracted audio

##### `transcribe(input_path: str, language: str = None, translate: bool = False, no_fallback: bool = False, **kwargs) -> dict`
Transcribes audio from a file or directory.

- **Parameters**:
  - `input_path`: Path to the audio/video file or directory
  - `language`: Language code (optional, auto-detected if not provided)
  - `translate`: Whether to translate to English
  - `no_fallback`: If `True`, do not fall back to CPU on error
  - `**kwargs`: Additional keyword arguments for the transcription
- **Returns**: `dict` - Transcription results

##### `generate_srt(result: dict, output_path: str) -> None`
Generates an SRT subtitle file from transcription results.

- **Parameters**:
  - `result`: Transcription results from `transcribe()`
  - `output_path`: Path to write the SRT file
- **Returns**: `None`

##### `process_batch(input_path: str, language: str = None, translate: bool = False, remux: bool = False, **kwargs) -> List[str]`
Processes a batch of files from a directory.

- **Parameters**:
  - `input_path`: Directory path containing media files
  - `language`: Language code
  - `translate`: Whether to translate to English
  - `remux`: Whether to remux subtitles into the original video
  - `**kwargs`: Additional keyword arguments
- **Returns**: `List[str]` - List of processed subtitle file paths

##### `remux_subtitles(video_path: str, subtitle_path: str, output_path: str = None) -> str`
Embeds subtitle file into video.

- **Parameters**:
  - `video_path`: Path to the video file
  - `subtitle_path`: Path to the subtitle file
  - `output_path`: Optional path for the output file
- **Returns**: `str` - Path to the remuxed video

#### Command Line Interface

```
usage: whisper_clean.py [-h] [--model MODEL] [--language LANGUAGE] [--translate] [--device DEVICE] [--compute-type COMPUTE_TYPE] [--beam-size BEAM_SIZE] [--no-fallback] [--remux] [--verbose] [--diag] input [output]

Generate subtitles from video files using faster-whisper.

positional arguments:
  input                 Input video file or directory
  output                Output subtitle file or directory (optional)

options:
  -h, --help            show this help message and exit
  --model MODEL         Whisper model size (tiny, base, small, medium, large-v3)
  --language LANGUAGE   Source language code (iso639-1)
  --translate           Translate to English
  --device DEVICE       Device to use (cuda or cpu)
  --compute-type COMPUTE_TYPE
                        Compute type (float16, int8_float16, int8)
  --beam-size BEAM_SIZE
                        Beam size for decoding
  --no-fallback         Disable CPU fallback on GPU error
  --remux               Remux subtitles into the original video
  --verbose             Show verbose output
  --diag                Show diagnostic information
```

## 2. asr_translate_srt.py

Two-pass pipeline for high-quality translations with subtitle formatting.

### Key Functions

#### `load_mt_model(model_name: str, device: str) -> tuple`
Loads a translation model and tokenizer.

- **Parameters**:
  - `model_name`: HF model name/path for translation
  - `device`: Device to use ("cuda" or "cpu")
- **Returns**: `tuple` - (model, tokenizer) or (None, None) on failure

#### `translate_segments(segments: List[dict], src_lang: str, tgt_lang: str, model, tokenizer, max_batch_tokens: int = 512) -> List[str]`
Translates transcribed segments using an external MT model.

- **Parameters**:
  - `segments`: List of transcript segments
  - `src_lang`: Source language code
  - `tgt_lang`: Target language code
  - `model`: Translation model
  - `tokenizer`: Translation tokenizer
  - `max_batch_tokens`: Maximum tokens per batch
- **Returns**: `List[str]` - Translated segments

#### `generate_srt_with_constraints(segments: List[dict], translations: List[str], output_path: str, max_line_length: int = 42, max_lines: int = 2, min_duration: float = 0.5, max_cps: float = 20.0, min_gap: float = 0.2) -> None`
Creates an SRT file with quality constraints.

- **Parameters**:
  - `segments`: Transcription segments
  - `translations`: Translated text for each segment
  - `output_path`: Path to write the SRT file
  - `max_line_length`: Maximum characters per line
  - `max_lines`: Maximum lines per segment
  - `min_duration`: Minimum duration for a segment
  - `max_cps`: Maximum characters per second
  - `min_gap`: Minimum gap between segments
- **Returns**: `None`

#### `process_file(input_path: str, output_path: str, language: str = None, whisper_model: str = "medium", mt_model: str = None, device: str = "cuda", compute_type: str = "float16", max_line_length: int = 42, max_cps: float = 20.0, min_gap: float = 0.2) -> None`
Processes a single file with the two-pass pipeline.

- **Parameters**:
  - `input_path`: Path to the input file
  - `output_path`: Path to the output SRT file
  - `language`: Source language code
  - `whisper_model`: Whisper model size
  - `mt_model`: Translation model name
  - `device`: Device to use
  - `compute_type`: Computation type
  - `max_line_length`: Maximum characters per line
  - `max_cps`: Maximum characters per second
  - `min_gap`: Minimum gap between segments
- **Returns**: `None`

#### Command Line Interface

```
usage: asr_translate_srt.py [-h] [--whisper-model WHISPER_MODEL] [--mt-model MT_MODEL] [--language LANGUAGE] [--device DEVICE] [--compute-type COMPUTE_TYPE] [--max-line-length MAX_LINE_LENGTH] [--max-cps MAX_CPS] [--min-gap MIN_GAP] [--verbose] input [output]

Two-pass pipeline: 1) ASR via faster-whisper, 2) MT via Hugging Face seq2seq model.

positional arguments:
  input                 Input video file
  output                Output subtitle file (optional)

options:
  -h, --help            show this help message and exit
  --whisper-model WHISPER_MODEL
                        Whisper model size (tiny, base, small, medium, large-v3)
  --mt-model MT_MODEL   Translation model (HuggingFace model ID or path)
  --language LANGUAGE   Source language code (iso639-1)
  --device DEVICE       Device to use (cuda or cpu)
  --compute-type COMPUTE_TYPE
                        Compute type (float16, int8_float16, int8)
  --max-line-length MAX_LINE_LENGTH
                        Maximum characters per line
  --max-cps MAX_CPS     Maximum characters per second
  --min-gap MIN_GAP     Minimum gap between segments in seconds
  --verbose             Show verbose output
```

## 3. Common Utility Functions

These utility functions are used across multiple scripts in the project.

### FFmpeg Integration

#### Audio Extraction
```python
def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extract audio from video using FFmpeg subprocess."""
    if not output_path:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name
    
    cmd = ['ffmpeg', '-i', video_path, '-ar', '16000', '-ac', '1', 
           '-c:a', 'pcm_s16le', '-y', output_path]
    
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
```

#### Remuxing Subtitles
```python
def remux_subtitles(video_path: str, subtitle_path: str, output_path: str = None) -> str:
    """Remux subtitles into video using FFmpeg."""
    if not output_path:
        video_stem = Path(video_path).stem
        output_path = str(Path(video_path).with_name(f"{video_stem}.subbed.mkv"))
    
    cmd = ['ffmpeg', '-i', video_path, '-i', subtitle_path, '-c', 'copy',
           '-c:s', 'srt', '-metadata:s:s', 'language=eng', '-y', output_path]
    
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
```

### File Processing

#### Finding Media Files
```python
def find_media_files(directory_path: str, supported_formats: set) -> List[str]:
    """Find all media files in a directory that match supported formats."""
    media_files = []
    directory = Path(directory_path)
    
    for path in directory.rglob('*'):
        if path.suffix.lower() in supported_formats:
            media_files.append(str(path))
            
    return sorted(media_files)
```

#### SRT Generation
```python
def generate_srt(segments: list, output_path: str) -> None:
    """Generate SRT file from segments."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg['text'].strip()
            
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
```

## 4. Script Interfaces and Extensions

### setup.ps1

PowerShell script for environment setup with these parameters:

- `-Force`: Recreate virtual environment
- `-Python`: Python executable path
- `-Model`: Whisper model to download
- `-NoValidate`: Skip validation
- `-NoFFmpegCheck`: Skip FFmpeg check
- `-Quiet`: Suppress non-error output

### test_clean.py

Environment validation script with:

- Import testing
- FFmpeg validation
- GPU availability check
- cuDNN verification
- Model loading test

### check_cuda.ps1

CUDA diagnostics script with:

- NVIDIA driver detection
- CUDA DLL verification
- cuDNN presence check
- Python GPU support testing

## 5. Constants and Configuration

### Model Sizes
```python
AVAILABLE_MODELS = {'tiny', 'base', 'small', 'medium', 'large-v3'}
```

### Supported Media Formats
```python
SUPPORTED_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', 
                     '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
```

### Compute Types
```python
COMPUTE_TYPES = {'float16', 'int8_float16', 'int8'}
```

### Device Options
```python
DEVICES = {'cuda', 'cpu', 'auto'}
```

### Default Quality Constraints
```python
DEFAULT_MAX_LINE_LENGTH = 42
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_DURATION = 0.5
DEFAULT_MAX_CPS = 20.0
DEFAULT_MIN_GAP = 0.2
```

## 6. Environment Variables

The following environment variables can affect behavior:

- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible
- `HF_HOME`: Override Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Override transformers model cache
- `CT2_CACHE`: Override ctranslate2 model cache

## 7. Return Types and Data Structures

### Transcription Result
```python
{
    'segments': [
        {
            'id': 0,
            'start': 0.0,
            'end': 2.5,
            'text': 'Transcribed text segment 1',
            'words': [{'word': 'word', 'start': 0.1, 'end': 0.3, 'probability': 0.99}, ...],
            'temperature': 0.0,
            'avg_logprob': -0.1,
            'compression_ratio': 1.2,
            'no_speech_prob': 0.1
        },
        # More segments...
    ],
    'language': 'en'  # Detected or specified language
}
```

### Translation Segment
```python
{
    'start': 0.0,          # Start time in seconds
    'end': 2.5,            # End time in seconds
    'text': 'Source text',  # Original transcription
    'translation': 'Translated text'  # Added by translation process
}
```

### SRT Format
```
1
00:00:00,000 --> 00:00:02,500
Subtitle text line 1
Subtitle text line 2

2
00:00:02,700 --> 00:00:05,200
Next subtitle segment
```

## 8. Error Codes and Messages

| Error Code | Message | Resolution |
|------------|---------|------------|
| 1 | "faster-whisper not installed" | Install faster-whisper |
| 1 | "FFmpeg not found" | Install FFmpeg and add to PATH |
| 2 | "CUDA error: no kernel image..." | Install CUDA Toolkit |
| 3 | "cudnn_ops64_9.dll not found" | Install cuDNN libraries |
| 4 | "GPU acceleration failed" | Check CUDA setup or use CPU |
| 5 | "Invalid model name" | Use supported model size |