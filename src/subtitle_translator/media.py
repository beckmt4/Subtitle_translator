"""
Media handling module for the subtitle translator package.
"""
from typing import Optional, Union
import subprocess
from pathlib import Path
import tempfile

from rich.console import Console

console = Console()

def extract_audio(
    input_path: Union[str, Path], 
    target_sr: int = 16000,
    temp_audio: bool = True,
) -> Path:
    """Extract audio from a video file
    
    Args:
        input_path: Path to the input media file
        target_sr: Target sample rate (default: 16000 for Whisper)
        temp_audio: If True, create a temporary file; if False, save alongside video
        
    Returns:
        Path to the extracted audio file
    """
    input_path = Path(input_path)
    
    # Determine output path
    if temp_audio:
        # Use a temp file with a .wav extension that won't be auto-deleted
        temp_dir = tempfile.gettempdir()
        output_path = Path(temp_dir) / f"{input_path.stem}_{target_sr}.wav"
    else:
        output_path = input_path.with_suffix(".asr.wav")
    
    # Run FFmpeg to extract audio
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-i", str(input_path),
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", str(target_sr),  # Sample rate
        "-f", "wav",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFmpeg audio extraction failed: {e}[/red]")
        console.print(f"[dim]{e.stderr}[/dim]")
        raise RuntimeError(f"Audio extraction failed: {e}")

def remux_subtitles(
    video_path: Union[str, Path],
    subtitle_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    subtitle_lang: str = "eng",
) -> Path:
    """Remux subtitles into a video file without re-encoding
    
    Args:
        video_path: Path to the input video file
        subtitle_path: Path to the subtitle file (SRT)
        output_path: Path to the output video file (default: auto-generate)
        subtitle_lang: Language code for the subtitle track
        
    Returns:
        Path to the output video file
    """
    video_path = Path(video_path)
    subtitle_path = Path(subtitle_path)
    
    # Determine output path if not specified
    if output_path is None:
        output_path = video_path.with_stem(f"{video_path.stem}.subbed").with_suffix(".mkv")
    else:
        output_path = Path(output_path)
    
    # Run FFmpeg to remux
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-i", str(video_path),
        "-i", str(subtitle_path),
        "-map", "0:v",  # Map all video streams
        "-map", "0:a",  # Map all audio streams
        "-map", "1",    # Map subtitle
        "-c", "copy",   # Stream copy (no re-encoding)
        "-metadata:s:s:0", f"language={subtitle_lang}",  # Set subtitle language
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFmpeg remux failed: {e}[/red]")
        console.print(f"[dim]{e.stderr}[/dim]")
        raise RuntimeError(f"Remux failed: {e}")

def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system
    
    Returns:
        True if FFmpeg is available, False otherwise
    """
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