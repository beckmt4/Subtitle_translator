"""
Video remuxing module for the subtitle translator package.
"""
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import subprocess
import json
import tempfile
import os

from rich.console import Console

console = Console()

def remux_subtitles(
    video_path: Union[str, Path],
    subtitle_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    subtitle_lang: str = "eng",
    subtitle_title: Optional[str] = None,
    copy_all_streams: bool = True,
) -> Path:
    """Remux subtitles into a video file without re-encoding
    
    Args:
        video_path: Path to the input video file
        subtitle_path: Path to the subtitle file (SRT)
        output_path: Path to the output video file (default: auto-generate)
        subtitle_lang: Language code for the subtitle track
        subtitle_title: Title for the subtitle track
        copy_all_streams: Copy all streams from input video
        
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
    
    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(subtitle_path)]
    
    # Add stream mapping
    if copy_all_streams:
        # Copy all streams from the input video
        cmd.extend(["-map", "0"])
    else:
        # Copy just video and audio
        cmd.extend(["-map", "0:v", "-map", "0:a?"])
    
    # Add subtitle mapping
    cmd.extend(["-map", "1"])
    
    # Set all streams to copy (no re-encoding)
    cmd.extend(["-c", "copy"])
    
    # Set subtitle metadata
    cmd.extend([
        "-metadata:s:s:0", f"language={subtitle_lang}",
    ])
    
    # Add subtitle title if provided
    if subtitle_title:
        cmd.extend(["-metadata:s:s:0", f"title={subtitle_title}"])
    
    # Add output path
    cmd.append(str(output_path))
    
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

def get_media_info(media_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a media file using FFprobe
    
    Args:
        media_path: Path to the media file
        
    Returns:
        Dictionary with media information
    """
    media_path = Path(media_path)
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(media_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFprobe failed: {e}[/red]")
        console.print(f"[dim]{e.stderr}[/dim]")
        raise RuntimeError(f"FFprobe failed: {e}")
    except json.JSONDecodeError:
        console.print(f"[red]Failed to parse FFprobe output as JSON[/red]")
        raise RuntimeError("Failed to parse FFprobe output")

def extract_subtitles(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    stream_index: Optional[int] = None,
) -> Path:
    """Extract subtitles from a video file
    
    Args:
        video_path: Path to the input video file
        output_path: Path to the output subtitle file (default: auto-generate)
        stream_index: Index of the subtitle stream to extract (default: first)
        
    Returns:
        Path to the extracted subtitle file
    """
    video_path = Path(video_path)
    
    # Get media info to find subtitle streams
    media_info = get_media_info(video_path)
    subtitle_streams = [
        (i, s) for i, s in enumerate(media_info.get("streams", []))
        if s.get("codec_type") == "subtitle"
    ]
    
    if not subtitle_streams:
        raise ValueError(f"No subtitle streams found in {video_path}")
    
    # Determine which subtitle stream to extract
    if stream_index is not None:
        # Find the stream with the specified index
        target_stream = next(
            (i, s) for i, s in subtitle_streams 
            if s.get("index") == stream_index
        )
    else:
        # Use the first subtitle stream
        target_stream = subtitle_streams[0]
    
    # Get target stream info
    stream_index, stream_info = target_stream
    stream_codec = stream_info.get("codec_name")
    
    # Determine output path if not specified
    if output_path is None:
        # Use language code in filename if available
        lang = stream_info.get("tags", {}).get("language", "unknown")
        output_path = video_path.with_stem(f"{video_path.stem}.{lang}").with_suffix(".srt")
    else:
        output_path = Path(output_path)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
    ]
    
    # Always convert to SRT format
    cmd.extend(["-c:s", "srt"])
    
    # Add output path
    cmd.append(str(output_path))
    
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
        console.print(f"[red]Subtitle extraction failed: {e}[/red]")
        console.print(f"[dim]{e.stderr}[/dim]")
        raise RuntimeError(f"Subtitle extraction failed: {e}")

def list_subtitle_streams(video_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """List subtitle streams in a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        List of subtitle stream information
    """
    media_info = get_media_info(video_path)
    
    subtitle_streams = []
    for stream in media_info.get("streams", []):
        if stream.get("codec_type") == "subtitle":
            subtitle_streams.append({
                "index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "language": stream.get("tags", {}).get("language", "unknown"),
                "title": stream.get("tags", {}).get("title", ""),
                "default": stream.get("disposition", {}).get("default", 0) == 1,
                "forced": stream.get("disposition", {}).get("forced", 0) == 1,
            })
    
    return subtitle_streams