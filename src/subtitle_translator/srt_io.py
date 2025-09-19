"""
SRT input/output module for the subtitle translator package.
"""
from typing import List, Dict, Union, Optional, TextIO
from pathlib import Path
import re

from rich.console import Console

from .asr import Segment

console = Console()

# Regex pattern for parsing SRT timestamps
TIMESTAMP_PATTERN = re.compile(r"(\d+):(\d+):(\d+)[.,](\d+)")

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT formatted timestamp (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

def parse_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp to seconds
    
    Args:
        timestamp: SRT timestamp (HH:MM:SS,mmm)
        
    Returns:
        Time in seconds
    """
    match = TIMESTAMP_PATTERN.search(timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

def read_srt(file_path: Union[str, Path]) -> List[Segment]:
    """Read segments from an SRT file
    
    Args:
        file_path: Path to the SRT file
        
    Returns:
        List of segments
    """
    file_path = Path(file_path)
    segments = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Split into individual subtitle entries (separated by double newline)
        entries = content.strip().split("\n\n")
        
        for entry in entries:
            lines = entry.strip().split("\n")
            if len(lines) < 3:
                continue  # Skip invalid entries
            
            try:
                index = int(lines[0])
                timestamps = lines[1].split(" --> ")
                start = parse_timestamp(timestamps[0])
                end = parse_timestamp(timestamps[1])
                text = "\n".join(lines[2:])
                
                segments.append(Segment(
                    id=index,
                    start=start,
                    end=end,
                    text=text
                ))
            except (ValueError, IndexError) as e:
                console.print(f"[yellow]Warning: Could not parse entry: {entry}[/yellow]")
                console.print(f"[dim]{e}[/dim]")
    
    except Exception as e:
        console.print(f"[red]Error reading SRT file: {e}[/red]")
        return []
    
    return segments

def write_srt(segments: List[Segment], file_path: Union[str, Path]) -> None:
    """Write segments to an SRT file
    
    Args:
        segments: List of segments to write
        file_path: Path to the output SRT file
    """
    file_path = Path(file_path)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for segment in segments:
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)
                
                f.write(f"{segment.id}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text}\n\n")
    
    except Exception as e:
        console.print(f"[red]Error writing SRT file: {e}[/red]")
        raise

def adjust_timings(
    segments: List[Segment],
    offset: float = 0.0,
    scale: float = 1.0,
    min_duration: float = 0.5,
    min_gap: float = 0.2,
) -> List[Segment]:
    """Adjust segment timings with offset, scale, and constraints
    
    Args:
        segments: List of segments to adjust
        offset: Time offset in seconds (positive or negative)
        scale: Time scale factor
        min_duration: Minimum segment duration
        min_gap: Minimum gap between segments
        
    Returns:
        List of adjusted segments
    """
    if not segments:
        return []
    
    result = []
    prev_end = None
    
    for i, segment in enumerate(segments):
        # Apply offset and scale
        start = segment.start * scale + offset
        end = segment.end * scale + offset
        
        # Ensure minimum duration
        if end - start < min_duration:
            end = start + min_duration
        
        # Ensure minimum gap from previous segment
        if prev_end is not None and start - prev_end < min_gap:
            start = prev_end + min_gap
            # Re-check minimum duration after gap enforcement
            if end - start < min_duration:
                end = start + min_duration
        
        # Create new segment with adjusted timing
        result.append(Segment(
            id=segment.id,
            start=start,
            end=end,
            text=segment.text,
            source_text=segment.source_text
        ))
        
        prev_end = end
    
    # Renumber IDs to ensure sequential order
    for i, segment in enumerate(result):
        segment.id = i + 1
    
    return result