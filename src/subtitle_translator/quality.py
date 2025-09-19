"""
Subtitle quality shaping module for the subtitle translator package.
"""
from typing import List, Dict, Union, Optional, Any
from pathlib import Path
import textwrap

from rich.console import Console

from .asr import Segment
from .srt_io import write_srt as write_srt_file

console = Console()

class SubtitleQualityShaper:
    """Applies quality constraints to subtitles for improved readability"""
    
    def __init__(
        self,
        max_line_chars: int = 42,
        max_lines: int = 2,
        max_cps: float = 20.0,
        min_duration: float = 0.5,
        min_gap: float = 0.2,
    ):
        """Initialize the subtitle quality shaper
        
        Args:
            max_line_chars: Maximum characters per line
            max_lines: Maximum number of lines per subtitle
            max_cps: Maximum characters per second
            min_duration: Minimum subtitle duration in seconds
            min_gap: Minimum gap between subtitles in seconds
        """
        self.max_line_chars = max_line_chars
        self.max_lines = max_lines
        self.max_cps = max_cps
        self.min_duration = min_duration
        self.min_gap = min_gap
    
    def shape_segments(self, segments: List[Segment]) -> List[Segment]:
        """Apply quality constraints to subtitle segments
        
        Args:
            segments: List of subtitle segments
            
        Returns:
            List of shaped segments
        """
        if not segments:
            return []
        
        result = []
        prev_end = 0.0
        
        for i, segment in enumerate(segments):
            # Apply minimum gap from previous segment
            start = segment.start
            if start - prev_end < self.min_gap:
                start = prev_end + self.min_gap
            
            # Get original duration
            original_duration = segment.end - segment.start
            
            # Apply minimum duration and adjust end time
            end = max(start + self.min_duration, segment.end)
            
            # Calculate allowed text length based on duration and CPS
            duration = end - start
            max_chars = int(self.max_cps * duration)
            
            # Apply shape_text to the segment content
            text = self._shape_text(
                segment.text,
                max_chars,
                self.max_line_chars,
                self.max_lines
            )
            
            # Create new segment with shaped text
            result.append(Segment(
                id=segment.id,
                start=start,
                end=end,
                text=text,
                source_text=segment.source_text
            ))
            
            prev_end = end
        
        # Renumber IDs to ensure sequential order
        for i, segment in enumerate(result):
            segment.id = i + 1
        
        return result
    
    def _shape_text(
        self,
        text: str,
        max_chars: int,
        max_line_chars: int,
        max_lines: int
    ) -> str:
        """Shape text according to subtitle constraints
        
        Args:
            text: Input text
            max_chars: Maximum total characters allowed
            max_line_chars: Maximum characters per line
            max_lines: Maximum number of lines
            
        Returns:
            Shaped text
        """
        # Truncate text if it exceeds max_chars
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].strip() + "..."
        
        # Split into lines with max_line_chars limit
        wrapped = textwrap.wrap(text, width=max_line_chars)
        
        # Limit number of lines
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            # Add ellipsis to last line if truncated
            if len(text) > sum(len(line) for line in wrapped):
                last = wrapped[-1]
                if len(last) > 3:  # Ensure we have space for ellipsis
                    wrapped[-1] = last[:-3] + "..."
        
        return "\n".join(wrapped)
    
    def write_srt(self, segments: List[Segment], file_path: Union[str, Path]) -> None:
        """Write segments to SRT file
        
        Args:
            segments: List of shaped segments
            file_path: Output SRT file path
        """
        write_srt_file(segments, file_path)
    
    def get_stats(self, segments: List[Segment]) -> Dict[str, Any]:
        """Get statistics about the subtitle segments
        
        Args:
            segments: List of segments
            
        Returns:
            Dictionary with statistics
        """
        if not segments:
            return {
                "segment_count": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "avg_cps": 0,
                "max_cps": 0,
                "avg_chars": 0,
                "max_chars": 0,
                "avg_lines": 0,
                "max_lines": 0,
            }
        
        total_duration = 0
        total_chars = 0
        max_cps_value = 0
        max_chars = 0
        total_lines = 0
        max_lines = 0
        
        for segment in segments:
            duration = segment.end - segment.start
            chars = len(segment.text)
            lines = segment.text.count('\n') + 1
            cps = chars / duration if duration > 0 else 0
            
            total_duration += duration
            total_chars += chars
            total_lines += lines
            
            max_cps_value = max(max_cps_value, cps)
            max_chars = max(max_chars, chars)
            max_lines = max(max_lines, lines)
        
        segment_count = len(segments)
        
        return {
            "segment_count": segment_count,
            "total_duration": total_duration,
            "avg_duration": total_duration / segment_count,
            "avg_cps": total_chars / total_duration if total_duration > 0 else 0,
            "max_cps": max_cps_value,
            "avg_chars": total_chars / segment_count,
            "max_chars": max_chars,
            "avg_lines": total_lines / segment_count,
            "max_lines": max_lines,
        }