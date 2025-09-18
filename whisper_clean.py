#!/usr/bin/env python3
"""
Whisper MVP - Clean Implementation
Uses FFmpeg for audio extraction and faster-whisper for transcription.
No PyAV dependencies - solves Windows compatibility issues.
"""

import argparse
import os
import sys
import subprocess
import shlex
import tempfile
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

# Supported video/audio formats
SUPPORTED_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', 
                    '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

# Available Whisper models
AVAILABLE_MODELS = {'tiny', 'base', 'small', 'medium', 'large-v3'}

class WhisperMVPClean:
    """Clean Whisper MVP implementation without PyAV."""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None
        self.compute_type = None
    
    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def load_model(self, model_name: str, device: str = "cuda", 
                   compute_type: str = "float16") -> bool:
        """Load the Whisper model."""
        if (self.model is not None and self.model_name == model_name 
            and self.device == device and self.compute_type == compute_type):
            return True
        
        try:
            # Import faster-whisper here to avoid import issues
            from faster_whisper import WhisperModel
            
            console.print(f"[blue]Loading Whisper model: {model_name}[/blue]")
            console.print(f"[dim]Device: {device}, Compute type: {compute_type}[/dim]")
            
            self.model = WhisperModel(
                model_name, 
                device=device, 
                compute_type=compute_type
            )
            
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type
            
            console.print("[green]✓ Model loaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load model: {e}[/red]")
            if "CUDA" in str(e) or "cuda" in str(e):
                console.print("[yellow]Tip: Try --device cpu if CUDA is not available[/yellow]")
            return False
    
    def extract_audio_ffmpeg(self, input_path: str) -> Optional[str]:
        """Extract audio using FFmpeg directly - no PyAV needed."""
        try:
            # Create temporary WAV file
            temp_dir = tempfile.gettempdir()
            temp_audio = os.path.join(temp_dir, f"whisper_temp_{os.getpid()}.wav")
            
            console.print(f"[blue]Extracting audio from {Path(input_path).name}...[/blue]")
            
            # FFmpeg command: extract 16kHz mono WAV
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', input_path,  # Input file
                '-vn',  # No video
                '-ac', '1',  # Mono
                '-ar', '16000',  # 16kHz sample rate
                '-f', 'wav',  # WAV format
                temp_audio
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                console.print(f"[red]✗ FFmpeg failed: {result.stderr}[/red]")
                return None
            
            if not os.path.exists(temp_audio):
                console.print("[red]✗ Audio extraction failed - output file not created[/red]")
                return None
            
            console.print("[green]✓ Audio extracted successfully[/green]")
            return temp_audio
            
        except subprocess.TimeoutExpired:
            console.print("[red]✗ FFmpeg timeout - video file too large or corrupted[/red]")
            return None
        except Exception as e:
            console.print(f"[red]✗ Audio extraction failed: {e}[/red]")
            return None
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None,
                        translate: bool = False, beam_size: int = 5) -> List[dict]:
        """Transcribe audio file using faster-whisper."""
        if self.model is None:
            console.print("[red]✗ Model not loaded[/red]")
            return []
        
        try:
            console.print("[blue]Starting transcription...[/blue]")
            
            # Set up transcription parameters
            transcribe_params = {
                'beam_size': beam_size,
                'task': 'translate' if translate else 'transcribe'
            }
            
            if language:
                transcribe_params['language'] = language
            
            # Perform transcription
            segments, info = self.model.transcribe(audio_path, **transcribe_params)
            
            # Convert segments to list
            segment_list = []
            for segment in segments:
                segment_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })
            
            console.print(f"[green]✓ Transcription completed ({len(segment_list)} segments)[/green]")
            console.print(f"[dim]Detected language: {info.language} (confidence: {info.language_probability:.2f})[/dim]")
            
            return segment_list
            
        except Exception as e:
            console.print(f"[red]✗ Transcription failed: {e}[/red]")
            return []
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def generate_srt(self, segments: List[dict]) -> str:
        """Generate SRT format subtitles."""
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            start_time = self.format_timestamp(segment['start'])
            end_time = self.format_timestamp(segment['end'])
            text = segment['text']
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")  # Empty line between segments
        
        return '\n'.join(srt_content)
    
    def process_file(self, input_path: str, output_path: Optional[str] = None,
                    model_name: str = "medium", language: Optional[str] = None,
                    translate: bool = False, beam_size: int = 5,
                    device: str = "cuda", compute_type: str = "float16") -> bool:
        """Process a single video/audio file."""
        input_path = Path(input_path)
        
        # Validate input
        if not input_path.exists():
            console.print(f"[red]✗ Input file not found: {input_path}[/red]")
            return False
        
        if input_path.suffix.lower() not in SUPPORTED_FORMATS:
            console.print(f"[red]✗ Unsupported file format: {input_path.suffix}[/red]")
            return False
        
        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix('.srt')
        else:
            output_path = Path(output_path)
        
        console.print(Panel(f"[bold]Processing: {input_path.name}[/bold]"))
        
        # Load model
        if not self.load_model(model_name, device, compute_type):
            return False
        
        # Extract audio with FFmpeg
        temp_audio_path = self.extract_audio_ffmpeg(str(input_path))
        if temp_audio_path is None:
            return False
        
        try:
            # Transcribe audio
            segments = self.transcribe_audio(temp_audio_path, language, translate, beam_size)
            
            if not segments:
                console.print("[red]✗ No segments generated[/red]")
                return False
            
            # Generate SRT content
            srt_content = self.generate_srt(segments)
            
            # Write SRT file
            output_path.write_text(srt_content, encoding='utf-8')
            
            console.print(f"[green]✓ Subtitles saved to: {output_path}[/green]")
            
            # Display summary
            self.display_summary(segments, input_path.name, str(output_path))
            
            return True
            
        finally:
            # Clean up temporary audio file
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except Exception:
                pass
    
    def display_summary(self, segments: List[dict], input_name: str, output_path: str):
        """Display processing summary."""
        if not segments:
            return
        
        total_duration = segments[-1]['end'] if segments else 0
        
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")
        
        table.add_row("Input File", input_name)
        table.add_row("Output File", Path(output_path).name)
        table.add_row("Total Duration", f"{total_duration:.2f} seconds")
        table.add_row("Number of Segments", str(len(segments)))
        table.add_row("Average Segment Length", f"{total_duration / len(segments):.2f} seconds")
        
        console.print(table)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate subtitles from video files using GPU-accelerated Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python whisper_clean.py video.mp4
  
  # Japanese to English translation  
  python whisper_clean.py anime.mkv --translate --lang ja
  
  # High quality with large model
  python whisper_clean.py video.mp4 --model large-v3 --beam 8
  
  # CPU mode (if CUDA issues)
  python whisper_clean.py video.mp4 --device cpu
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('--output', '-o', help='Output SRT file path')
    parser.add_argument('--model', '-m', choices=AVAILABLE_MODELS, default='medium',
                       help='Whisper model to use (default: medium)')
    parser.add_argument('--lang', help='Source language code (e.g., ja, fr, de)')
    parser.add_argument('--translate', action='store_true',
                       help='Translate to English instead of transcribe')
    parser.add_argument('--beam', type=int, default=5,
                       help='Beam size for decoding (default: 5)')
    parser.add_argument('--compute', choices=['float16', 'int8_float16', 'int8'], 
                       default='float16', help='Compute type (default: float16)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Initialize the application
    app = WhisperMVPClean()
    
    # Check dependencies
    if not app.check_ffmpeg():
        console.print("[red]✗ FFmpeg not found! Please install FFmpeg.[/red]")
        console.print("[yellow]Windows: Use the PowerShell script in README.md[/yellow]")
        sys.exit(1)
    
    # Process the file
    success = app.process_file(
        args.input,
        args.output,
        args.model,
        args.lang,
        args.translate,
        args.beam,
        args.device,
        args.compute
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)