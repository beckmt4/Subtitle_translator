#!/usr/bin/env python3
"""
Whisper MVP - GPU-accelerated subtitle generation from video files.

A simple command-line tool that uses faster-whisper with CUDA acceleration
to generate subtitles from video files, with support for transcription and
translation to English.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

import ffmpeg
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

console = Console()

# Supported video/audio formats
SUPPORTED_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', 
                    '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

# Available Whisper models
AVAILABLE_MODELS = {'tiny', 'base', 'small', 'medium', 'large-v3'}

# Compute types
COMPUTE_TYPES = {'float16', 'int8_float16', 'int8'}


class WhisperMVP:
    """Main application class for the Whisper MVP."""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None
        self.compute_type = None
    
    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            ffmpeg.probe('dummy', v='quiet')
        except ffmpeg.Error:
            # This is expected for a dummy file
            return True
        except FileNotFoundError:
            return False
        return True
    
    def load_model(self, model_name: str, device: str = "cuda", 
                   compute_type: str = "float16") -> bool:
        """Load the Whisper model with specified parameters."""
        if (self.model is not None and self.model_name == model_name 
            and self.device == device and self.compute_type == compute_type):
            return True
        
        try:
            console.print(f"[blue]Loading Whisper model: {model_name}[/blue]")
            console.print(f"[dim]Device: {device}, Compute type: {compute_type}[/dim]")
            
            self.model = WhisperModel(
                model_name, 
                device=device, 
                compute_type=compute_type,
                download_root=None  # Use default cache location
            )
            
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type
            
            console.print("[green]✓ Model loaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load model: {e}[/red]")
            return False
    
    def extract_audio(self, input_path: str) -> Optional[str]:
        """Extract audio from video file as 16kHz mono WAV."""
        try:
            # Create temporary file for extracted audio
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            console.print(f"[blue]Extracting audio from {Path(input_path).name}...[/blue]")
            
            # Extract audio using ffmpeg-python
            (
                ffmpeg
                .input(input_path)
                .output(
                    temp_audio_path,
                    acodec='pcm_s16le',  # 16-bit PCM
                    ac=1,                # Mono
                    ar='16000'           # 16kHz sample rate
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            console.print("[green]✓ Audio extracted successfully[/green]")
            return temp_audio_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            console.print(f"[red]✗ FFmpeg error: {error_msg}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]✗ Failed to extract audio: {e}[/red]")
            return None
    
    def format_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None,
                        translate: bool = False, beam_size: int = 5) -> List[dict]:
        """Transcribe audio file and return segments."""
        if self.model is None:
            console.print("[red]✗ Model not loaded[/red]")
            return []
        
        try:
            console.print("[blue]Starting transcription...[/blue]")
            
            # Set up transcription parameters
            transcribe_params = {
                'beam_size': beam_size,
                'language': language,
                'task': 'translate' if translate else 'transcribe'
            }
            
            # Remove None values
            transcribe_params = {k: v for k, v in transcribe_params.items() if v is not None}
            
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
    
    def generate_srt(self, segments: List[dict]) -> str:
        """Generate SRT format subtitles from segments."""
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
        
        # Validate input file
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
        
        # Extract audio
        temp_audio_path = self.extract_audio(str(input_path))
        if temp_audio_path is None:
            return False
        
        try:
            # Transcribe audio
            segments = self.transcribe_audio(
                temp_audio_path, language, translate, beam_size
            )
            
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
  python whisper_mvp.py video.mp4
  
  # Japanese to English translation
  python whisper_mvp.py anime.mkv --translate --lang ja
  
  # Custom model and output
  python whisper_mvp.py video.mp4 --model large-v3 --output subtitles.srt
  
  # Batch process folder
  python whisper_mvp.py folder/ --model medium --translate
        """
    )
    
    parser.add_argument('input', help='Input video file or folder')
    parser.add_argument('--output', '-o', help='Output SRT file path')
    parser.add_argument('--model', '-m', choices=AVAILABLE_MODELS, default='medium',
                       help='Whisper model to use (default: medium)')
    parser.add_argument('--lang', help='Source language code (e.g., ja, fr, de)')
    parser.add_argument('--translate', action='store_true',
                       help='Translate to English instead of transcribe')
    parser.add_argument('--beam', type=int, default=5,
                       help='Beam size for decoding (default: 5)')
    parser.add_argument('--compute', choices=COMPUTE_TYPES, default='float16',
                       help='Compute type for inference (default: float16)')
    parser.add_argument('--device', default='cuda',
                       help='Device to use for inference (default: cuda)')
    
    args = parser.parse_args()
    
    # Initialize the application
    app = WhisperMVP()
    
    # Check dependencies
    if not app.check_ffmpeg():
        console.print("[red]✗ FFmpeg not found! Please install FFmpeg and ensure it's in your PATH.[/red]")
        console.print("[yellow]Download from: https://ffmpeg.org/download.html[/yellow]")
        sys.exit(1)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file processing
        success = app.process_file(
            str(input_path),
            args.output,
            args.model,
            args.lang,
            args.translate,
            args.beam,
            args.device,
            args.compute
        )
        sys.exit(0 if success else 1)
    
    elif input_path.is_dir():
        # Batch processing
        video_files = []
        for ext in SUPPORTED_FORMATS:
            video_files.extend(input_path.glob(f"*{ext}"))
            video_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not video_files:
            console.print(f"[red]✗ No supported video files found in: {input_path}[/red]")
            sys.exit(1)
        
        console.print(f"[blue]Found {len(video_files)} video files to process[/blue]")
        
        success_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(video_files))
            
            for video_file in video_files:
                progress.update(task, description=f"Processing {video_file.name}")
                
                # Determine output path for batch processing
                output_file = video_file.with_suffix('.srt') if args.output is None else None
                
                success = app.process_file(
                    str(video_file),
                    str(output_file) if output_file else args.output,
                    args.model,
                    args.lang,
                    args.translate,
                    args.beam,
                    args.device,
                    args.compute
                )
                
                if success:
                    success_count += 1
                
                progress.advance(task)
        
        console.print(f"[green]✓ Batch processing completed: {success_count}/{len(video_files)} files successful[/green]")
        sys.exit(0 if success_count == len(video_files) else 1)
    
    else:
        console.print(f"[red]✗ Input not found: {input_path}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)