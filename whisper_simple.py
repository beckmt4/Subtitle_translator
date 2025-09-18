#!/usr/bin/env python3
"""
Whisper MVP - Alternative Version using OpenAI Whisper
Works around PyAV issues on Windows by using openai-whisper directly.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

import ffmpeg
import whisper
import torch
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
AVAILABLE_MODELS = {'tiny', 'base', 'small', 'medium', 'large'}


class WhisperMVPSimple:
    """Simplified Whisper MVP using OpenAI Whisper."""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None
    
    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            ffmpeg.probe('dummy', v='quiet')
        except ffmpeg.Error:
            return True
        except FileNotFoundError:
            return False
        return True
    
    def load_model(self, model_name: str, device: str = "cuda") -> bool:
        """Load the Whisper model."""
        if self.model is not None and self.model_name == model_name and self.device == device:
            return True
        
        try:
            console.print(f"[blue]Loading Whisper model: {model_name}[/blue]")
            
            # Check CUDA availability
            if device == "cuda" and not torch.cuda.is_available():
                console.print("[yellow]CUDA not available, using CPU[/yellow]")
                device = "cpu"
            
            console.print(f"[dim]Device: {device}[/dim]")
            
            self.model = whisper.load_model(model_name, device=device)
            self.model_name = model_name
            self.device = device
            
            console.print("[green]✓ Model loaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load model: {e}[/red]")
            return False
    
    def extract_audio(self, input_path: str) -> Optional[str]:
        """Extract audio from video file as 16kHz mono WAV."""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            console.print(f"[blue]Extracting audio from {Path(input_path).name}...[/blue]")
            
            (
                ffmpeg
                .input(input_path)
                .output(
                    temp_audio_path,
                    acodec='pcm_s16le',
                    ac=1,
                    ar='16000'
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
        """Format timestamp for SRT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None,
                        translate: bool = False) -> List[dict]:
        """Transcribe audio file and return segments."""
        if self.model is None:
            console.print("[red]✗ Model not loaded[/red]")
            return []
        
        try:
            console.print("[blue]Starting transcription...[/blue]")
            
            transcribe_params = {}
            if language:
                transcribe_params['language'] = language
            if translate:
                transcribe_params['task'] = 'translate'
            
            result = self.model.transcribe(audio_path, **transcribe_params)
            
            segment_list = []
            for segment in result['segments']:
                segment_list.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
            
            console.print(f"[green]✓ Transcription completed ({len(segment_list)} segments)[/green]")
            if 'language' in result:
                console.print(f"[dim]Detected language: {result['language']}[/dim]")
            
            return segment_list
            
        except Exception as e:
            console.print(f"[red]✗ Transcription failed: {e}[/red]")
            return []
    
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
            srt_content.append("")
        
        return '\n'.join(srt_content)
    
    def process_file(self, input_path: str, output_path: Optional[str] = None,
                    model_name: str = "medium", language: Optional[str] = None,
                    translate: bool = False, device: str = "cuda") -> bool:
        """Process a single video/audio file."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            console.print(f"[red]✗ Input file not found: {input_path}[/red]")
            return False
        
        if input_path.suffix.lower() not in SUPPORTED_FORMATS:
            console.print(f"[red]✗ Unsupported file format: {input_path.suffix}[/red]")
            return False
        
        if output_path is None:
            output_path = input_path.with_suffix('.srt')
        else:
            output_path = Path(output_path)
        
        console.print(Panel(f"[bold]Processing: {input_path.name}[/bold]"))
        
        if not self.load_model(model_name, device):
            return False
        
        temp_audio_path = self.extract_audio(str(input_path))
        if temp_audio_path is None:
            return False
        
        try:
            segments = self.transcribe_audio(temp_audio_path, language, translate)
            
            if not segments:
                console.print("[red]✗ No segments generated[/red]")
                return False
            
            srt_content = self.generate_srt(segments)
            output_path.write_text(srt_content, encoding='utf-8')
            
            console.print(f"[green]✓ Subtitles saved to: {output_path}[/green]")
            
            # Display summary
            total_duration = segments[-1]['end'] if segments else 0
            table = Table(title="Processing Summary")
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="cyan")
            table.add_row("Input File", input_path.name)
            table.add_row("Output File", output_path.name)
            table.add_row("Total Duration", f"{total_duration:.2f} seconds")
            table.add_row("Number of Segments", str(len(segments)))
            console.print(table)
            
            return True
            
        finally:
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate subtitles from video files using Whisper (OpenAI version)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('--output', '-o', help='Output SRT file path')
    parser.add_argument('--model', '-m', choices=AVAILABLE_MODELS, default='medium',
                       help='Whisper model to use (default: medium)')
    parser.add_argument('--lang', help='Source language code (e.g., ja, fr, de)')
    parser.add_argument('--translate', action='store_true',
                       help='Translate to English instead of transcribe')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for inference (default: cuda)')
    
    args = parser.parse_args()
    
    app = WhisperMVPSimple()
    
    if not app.check_ffmpeg():
        console.print("[red]✗ FFmpeg not found![/red]")
        sys.exit(1)
    
    success = app.process_file(
        args.input,
        args.output,
        args.model,
        args.lang,
        args.translate,
        args.device
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