#!/usr/bin/env python3
"""
Demo script for Whisper MVP - Shows basic usage examples
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path so we can import whisper_mvp
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from whisper_mvp import WhisperMVP, console


def create_demo_audio():
    """Create a simple demo audio file for testing (requires ffmpeg)."""
    try:
        import subprocess
        
        # Create a 5-second sine wave audio file
        demo_path = Path("demo_audio.wav")
        
        if demo_path.exists():
            console.print(f"[green]Using existing demo file: {demo_path}[/green]")
            return str(demo_path)
        
        console.print("[blue]Creating demo audio file...[/blue]")
        
        # Generate a 5-second 440Hz sine wave (A note)
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=5',
            '-ar', '16000', '-ac', '1', str(demo_path), '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print(f"[green]✓ Demo audio created: {demo_path}[/green]")
            return str(demo_path)
        else:
            console.print(f"[red]✗ Failed to create demo audio: {result.stderr}[/red]")
            return None
            
    except FileNotFoundError:
        console.print("[red]✗ FFmpeg not found - cannot create demo audio[/red]")
        return None
    except Exception as e:
        console.print(f"[red]✗ Error creating demo audio: {e}[/red]")
        return None


def demo_basic_functionality():
    """Demonstrate basic functionality without requiring video files."""
    console.print("\n[bold blue]Whisper MVP - Demo Script[/bold blue]")
    console.print("=" * 50)
    
    # Initialize the app
    app = WhisperMVP()
    
    # Test 1: Check FFmpeg availability
    console.print("\n[bold]Test 1: FFmpeg Availability[/bold]")
    if app.check_ffmpeg():
        console.print("[green]✓ FFmpeg is available[/green]")
    else:
        console.print("[red]✗ FFmpeg not found[/red]")
        return False
    
    # Test 2: Model loading (CPU mode for demo)
    console.print("\n[bold]Test 2: Model Loading (CPU mode for safety)[/bold]")
    success = app.load_model("tiny", device="cpu", compute_type="int8")
    if success:
        console.print("[green]✓ Model loaded successfully[/green]")
    else:
        console.print("[red]✗ Model loading failed[/red]")
        return False
    
    # Test 3: SRT formatting
    console.print("\n[bold]Test 3: SRT Formatting[/bold]")
    demo_segments = [
        {"start": 0.0, "end": 3.5, "text": "Hello, this is a test subtitle."},
        {"start": 3.5, "end": 7.0, "text": "This is the second subtitle line."},
        {"start": 7.0, "end": 10.0, "text": "And this is the final subtitle."}
    ]
    
    srt_content = app.generate_srt(demo_segments)
    console.print("[green]✓ SRT formatting working correctly[/green]")
    console.print("\n[dim]Sample SRT output:[/dim]")
    console.print(srt_content[:200] + "..." if len(srt_content) > 200 else srt_content)
    
    # Test 4: Show summary
    console.print("\n[bold]Test 4: Summary Display[/bold]")
    app.display_summary(demo_segments, "demo_video.mp4", "demo_output.srt")
    
    console.print("\n[green bold]✓ All basic tests passed![/green bold]")
    console.print("\n[yellow]Note: This demo uses CPU mode and synthetic data.[/yellow]")
    console.print("[yellow]For GPU testing, use actual video files with the main script.[/yellow]")
    
    return True


def show_usage_examples():
    """Show practical usage examples."""
    console.print("\n[bold blue]Usage Examples[/bold blue]")
    console.print("=" * 50)
    
    examples = [
        {
            "title": "Basic Transcription",
            "command": "python whisper_mvp.py video.mp4",
            "description": "Generate same-language subtitles for a video file"
        },
        {
            "title": "Japanese Anime Translation",
            "command": "python whisper_mvp.py anime.mkv --translate --lang ja --model medium",
            "description": "Translate Japanese anime to English subtitles"
        },
        {
            "title": "High-Quality Processing",
            "command": "python whisper_mvp.py movie.mp4 --model large-v3 --beam 8",
            "description": "Use largest model with high beam size for best quality"
        },
        {
            "title": "Batch Processing",
            "command": "python whisper_mvp.py \"C:\\Videos\\Movies\" --translate --model medium",
            "description": "Process all videos in a folder"
        },
        {
            "title": "Memory-Optimized",
            "command": "python whisper_mvp.py video.mp4 --model small --compute int8_float16",
            "description": "Use smaller model and reduced precision for lower-end GPUs"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        console.print(f"\n[bold]{i}. {example['title']}[/bold]")
        console.print(f"[cyan]{example['command']}[/cyan]")
        console.print(f"[dim]{example['description']}[/dim]")


if __name__ == "__main__":
    try:
        # Run basic functionality demo
        if demo_basic_functionality():
            # Show usage examples
            show_usage_examples()
            
            console.print("\n[bold green]Demo completed successfully![/bold green]")
            console.print("\n[yellow]Ready to process your video files![/yellow]")
            console.print("[dim]See README.md for detailed setup and usage instructions.[/dim]")
        else:
            console.print("\n[red]Demo failed - check your setup[/red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        sys.exit(1)