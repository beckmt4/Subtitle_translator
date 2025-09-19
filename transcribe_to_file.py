#!/usr/bin/env python
"""
Simple script to transcribe an audio/video file and save the result to a specified output file.
Uses faster-whisper for transcription.
"""
import sys
import os
from pathlib import Path
import subprocess
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Import faster-whisper if available
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed. Install with: pip install faster-whisper")
    sys.exit(1)

# Create console for rich output
console = Console()

def extract_audio(input_path, target_sr=16000):
    """Extract audio from video file using ffmpeg"""
    input_path = Path(input_path)
    wav_path = input_path.with_suffix(".asr.wav")
    
    console.print(f"Extracting audio to: {wav_path}")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "wav",
        str(wav_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return wav_path
    except subprocess.CalledProcessError as e:
        console.print(f"[red]FFmpeg extraction failed: {e}[/red]")
        sys.exit(1)

def format_timestamp(seconds):
    """Format seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    ms = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{ms:03d}"

def transcribe_to_file(input_path, output_path, language="auto", model_name="medium", device="cuda"):
    """Transcribe audio to text and write to SRT file"""
    console.print(f"[bold blue]Transcribing: {input_path}[/bold blue]")
    console.print(f"Output will be written to: {output_path}")
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Extract audio
        t_extract = progress.add_task("Extracting audio...", total=100)
        audio_path = extract_audio(input_path)
        progress.update(t_extract, completed=100)
        
        # Load model
        t_model = progress.add_task("Loading ASR model...", total=None)
        model = WhisperModel(model_name, device=device, compute_type="float16")
        progress.update(t_model, completed=100)
        
        # Transcribe
        t_transcribe = progress.add_task("Transcribing...", total=None)
        result = model.transcribe(
            str(audio_path), 
            language=None if language == "auto" else language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        progress.update(t_transcribe, completed=100)
        
        # Write SRT
        t_write = progress.add_task("Writing SRT...", total=100)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result[0]):
                # Write SRT entry (index, timestamp, text)
                f.write(f"{i+1}\n")
                f.write(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n")
                f.write(f"{segment.text.strip()}\n\n")
        progress.update(t_write, completed=100)
    
    console.print(f"[green]✓ Transcription complete: {output_path}[/green]")
    return output_path

if __name__ == "__main__":
    # Simple argument parsing
    if len(sys.argv) < 2:
        console.print("[red]Error: Input file required[/red]")
        console.print("Usage: python transcribe_to_file.py INPUT_FILE [OUTPUT_FILE] [LANGUAGE]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Use default output path if not specified
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        output_file = sys.argv[2]
    else:
        output_file = str(Path(input_file).with_suffix(".srt"))
    
    # Use default language if not specified
    language = "auto"
    for i, arg in enumerate(sys.argv):
        if arg == "--language" and i+1 < len(sys.argv):
            language = sys.argv[i+1]
    
    # Transcribe and save
    transcribe_to_file(input_file, output_file, language=language)