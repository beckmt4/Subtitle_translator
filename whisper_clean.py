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
import wave
from pathlib import Path
from typing import List, Optional, cast

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
                   compute_type: str = "float16", quiet: bool = False) -> bool:
        """Load the Whisper model.

        quiet: suppress console output (used when progress bar active)
        """
        if (self.model is not None and self.model_name == model_name 
            and self.device == device and self.compute_type == compute_type):
            return True
        
        try:
            # Import faster-whisper here to avoid import issues
            from faster_whisper import WhisperModel
            
            if not quiet:
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
            
            if not quiet:
                console.print("[green]✓ Model loaded successfully[/green]")
            return True
            
        except Exception as e:
            if not quiet:
                console.print(f"[red]✗ Failed to load model: {e}[/red]")
                if "CUDA" in str(e) or "cuda" in str(e):
                    console.print("[yellow]Tip: Try --device cpu if CUDA is not available or install CUDA Toolkit + cuDNN[/yellow]")
            return False

    def _check_cuda_runtime(self) -> tuple[bool, list[str]]:
        """Check presence of critical CUDA/cuDNN runtime DLLs on Windows.

        Returns (ready, missing_list). Only meaningful if device == 'cuda'.
        """
        if os.name != 'nt' or self.device != 'cuda':
            return True, []
        required = [
            'cudart64_12.dll',
            'cublas64_12.dll',
            'cublasLt64_12.dll',
            'cudnn_ops64_9.dll'
        ]
        missing: list[str] = []
        for dll in required:
            try:
                r = subprocess.run(['where', dll], capture_output=True, text=True)
                if r.returncode != 0:
                    missing.append(dll)
            except Exception:
                # If where not available, skip detailed check
                return True, []  # don't block
        return (len(missing) == 0, missing)
    
    def extract_audio_ffmpeg(self, input_path: str, quiet: bool = False) -> Optional[str]:
        """Extract audio using FFmpeg directly - no PyAV needed.

        quiet: suppress console output
        """
        try:
            # Create temporary WAV file
            temp_dir = tempfile.gettempdir()
            temp_audio = os.path.join(temp_dir, f"whisper_temp_{os.getpid()}.wav")
            
            if not quiet:
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
                if not quiet:
                    console.print(f"[red]✗ FFmpeg failed: {result.stderr}[/red]")
                return None
            
            if not os.path.exists(temp_audio):
                if not quiet:
                    console.print("[red]✗ Audio extraction failed - output file not created[/red]")
                return None
            
            if not quiet:
                console.print("[green]✓ Audio extracted successfully[/green]")
            return temp_audio
            
        except subprocess.TimeoutExpired:
            if not quiet:
                console.print("[red]✗ FFmpeg timeout - video file too large or corrupted[/red]")
            return None
        except Exception as e:
            if not quiet:
                console.print(f"[red]✗ Audio extraction failed: {e}[/red]")
            return None
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None,
                        translate: bool = False, beam_size: int = 5,
                        quiet: bool = False,
                        progress: Optional[Progress] = None,
                        progress_task: Optional[int] = None,
                        total_duration: Optional[float] = None) -> List[dict]:
        """Transcribe audio file using faster-whisper.

        If progress & task provided, update progress using segment end time relative to total_duration.
        """
        if self.model is None:
            if not quiet:
                console.print("[red]✗ Model not loaded[/red]")
            return []
        
        try:
            if not quiet:
                console.print("[blue]Starting transcription...[/blue]")
            
            # Set up transcription parameters
            transcribe_params = {
                'beam_size': beam_size,
                'task': 'translate' if translate else 'transcribe'
            }
            
            if language:
                transcribe_params['language'] = language
            
            # Perform transcription
            seg_iter, info = self.model.transcribe(audio_path, **transcribe_params)
            segment_list: List[dict] = []
            for seg in seg_iter:
                segment_list.append({
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip()
                })
                if progress and progress_task is not None and total_duration:
                    completed = min(seg.end, total_duration)
                    progress.update(cast("int", progress_task), completed=completed)  # type: ignore[arg-type]
            if progress and progress_task is not None and total_duration:
                progress.update(cast("int", progress_task), completed=total_duration)  # type: ignore[arg-type]
            if not quiet:
                console.print(f"[green]✓ Transcription completed ({len(segment_list)} segments)[/green]")
                console.print(f"[dim]Detected language: {info.language} (confidence: {info.language_probability:.2f})[/dim]")
            return segment_list
            
        except Exception as e:
            if not quiet:
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
    
    def process_file(self, input_path: str | Path, output_path: Optional[str | Path] = None,
                    model_name: str = "medium", language: Optional[str] = None,
                    translate: bool = False, beam_size: int = 5,
                    device: str = "cuda", compute_type: str = "float16",
                    allow_fallback: bool = True,
                    device_order: Optional[list[str]] = None,
                    show_progress: bool = True,
                    remux: bool = False,
                    remux_language: str | None = None,
                    remux_overwrite: bool = False) -> bool:
        """Process a single video/audio file.

        Parameters:
            device: Primary requested device (legacy flag).
            device_order: Optional ordered list of devices to try. Recognized tokens:
                - 'cuda': NVIDIA GPU via CUDA
                - 'igpu': Integrated GPU placeholder (currently maps to optimized CPU path)
                - 'cpu': Standard CPU execution
        If device_order is provided it takes precedence over automatic internal fallback logic; we iterate devices.
        """
        if device_order:
            # Normalize + deduplicate while preserving order
            seen = set()
            norm_order = []
            for d in device_order:
                d_norm = d.strip().lower()
                if d_norm in ("gpu",):
                    d_norm = 'cuda'
                if d_norm not in {"cuda", "igpu", "cpu"}:
                    console.print(f"[yellow]Ignoring unknown device token: {d}[/yellow]")
                    continue
                if d_norm not in seen:
                    norm_order.append(d_norm)
                    seen.add(d_norm)
            if not norm_order:
                norm_order = [device]
            for idx, d_try in enumerate(norm_order):
                last = (idx == len(norm_order)-1)
                # For 'igpu' we currently map to CPU with int8 preference
                mapped_device = 'cpu' if d_try == 'igpu' else d_try
                # Choose compute type heuristics for mapped device
                ct = compute_type
                if d_try == 'igpu':
                    # Prefer int8 (smaller / faster on typical integrated GPUs via CPU fallback)
                    if ct == 'float16':
                        ct = 'int8'
                # Only allow internal fallback on last attempt; earlier attempts should not auto-fallback to mask next devices
                success = self.process_file(
                    input_path=input_path,
                    output_path=output_path,
                    model_name=model_name,
                    language=language,
                    translate=translate,
                    beam_size=beam_size,
                    device=mapped_device,
                    compute_type=ct,
                    allow_fallback=allow_fallback if last else False,
                    device_order=None,  # prevent recursion loops
                    show_progress=show_progress
                )
                if success:
                    if d_try == 'igpu':
                        console.print("[green]✓ Completed using 'igpu' (optimized CPU path).[/green]")
                    return True
                else:
                    if not last:
                        console.print(f"[yellow]Device '{d_try}' failed. Trying next: {norm_order[idx+1]}[/yellow]")
            return False
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
        
        # Adjust compute type for CPU if user asked for float16 (unsupported efficiently)
        orig_device = device
        orig_compute = compute_type
        if device == 'cpu' and compute_type == 'float16':
            # Choose a more appropriate compute type for CPU
            compute_type = 'int8_float16'
            console.print("[yellow]Float16 not efficient on CPU. Using int8_float16 instead.[/yellow]")

        # Setup progress UI
        progress: Optional[Progress] = None
        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
                transient=False
            )
            progress.start()
        t_load = t_extract = t_transcribe = t_srt = t_write = None
        if progress:
            t_load = progress.add_task("Load Model", total=1)

        # Load model (with potential fallback logic inside this scope)
        if not self.load_model(model_name, device, compute_type, quiet=bool(progress)):
            # If initial load failed on CUDA and fallback allowed, attempt CPU
            if device == 'cuda' and allow_fallback:
                console.print("[yellow]Attempting automatic fallback to CPU...[/yellow]")
                # Reset model reference
                self.model = None
                # Switch compute type if needed
                fb_compute = 'int8_float16' if orig_compute == 'float16' else orig_compute
                if not self.load_model(model_name, 'cpu', fb_compute, quiet=bool(progress)):
                    return False
                device = 'cpu'
                compute_type = fb_compute
            else:
                return False
        if progress and t_load is not None:
            progress.update(t_load, advance=1)
        
        # Extract audio with FFmpeg
        if progress:
            t_extract = progress.add_task("Extract Audio", total=1)
        temp_audio_path = self.extract_audio_ffmpeg(str(input_path), quiet=bool(progress))
        if progress and t_extract is not None:
            progress.update(t_extract, advance=1)
        if temp_audio_path is None:
            return False

        # Determine audio duration for transcription progress
        audio_duration: Optional[float] = None
        try:
            with wave.open(temp_audio_path, 'rb') as wf:  # type: ignore[arg-type]
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    audio_duration = frames / float(rate)
        except Exception:
            audio_duration = None
        
        # Proactive CUDA runtime verification (avoid crash inside transcribe)
        if self.device == 'cuda':
            ready, missing = self._check_cuda_runtime()
            if not ready:
                console.print("[yellow]Detected GPU but missing runtime DLLs: " + ', '.join(missing) + "[/yellow]")
                if allow_fallback:
                    console.print("[yellow]Falling back to CPU before transcription starts.[/yellow]")
                    # Try a sequence of CPU-friendly compute types
                    fallback_candidates = []
                    if orig_compute == 'float16':
                        fallback_candidates.extend(['int8_float16', 'int8', 'float32'])
                    else:
                        fallback_candidates.append(orig_compute)
                        if orig_compute != 'int8':
                            fallback_candidates.append('int8')
                        fallback_candidates.append('float32')
                    loaded = False
                    last_error = None
                    for fb_compute in fallback_candidates:
                        try:
                            self.model = None
                            if self.load_model(model_name, 'cpu', fb_compute):
                                device = 'cpu'
                                compute_type = fb_compute
                                loaded = True
                                console.print(f"[green]✓ CPU fallback using compute type: {fb_compute}[/green]")
                                break
                        except Exception as e:  # pragma: no cover
                            last_error = e
                            continue
                    if not loaded:
                        console.print(f"[red]✗ CPU fallback failed. Tried: {', '.join(fallback_candidates)}[/red]")
                        if last_error:
                            console.print(f"[red]{last_error}[/red]")
                        return False
                else:
                    console.print("[red]CUDA runtime incomplete and fallback disabled (--no-fallback). Aborting.[/red]")
                    return False

        try:
            # Transcribe audio
            if progress:
                t_transcribe = progress.add_task("Transcribe", total=audio_duration if audio_duration else 1)
            segments = self.transcribe_audio(
                temp_audio_path,
                language,
                translate,
                beam_size,
                quiet=bool(progress),
                progress=progress,
                progress_task=t_transcribe,
                total_duration=audio_duration
            )

            # If transcription failed due to CUDA/cuDNN error and fallback allowed, retry on CPU
            if not segments and orig_device == 'cuda' and self.device == 'cuda' and allow_fallback:
                # Heuristic: attempt fallback if common cuda/cudnn markers appear in recent stderr (not captured here) or segments empty after CUDA start
                console.print("[yellow]CUDA transcription failed. Retrying on CPU (fallback).[/yellow]")
                try:
                    self.model = None
                    fb_compute = 'int8_float16' if orig_compute == 'float16' else orig_compute
                    if not self.load_model(model_name, 'cpu', fb_compute):
                        return False
                    segments = self.transcribe_audio(temp_audio_path, language, translate, beam_size)
                    if not segments:
                        return False
                    device = 'cpu'
                except Exception as e:
                    console.print(f"[red]✗ CPU fallback failed: {e}[/red]")
                    return False
            
            if not segments:
                console.print("[red]✗ No segments generated[/red]")
                return False
            
            # Generate SRT content
            if progress:
                t_srt = progress.add_task("Generate SRT", total=1)
            srt_content = self.generate_srt(segments)
            if progress and t_srt is not None:
                progress.update(t_srt, advance=1)
            
            # Write SRT file
            if progress:
                t_write = progress.add_task("Write File", total=1)
            output_path.write_text(srt_content, encoding='utf-8')
            if progress and t_write is not None:
                progress.update(t_write, advance=1)
            
            console.print(f"[green]✓ Subtitles saved to: {output_path}[/green]")
            
            # Display summary
            self.display_summary(segments, input_path.name, str(output_path))

            # Optional remux step
            if remux:
                try:
                    self.remux_subtitles(
                        source_media=input_path,
                        srt_file=output_path,
                        language_code=remux_language or (language if language else ("en" if translate else "und")),
                        overwrite=remux_overwrite
                    )
                except Exception as e:  # pragma: no cover
                    console.print(f"[yellow]Remux failed: {e}[/yellow]")
            
            return True
            
        finally:
            # Clean up temporary audio file
            try:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            except Exception:
                pass
            if progress:
                progress.stop()
        # Should not reach here; explicit False for static analysis
        return False
    
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

    def remux_subtitles(self, source_media: Path | str, srt_file: Path | str, language_code: str = "en", overwrite: bool = False) -> Path:
        """Remux generated SRT into a new media file with added .subbed before extension.

        - Keeps original SRT file.
        - Copies original audio/video streams (no re-encode).
        - Adds SRT as a new subtitle stream with optional language metadata.
        """
        src = Path(source_media)
        srt = Path(srt_file)
        if not src.exists():
            raise FileNotFoundError(f"Source media not found: {src}")
        if not srt.exists():
            raise FileNotFoundError(f"SRT file not found: {srt}")
        remux_path = src.with_name(f"{src.stem}.subbed{src.suffix}")
        if remux_path.exists() and not overwrite:
            console.print(f"[yellow]Remux target exists: {remux_path} (use --remux-overwrite to replace)[/yellow]")
            return remux_path
        console.print(f"[blue]Remuxing with subtitles -> {remux_path.name}[/blue]")
        # ffmpeg command: copy streams, add subtitle
        # Use -map 0 to include all original streams, then add SRT as new input
        # Language metadata if provided
        cmd = [
            'ffmpeg', '-y' if overwrite else '-n',
            '-i', str(src),
            '-i', str(srt),
            '-map', '0', '-map', '1:0',
            '-c', 'copy',
            '-c:s:1', 'srt',  # ensure SRT codec for new track (position after original subs)
        ]
        if language_code:
            cmd += ['-metadata:s:s:1', f'language={language_code}']
        cmd.append(str(remux_path))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr else "ffmpeg remux failed")
            console.print(f"[green]✓ Remux complete: {remux_path}[/green]")
        except Exception as e:
            raise RuntimeError(f"Remux failed: {e}")
        return remux_path


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
    parser.add_argument('--device-order',
                       help='Comma-separated list of devices to try in order (e.g., cuda,igpu,cpu). Overrides --device when provided. igpu currently maps to optimized CPU path.')
    parser.add_argument('--no-fallback', action='store_true',
                       help='Disable automatic CUDA->CPU fallback when GPU initialization fails')
    parser.add_argument('--remux', action='store_true',
                       help='After generating SRT, remux it into a new media file with .subbed before extension (keeps SRT)')
    parser.add_argument('--remux-language',
                       help='Language code metadata for remuxed subtitle track (default: source language or en if translated)')
    parser.add_argument('--remux-overwrite', action='store_true',
                       help='Overwrite existing .subbed file if present')
    parser.add_argument('--diag', action='store_true',
                       help='Print diagnostics about model/device, CUDA availability, and compute type')
    
    args = parser.parse_args()
    
    # Initialize the application
    app = WhisperMVPClean()
    
    # Check dependencies
    if not app.check_ffmpeg():
        console.print("[red]✗ FFmpeg not found! Please install FFmpeg.[/red]")
        console.print("[yellow]Windows: Use the PowerShell script in README.md[/yellow]")
        sys.exit(1)
    
    # Process the file
    device_order = None
    if args.device_order:
        device_order = [d.strip() for d in args.device_order.split(',') if d.strip()]

    success = app.process_file(
        args.input,
        args.output,
        args.model,
        args.lang,
        args.translate,
        args.beam,
        args.device,
        args.compute,
        allow_fallback = (not args.no_fallback),
        device_order = device_order,
        remux = args.remux,
        remux_language = args.remux_language,
        remux_overwrite = args.remux_overwrite
    )

    if success and args.diag:
        # Diagnostics: report CUDA DLL presence, device actually used, compute type
        used_device = app.device or args.device
        compute = app.compute_type
        try:
            import torch
            torch_cuda = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if torch_cuda else 'N/A'
            torch_version = torch.__version__
        except Exception:
            torch_cuda = False
            gpu_name = 'N/A'
            torch_version = 'missing'
        ready, missing = app._check_cuda_runtime()
        console.rule("Diagnostics")
        console.print(f"Model: {app.model_name}")
        console.print(f"Requested Device: {args.device} | Used Device: {used_device}")
        console.print(f"Compute Type: {compute}")
        console.print(f"Torch CUDA Available: {torch_cuda} (torch {torch_version})")
        console.print(f"GPU: {gpu_name}")
        if used_device == 'cuda':
            if ready:
                console.print("CUDA Runtime DLL Check: OK")
            else:
                console.print(f"[yellow]Missing CUDA/cuDNN DLLs: {', '.join(missing)}[/yellow]")
        console.rule()
    
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