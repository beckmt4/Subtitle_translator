"""
CLI application for subtitle-translator
"""
from pathlib import Path
from typing import Optional, List, Annotated
import typer

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from subtitle_translator import (
    WhisperASR,
    TranslationEngine,
    SubtitleQualityShaper,
    extract_audio,
    remux_subtitles,
    get_profile,
    list_profiles,
)

app = typer.Typer(
    help="GPU-accelerated subtitle generation and translation with quality controls",
    add_completion=False,
)
console = Console()

@app.command()
def transcribe(
    input_file: Annotated[Path, typer.Argument(help="Input media file (video/audio)")],
    output_file: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output SRT path")] = None,
    language: Annotated[str, typer.Option(help="Source language code")] = "auto",
    model: Annotated[str, typer.Option(help="Whisper model size")] = "medium",
    device: Annotated[str, typer.Option(help="Device (cuda or cpu)")] = "cuda",
    compute_type: Annotated[Optional[str], typer.Option(help="Compute type (float16, int8_float16, int8)")] = None,
    beam_size: Annotated[int, typer.Option(help="Beam size for decoding")] = 5,
    no_fallback: Annotated[bool, typer.Option(help="Disable CPU fallback")] = False,
    verbose: Annotated[bool, typer.Option(help="Show verbose output")] = False,
):
    """Transcribe audio/video to SRT subtitles in the original language"""
    # Input validation
    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    # Determine output path if not specified
    if output_file is None:
        output_file = input_file.with_suffix(".srt")
    
    # Create ASR engine
    asr = WhisperASR(
        model_name=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        no_fallback=no_fallback,
    )
    
    # Extract audio if needed and run transcription
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Extract audio if input is video
        t_extract = progress.add_task("Extracting audio...", total=100)
        audio_path = extract_audio(input_file)
        progress.update(t_extract, completed=100)
        
        # Transcribe
        t_transcribe = progress.add_task("Transcribing...", total=None)
        segments = asr.transcribe(audio_path, language=language)
        progress.update(t_transcribe, completed=100)
        
        # Write SRT
        t_write = progress.add_task("Writing SRT...", total=100)
        asr.segments_to_srt(segments, output_file)
        progress.update(t_write, completed=100)
    
    console.print(f"[green]✓ Transcription complete: {output_file}[/green]")

@app.command()
def translate(
    input_file: Annotated[Path, typer.Argument(help="Input media file (video/audio)")],
    output_file: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output SRT path")] = None,
    language: Annotated[str, typer.Option(help="Source language code")] = "auto",
    asr_model: Annotated[str, typer.Option(help="Whisper model size")] = "medium",
    mt_model: Annotated[Optional[str], typer.Option(help="Translation model name")] = "facebook/nllb-200-distilled-600M",
    target_language: Annotated[str, typer.Option(help="Target language code")] = "eng_Latn",
    device: Annotated[str, typer.Option(help="Device (cuda or cpu)")] = "cuda",
    compute_type: Annotated[Optional[str], typer.Option(help="Compute type")] = None,
    max_line_chars: Annotated[int, typer.Option(help="Maximum characters per line")] = 42,
    max_lines: Annotated[int, typer.Option(help="Maximum lines per subtitle")] = 2,
    max_cps: Annotated[float, typer.Option(help="Maximum characters per second")] = 20.0,
    min_duration: Annotated[float, typer.Option(help="Minimum duration in seconds")] = 0.5,
    min_gap: Annotated[float, typer.Option(help="Minimum gap between subtitles")] = 0.2,
    remux: Annotated[bool, typer.Option(help="Remux subtitles into video")] = False,
    profile: Annotated[Optional[str], typer.Option(help="Use predefined quality profile")] = None,
):
    """Two-pass ASR+MT subtitle generation with quality controls"""
    # Input validation
    if not input_file.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    # Determine output path if not specified
    if output_file is None:
        output_file = input_file.with_suffix(f".{target_language}.srt")
    
    # Apply profile if specified
    if profile:
        profile_settings = get_profile(profile)
        if profile_settings:
            max_line_chars = profile_settings.get("max_line_chars", max_line_chars)
            max_lines = profile_settings.get("max_lines", max_lines)
            max_cps = profile_settings.get("max_cps", max_cps)
            min_duration = profile_settings.get("min_duration", min_duration)
            min_gap = profile_settings.get("min_gap", min_gap)
    
    # Create engines
    asr = WhisperASR(
        model_name=asr_model,
        device=device,
        compute_type=compute_type,
    )
    
    mt = TranslationEngine(
        model_name=mt_model,
        device=device,
    )
    
    quality = SubtitleQualityShaper(
        max_line_chars=max_line_chars,
        max_lines=max_lines,
        max_cps=max_cps,
        min_duration=min_duration,
        min_gap=min_gap,
    )
    
    # Phase 1: ASR
    console.print("[bold blue]Phase 1: Automatic Speech Recognition[/bold blue]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        # Extract audio if input is video
        t_extract = progress.add_task("Extracting audio...", total=100)
        audio_path = extract_audio(input_file)
        progress.update(t_extract, completed=100)
        
        # Transcribe
        t_transcribe = progress.add_task("Transcribing...", total=None)
        segments = asr.transcribe(audio_path, language=language)
        progress.update(t_transcribe, completed=100)
    
    # Phase 2: MT
    console.print("\n[bold green]Phase 2: Machine Translation[/bold green]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        t_translate = progress.add_task("Translating...", total=len(segments))
        translated_segments = mt.translate_segments(segments, source_lang=language, target_lang=target_language)
        progress.update(t_translate, completed=len(segments))
    
    # Phase 3: Quality Shaping & Output
    console.print("\n[bold cyan]Phase 3: Subtitle Quality & Output[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        t_shape = progress.add_task("Applying quality controls...", total=100)
        final_segments = quality.shape_segments(translated_segments)
        progress.update(t_shape, completed=100)
        
        t_write = progress.add_task("Writing SRT...", total=100)
        quality.write_srt(final_segments, output_file)
        progress.update(t_write, completed=100)
        
        if remux:
            t_remux = progress.add_task("Remuxing subtitles into video...", total=100)
            remux_path = remux_subtitles(input_file, output_file)
            progress.update(t_remux, completed=100)
            console.print(f"[green]✓ Remuxed output: {remux_path}[/green]")
    
    console.print(f"[green]✓ Translation complete: {output_file}[/green]")

@app.command()
def profiles():
    """List available quality profiles"""
    profiles = list_profiles()
    console.print("[bold]Available subtitle quality profiles:[/bold]")
    for name, settings in profiles.items():
        console.print(f"[bold cyan]{name}[/bold cyan]")
        for key, value in settings.items():
            console.print(f"  {key}: {value}")
        console.print()

if __name__ == "__main__":
    app()