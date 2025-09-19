"""
Automatic Speech Recognition (ASR) module using faster-whisper for the subtitle translator package.
"""
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import tempfile
import threading
import time
from dataclasses import dataclass

from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress

console = Console()

@dataclass
class Segment:
    """Represents a subtitle segment with timing and text"""
    id: int
    start: float
    end: float
    text: str
    source_text: Optional[str] = None

class WhisperASR:
    """Wrapper for faster-whisper ASR model with advanced features"""
    
    def __init__(
        self, 
        model_name: str = "medium", 
        device: str = "cuda",
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        no_fallback: bool = False,
    ):
        """Initialize the WhisperASR model
        
        Args:
            model_name: The name/size of the Whisper model to use
            device: Device to use ("cuda" or "cpu")
            compute_type: Compute type (float16, int8_float16, int8)
            beam_size: Beam size for decoding
            no_fallback: If True, disable CPU fallback
        """
        self.model_name = model_name
        self.device = device
        self.no_fallback = no_fallback
        self.beam_size = beam_size
        
        # Set default compute type based on device if not specified
        if compute_type is None:
            self.compute_type = "float16" if device == "cuda" else "int8_float16"
        else:
            self.compute_type = compute_type
            
        # Lazy loading - model will be initialized on first use
        self._model = None
    
    @property
    def model(self) -> WhisperModel:
        """Lazy-load and cache the Whisper model"""
        if self._model is None:
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as e:
                if not self.no_fallback and self.device == "cuda":
                    console.print(f"[yellow]GPU acceleration failed: {e}[/yellow]")
                    console.print("[yellow]Falling back to CPU execution...[/yellow]")
                    self._model = WhisperModel(
                        self.model_name,
                        device="cpu",
                        compute_type="int8_float16",
                    )
                else:
                    raise
        return self._model
    
    def transcribe(
        self, 
        audio_path: Union[str, Path], 
        language: Optional[str] = None,
        task: str = "transcribe",
        progress_callback: Optional[Any] = None,
    ) -> List[Segment]:
        """Transcribe audio file to segments
        
        Args:
            audio_path: Path to the audio file
            language: Language code (auto-detect if None)
            task: Task to perform ("transcribe" or "translate")
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of transcription segments
        """
        # Initialize progress tracking if callback provided
        transcribe_done = threading.Event()
        progress_thread = None
        
        if progress_callback:
            def update_progress():
                progress = 0
                pulse_step = 2
                direction = 1
                while not transcribe_done.is_set():
                    progress += pulse_step * direction
                    if progress >= 90:
                        direction = -1
                    elif progress <= 0:
                        direction = 1
                    progress_callback(progress)
                    time.sleep(0.1)
            
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()
        
        try:
            # Run transcription
            segments, info = self.model.transcribe(
                str(audio_path),
                language=None if language == "auto" else language,
                task=task,
                beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            
            # Convert segments to our internal format
            result = []
            for i, seg in enumerate(segments):
                result.append(Segment(
                    id=i + 1,  # 1-indexed for SRT
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                ))
            
            # Return detected language in the first segment if auto-detect was used
            detected_lang = info.language
            if language == "auto" and detected_lang and result:
                console.print(f"[blue]Detected language: {detected_lang}[/blue]")
            
            return result
        
        finally:
            # Stop the progress thread if it was created
            if progress_thread:
                transcribe_done.set()
                progress_thread.join(timeout=1)
    
    def segments_to_srt(self, segments: List[Segment], output_path: Union[str, Path]) -> None:
        """Write segments to an SRT file
        
        Args:
            segments: List of transcription segments
            output_path: Path to output SRT file
        """
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                # Format timestamps as SRT format (HH:MM:SS,mmm)
                start_time = self._format_timestamp(seg.start)
                end_time = self._format_timestamp(seg.end)
                
                # Write SRT entry
                f.write(f"{seg.id}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{seg.text}\n\n")
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
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