"""
Subtitle Translator - Fast ASR→MT→SRT pipeline with readability controls

A package for high-quality subtitle generation from media files,
with support for:
- ASR (Automatic Speech Recognition) using Whisper via faster-whisper
- MT (Machine Translation) with external models (NLLB)
- Subtitle quality controls for readability
- Media remuxing capabilities
"""

__version__ = "0.1.0"

# Import core classes and functions
from .asr import WhisperASR, Segment
from .mt import TranslationEngine
from .quality import SubtitleQualityShaper
from .media import extract_audio, check_ffmpeg_available
from .remux import remux_subtitles, get_media_info, extract_subtitles, list_subtitle_streams
from .profiles import get_profile, list_profiles, save_user_profile, delete_user_profile
from .srt_io import read_srt, write_srt, adjust_timings, format_timestamp, parse_timestamp

__all__ = [
    # Classes
    "WhisperASR",
    "TranslationEngine",
    "SubtitleQualityShaper", 
    "Segment",
    
    # Media functions
    "extract_audio",
    "check_ffmpeg_available",
    "remux_subtitles",
    "get_media_info",
    "extract_subtitles",
    "list_subtitle_streams",
    
    # SRT handling
    "read_srt",
    "write_srt",
    "adjust_timings",
    "format_timestamp",
    "parse_timestamp",
    
    # Profiles
    "get_profile",
    "list_profiles",
    "save_user_profile",
    "delete_user_profile",
]