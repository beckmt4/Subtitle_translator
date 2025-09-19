"""
Subtitle quality profiles for the subtitle translator package.
"""
from typing import Dict, Any, Optional
import json
from pathlib import Path
import os

from rich.console import Console

console = Console()

# Default profiles
DEFAULT_PROFILES = {
    "netflix": {
        "max_line_chars": 42,
        "max_lines": 2,
        "max_cps": 20.0,
        "min_duration": 0.5,
        "min_gap": 0.2,
        "description": "Netflix English subtitling guidelines"
    },
    "bbc": {
        "max_line_chars": 37,
        "max_lines": 2,
        "max_cps": 18.0,
        "min_duration": 0.833,  # 5/6 second (20 frames at 24fps)
        "min_gap": 0.167,       # 1/6 second (4 frames at 24fps)
        "description": "BBC subtitle guidelines"
    },
    "timed-text": {
        "max_line_chars": 40,
        "max_lines": 2,
        "max_cps": 17.0,
        "min_duration": 0.7,
        "min_gap": 0.2,
        "description": "W3C Timed Text guidelines"
    },
    "youtube": {
        "max_line_chars": 60,
        "max_lines": 2,
        "max_cps": 25.0,
        "min_duration": 0.3,
        "min_gap": 0.1,
        "description": "YouTube CC optimization"
    },
    "liberal": {
        "max_line_chars": 80,
        "max_lines": 3,
        "max_cps": 30.0,
        "min_duration": 0.2,
        "min_gap": 0.0,
        "description": "Relaxed constraints for technical content"
    }
}

# Path for user-defined profiles
def get_user_profiles_path() -> Path:
    """Get the path to user-defined profiles
    
    Returns:
        Path to user profiles JSON file
    """
    # Use platform-appropriate config location
    if os.name == "nt":  # Windows
        config_dir = Path(os.environ.get("APPDATA", "")) / "SubtitleTranslator"
    else:  # Unix-like
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "subtitle-translator"
    
    # Ensure directory exists
    config_dir.mkdir(parents=True, exist_ok=True)
    
    return config_dir / "profiles.json"

def get_profile(name: str) -> Optional[Dict[str, Any]]:
    """Get a subtitle quality profile by name
    
    Args:
        name: Profile name
        
    Returns:
        Profile settings dict or None if not found
    """
    # Try user profiles first
    user_profiles = load_user_profiles()
    if name in user_profiles:
        return user_profiles[name]
    
    # Fall back to default profiles
    if name in DEFAULT_PROFILES:
        return DEFAULT_PROFILES[name]
    
    console.print(f"[yellow]Profile '{name}' not found. Using default settings.[/yellow]")
    return None

def list_profiles() -> Dict[str, Dict[str, Any]]:
    """List all available profiles (defaults and user-defined)
    
    Returns:
        Dict of profile name -> settings
    """
    # Start with default profiles
    profiles = DEFAULT_PROFILES.copy()
    
    # Add user profiles, overriding defaults if same name
    user_profiles = load_user_profiles()
    profiles.update(user_profiles)
    
    return profiles

def load_user_profiles() -> Dict[str, Dict[str, Any]]:
    """Load user-defined profiles from config file
    
    Returns:
        Dict of user profiles
    """
    profiles_path = get_user_profiles_path()
    
    if not profiles_path.exists():
        return {}
    
    try:
        with open(profiles_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[yellow]Error loading user profiles: {e}[/yellow]")
        return {}

def save_user_profile(
    name: str,
    settings: Dict[str, Any],
    overwrite: bool = False
) -> bool:
    """Save a user-defined profile
    
    Args:
        name: Profile name
        settings: Profile settings
        overwrite: Whether to overwrite existing profile
        
    Returns:
        True if saved successfully, False otherwise
    """
    profiles_path = get_user_profiles_path()
    
    # Load existing profiles
    profiles = load_user_profiles()
    
    # Check if profile exists
    if name in profiles and not overwrite:
        console.print(f"[yellow]Profile '{name}' already exists. Use overwrite=True to replace.[/yellow]")
        return False
    
    # Add or update profile
    profiles[name] = settings
    
    # Save profiles
    try:
        with open(profiles_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error saving profile: {e}[/red]")
        return False

def delete_user_profile(name: str) -> bool:
    """Delete a user-defined profile
    
    Args:
        name: Profile name
        
    Returns:
        True if deleted successfully, False otherwise
    """
    # Cannot delete default profiles
    if name in DEFAULT_PROFILES:
        console.print(f"[yellow]Cannot delete default profile '{name}'.[/yellow]")
        return False
    
    profiles_path = get_user_profiles_path()
    
    # Load existing profiles
    profiles = load_user_profiles()
    
    # Check if profile exists
    if name not in profiles:
        console.print(f"[yellow]Profile '{name}' not found.[/yellow]")
        return False
    
    # Remove profile
    del profiles[name]
    
    # Save profiles
    try:
        with open(profiles_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error deleting profile: {e}[/red]")
        return False