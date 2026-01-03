"""
First-run setup and secrets management for PhotoPipe.

Handles secure storage of API keys and user preferences.
"""

import os
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class UserSettings(BaseModel):
    """User-specific settings stored separately from config."""
    anthropic_api_key: Optional[str] = None
    default_location: Optional[str] = None
    default_people: list[str] = []
    copyright_holder: Optional[str] = None
    setup_complete: bool = False


def get_settings_path() -> Path:
    """Get the path to the user settings file."""
    # Check for Docker data directory first
    data_dir = os.environ.get("PHOTOPIPE_DATA_DIR")
    if data_dir:
        settings_dir = Path(data_dir) / "config"
    else:
        settings_dir = Path.home() / ".photopipe"

    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir / "settings.json"


def load_settings() -> UserSettings:
    """Load user settings from disk."""
    settings_path = get_settings_path()

    if settings_path.exists():
        try:
            with open(settings_path) as f:
                data = json.load(f)
            return UserSettings(**data)
        except (json.JSONDecodeError, Exception):
            pass

    return UserSettings()


def save_settings(settings: UserSettings) -> None:
    """Save user settings to disk."""
    settings_path = get_settings_path()

    with open(settings_path, "w") as f:
        json.dump(settings.model_dump(), f, indent=2)

    # Set restrictive permissions (owner read/write only)
    os.chmod(settings_path, 0o600)


def is_setup_complete() -> bool:
    """Check if initial setup has been completed."""
    settings = load_settings()
    return settings.setup_complete


def get_api_key() -> Optional[str]:
    """
    Get the Anthropic API key.

    Checks in order:
    1. Environment variable ANTHROPIC_API_KEY
    2. User settings file
    """
    # Environment variable takes precedence
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    # Fall back to settings file
    settings = load_settings()
    return settings.anthropic_api_key


def complete_setup(
    api_key: Optional[str] = None,
    default_location: Optional[str] = None,
    default_people: Optional[list[str]] = None,
    copyright_holder: Optional[str] = None,
) -> None:
    """Complete the initial setup with user-provided values."""
    settings = load_settings()

    if api_key:
        settings.anthropic_api_key = api_key
    if default_location:
        settings.default_location = default_location
    if default_people is not None:
        settings.default_people = default_people
    if copyright_holder:
        settings.copyright_holder = copyright_holder

    settings.setup_complete = True
    save_settings(settings)
