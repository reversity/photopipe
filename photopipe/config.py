"""
Configuration management for PhotoPipe.

Loads configuration from YAML file with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv()  # Also check current directory


class PathsConfig(BaseModel):
    """File system paths configuration."""
    input_folder: Path = Field(default=Path.home() / "Pictures" / "Scanner_Input")
    output_folder: Path = Field(default=Path.home() / "Pictures" / "Scanned_Photos")
    archive_folder: Path = Field(default=Path.home() / "Pictures" / "Scanned_Photos" / "_archive")
    database: Path = Field(default=Path.home() / ".photopipe" / "photopipe.db")

    def model_post_init(self, __context) -> None:
        """Expand ~ in all paths after initialization."""
        object.__setattr__(self, 'input_folder', Path(self.input_folder).expanduser().resolve())
        object.__setattr__(self, 'output_folder', Path(self.output_folder).expanduser().resolve())
        object.__setattr__(self, 'archive_folder', Path(self.archive_folder).expanduser().resolve())
        object.__setattr__(self, 'database', Path(self.database).expanduser().resolve())


class ScannerConfig(BaseModel):
    """Scanner and naming pattern configuration."""
    # File naming patterns (FastFoto uses _b suffix for backs)
    front_pattern: str = "{name}_{num}.jpg"
    back_pattern: str = "{name}_{num}_b.jpg"
    default_name_prefix: str = "photo"
    watch_interval_seconds: int = 2

    # Scanner hardware settings
    device: Optional[str] = None  # SANE device name, auto-detect if None
    resolution: int = 600  # DPI
    mode: str = "color"  # color, gray, lineart
    duplex: bool = True  # Scan front and back
    source: str = "ADF"  # ADF (auto document feeder) or Flatbed


class VLMConfig(BaseModel):
    """Vision-language model transport configuration."""
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    max_image_dimension: int = 1568  # matches Claude vision token grid
    cache_ttl: Literal["5m", "1h"] = "5m"
    batch_api_threshold: int = 50  # use Batch API when >= this many photos


class HandwritingOCRConfig(BaseModel):
    """Handwriting OCR (photo backs) configuration.

    Primary provider is Mistral OCR 3; falls back to the Claude VLM
    when Mistral confidence drops below ``confidence_fallback_threshold``
    (and the user has not pinned ``provider="mistral"``).
    """
    provider: Literal["mistral", "claude", "auto"] = "auto"
    mistral_api_key_env_var: str = "MISTRAL_API_KEY"
    mistral_model: str = "mistral-ocr-3"
    mistral_max_image_dim: int = 2048
    confidence_fallback_threshold: float = 0.6  # below this, fall back to VLM
    use_batch_api: bool = True


class MetadataConfig(BaseModel):
    """Metadata defaults configuration."""
    default_timezone: str = "America/New_York"
    copyright_template: str = "\u00a9 {year} Family Archive"


class OutputConfig(BaseModel):
    """Output organization configuration."""
    folder_structure: str = "{year}/{year}-{month}_{batch_name}"
    filename_template: str = "{date}_{batch_name}_{sequence:04d}_{side}"
    preserve_originals: bool = True
    generate_web_copies: bool = False
    web_copy_max_dimension: int = 2048


class UIConfig(BaseModel):
    """UI configuration."""
    theme: str = "light"
    thumbnails_per_page: int = 20


class Config(BaseModel):
    """Main configuration container."""
    paths: PathsConfig = Field(default_factory=PathsConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    handwriting_ocr: HandwritingOCRConfig = Field(default_factory=HandwritingOCRConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        self.paths.input_folder.mkdir(parents=True, exist_ok=True)
        self.paths.output_folder.mkdir(parents=True, exist_ok=True)
        self.paths.archive_folder.mkdir(parents=True, exist_ok=True)
        self.paths.database.parent.mkdir(parents=True, exist_ok=True)

    def get_api_key(self) -> Optional[str]:
        """Get the Anthropic API key from environment or settings."""
        # Try environment first
        key = os.environ.get(self.vlm.api_key_env_var)
        if key:
            return key

        # Fall back to user settings
        try:
            from photopipe.setup import get_api_key as get_settings_api_key
            return get_settings_api_key()
        except ImportError:
            return None


# Default config file locations
DEFAULT_CONFIG_PATHS = [
    Path.cwd() / "config.yaml",
    Path.home() / ".photopipe" / "config.yaml",
]


def find_config_file() -> Optional[Path]:
    """Find the first existing config file from default locations."""
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get the global configuration instance.

    Loads from the first found config file, or returns defaults.
    Result is cached for performance.
    """
    config_path = find_config_file()
    if config_path:
        return Config.from_yaml(config_path)
    return Config()


def reload_config() -> Config:
    """Force reload configuration from file."""
    get_config.cache_clear()
    return get_config()


def save_config(config: Config, path: Optional[Path] = None) -> Path:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        path: Target path (defaults to ~/.photopipe/config.yaml)

    Returns:
        Path where config was saved
    """
    if path is None:
        path = Path.home() / ".photopipe" / "config.yaml"

    config.to_yaml(path)
    reload_config()
    return path
