"""
PhotoPipe - Photo Scanning Metadata Pipeline

A tool for managing scanned photo metadata, OCR date extraction,
and AI-assisted date estimation for digitized family photos.
"""

__version__ = "0.1.0"
__author__ = "PhotoPipe"

from photopipe.config import get_config, Config
from photopipe.models import Batch, PhotoPair, Location, OCRResult
from photopipe.database import Database

__all__ = [
    "get_config",
    "Config",
    "Batch",
    "PhotoPair",
    "Location",
    "OCRResult",
    "Database",
]
