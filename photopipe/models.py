"""
Data models for PhotoPipe.

Pydantic models for batches, photos, locations, and OCR results.
"""

from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DateSource(str, Enum):
    """Source of date extraction."""
    OCR_BACK = "ocr_back"
    OCR_FRONT = "ocr_front"
    BATCH_DEFAULT = "batch_default"
    AI_ESTIMATED = "ai_estimated"
    MANUAL = "manual"


class DateConfidence(str, Enum):
    """Confidence level of date extraction."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PhotoStatus(str, Enum):
    """Processing status of a photo."""
    INGESTED = "ingested"
    PROCESSED = "processed"
    REVIEWED = "reviewed"
    FINALIZED = "finalized"


class BatchStatus(str, Enum):
    """Processing status of a batch."""
    PENDING = "pending"
    PROCESSING = "processing"
    REVIEW = "review"
    COMPLETE = "complete"


class PhotoPhase(str, Enum):
    """Lifecycle phase of a photo in the rebuild pipeline."""
    CAPTURED = "captured"
    CURATED = "curated"
    FINALIZED = "finalized"


class BucketStatus(str, Enum):
    """Lifecycle status of a capture bucket."""
    OPEN = "open"
    CLOSED = "closed"
    CONVERTED = "converted"


class AIJobStatus(str, Enum):
    """Status of an AI batch job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LocationAccuracy(str, Enum):
    """Accuracy level of geocoded location."""
    EXACT = "exact"
    APPROXIMATE = "approximate"
    REGION = "region"


class Location(BaseModel):
    """Geocoded location with coordinates."""
    description: str
    address: Optional[str] = None
    latitude: float
    longitude: float
    accuracy: LocationAccuracy = LocationAccuracy.APPROXIMATE

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "Location":
        """Create from dictionary."""
        return cls(**data)


class OCRResult(BaseModel):
    """Result from OCR processing."""
    raw_text: str
    confidence: float
    detected_dates: list[str] = Field(default_factory=list)
    word_confidences: dict[str, float] = Field(default_factory=dict)
    preprocessing_applied: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "OCRResult":
        """Create from dictionary."""
        return cls(**data)


class AIDateEstimate(BaseModel):
    """AI-generated date estimate from Claude Vision."""
    year: Optional[int] = None
    year_range: Optional[tuple[int, int]] = None
    month: Optional[int] = None  # 1-12, more specific than season
    season: Optional[str] = None
    confidence: DateConfidence
    evidence: list[str] = Field(default_factory=list)
    reasoning: str
    location_guess: Optional[str] = None  # AI's guess at location
    location_confidence: Optional[str] = None  # high/medium/low
    location_evidence: list[str] = Field(default_factory=list)

    def get_best_date(self, southern_hemisphere: bool = False) -> Optional[date]:
        """Get the best single date estimate."""
        if self.year:
            # Use month if specified, otherwise derive from season
            if self.month:
                month = self.month
            elif self.season:
                # Adjust for southern hemisphere (seasons are reversed)
                season_months = {
                    "spring": 4 if not southern_hemisphere else 10,
                    "summer": 7 if not southern_hemisphere else 1,
                    "fall": 10 if not southern_hemisphere else 4,
                    "autumn": 10 if not southern_hemisphere else 4,
                    "winter": 1 if not southern_hemisphere else 7,
                }
                month = season_months.get(self.season.lower(), 6)
            else:
                month = 6  # Default to middle of year
            return date(self.year, month, 15)
        elif self.year_range:
            mid_year = (self.year_range[0] + self.year_range[1]) // 2
            return date(mid_year, 6, 15)
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "AIDateEstimate":
        """Create from dictionary."""
        # Handle missing fields for backwards compatibility
        data.setdefault("month", None)
        data.setdefault("location_guess", None)
        data.setdefault("location_confidence", None)
        data.setdefault("location_evidence", [])
        return cls(**data)


class PhotoPair(BaseModel):
    """A paired front/back photo scan."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    batch_id: str
    sequence_num: int
    front_path: Path
    back_path: Optional[Path] = None

    # Extracted metadata (populated during processing)
    extracted_date: Optional[date] = None
    date_source: Optional[DateSource] = None
    date_confidence: Optional[DateConfidence] = None
    ocr_text_back: Optional[str] = None
    ocr_raw_results: Optional[dict] = None
    ai_analysis: Optional[dict] = None

    # Final metadata (after review)
    final_date: Optional[date] = None
    final_location: Optional[Location] = None
    final_description: Optional[str] = None
    final_keywords: list[str] = Field(default_factory=list)

    # Processing state
    status: PhotoStatus = PhotoStatus.INGESTED
    phase: PhotoPhase = PhotoPhase.FINALIZED
    bucket_id: Optional[str] = None
    needs_review: bool = False
    review_notes: Optional[str] = None

    # Handwriting OCR (from rebuild pipeline)
    handwriting_ocr_text: Optional[str] = None
    handwriting_ocr_provider: Optional[str] = None
    handwriting_ocr_confidence: Optional[float] = None

    # Output paths
    output_front_path: Optional[Path] = None
    output_back_path: Optional[Path] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True

    def get_effective_date(self) -> Optional[date]:
        """Get the best available date (final > extracted)."""
        return self.final_date or self.extracted_date

    def has_back(self) -> bool:
        """Check if photo has a back scan."""
        return self.back_path is not None

    def get_ocr_result(self) -> Optional[OCRResult]:
        """Get OCR result object if available."""
        if self.ocr_raw_results:
            return OCRResult.from_dict(self.ocr_raw_results)
        return None

    def get_ai_estimate(self) -> Optional[AIDateEstimate]:
        """Get AI date estimate if available."""
        if self.ai_analysis:
            return AIDateEstimate.from_dict(self.ai_analysis)
        return None

    def to_db_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        data = self.model_dump()
        # Convert Path objects to strings
        for key in ["front_path", "back_path", "output_front_path", "output_back_path"]:
            if data.get(key):
                data[key] = str(data[key])
        # Convert Location to JSON
        if data.get("final_location"):
            data["final_location"] = data["final_location"]
        return data

    @classmethod
    def from_db_dict(cls, data: dict) -> "PhotoPair":
        """Create from database dictionary."""
        # Convert string paths back to Path objects
        for key in ["front_path", "back_path", "output_front_path", "output_back_path"]:
            if data.get(key):
                data[key] = Path(data[key])
        # Handle Location reconstruction
        if data.get("final_location") and isinstance(data["final_location"], dict):
            data["final_location"] = Location.from_dict(data["final_location"])
        return cls(**data)


class Batch(BaseModel):
    """A batch of photos to process together."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    location_description: Optional[str] = None
    location: Optional[Location] = None
    event_description: Optional[str] = None
    people: list[str] = Field(default_factory=list)
    input_folder: Optional[Path] = None
    status: BatchStatus = BatchStatus.PENDING

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True

    def get_date_range_str(self) -> str:
        """Get human-readable date range string."""
        if self.date_start and self.date_end:
            if self.date_start == self.date_end:
                return self.date_start.strftime("%B %d, %Y")
            elif self.date_start.year == self.date_end.year:
                if self.date_start.month == self.date_end.month:
                    return f"{self.date_start.strftime('%B %d')} - {self.date_end.strftime('%d, %Y')}"
                return f"{self.date_start.strftime('%B')} - {self.date_end.strftime('%B %Y')}"
            return f"{self.date_start.strftime('%B %Y')} - {self.date_end.strftime('%B %Y')}"
        elif self.date_start:
            return f"From {self.date_start.strftime('%B %Y')}"
        elif self.date_end:
            return f"Until {self.date_end.strftime('%B %Y')}"
        return "No date range specified"

    def calculate_date_for_sequence(self, sequence_num: int, total_photos: int) -> Optional[date]:
        """
        Calculate a date for a photo based on its sequence within the batch.

        Spreads photos evenly across the batch date range.
        """
        if not self.date_start:
            return None

        if not self.date_end or self.date_start == self.date_end:
            return self.date_start

        if total_photos <= 1:
            return self.date_start

        # Calculate the fraction through the batch
        fraction = (sequence_num - 1) / (total_photos - 1) if total_photos > 1 else 0

        # Calculate days between start and end
        delta = self.date_end - self.date_start
        days_offset = int(delta.days * fraction)

        from datetime import timedelta
        return self.date_start + timedelta(days=days_offset)

    def to_db_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        data = self.model_dump()
        # Convert Path to string
        if data.get("input_folder"):
            data["input_folder"] = str(data["input_folder"])
        # Location is stored as JSON
        if data.get("location"):
            data["location_lat"] = data["location"]["latitude"]
            data["location_lon"] = data["location"]["longitude"]
        del data["location"]
        return data

    @classmethod
    def from_db_dict(cls, data: dict) -> "Batch":
        """Create from database dictionary."""
        # Convert string path back to Path
        if data.get("input_folder"):
            data["input_folder"] = Path(data["input_folder"])
        # Reconstruct location from lat/lon
        if data.get("location_lat") and data.get("location_lon"):
            data["location"] = Location(
                description=data.get("location_description", ""),
                latitude=data["location_lat"],
                longitude=data["location_lon"],
            )
        # Remove db-specific fields
        for key in ["location_lat", "location_lon"]:
            data.pop(key, None)
        return cls(**data)


class BatchTemplate(BaseModel):
    """Saved batch configuration template."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    location_description: Optional[str] = None
    location: Optional[Location] = None
    people: list[str] = Field(default_factory=list)
    default_keywords: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    def to_batch(self, batch_name: str) -> Batch:
        """Create a new Batch from this template."""
        return Batch(
            name=batch_name,
            location_description=self.location_description,
            location=self.location,
            people=self.people.copy(),
        )


class ProcessingLogEntry(BaseModel):
    """Log entry for processing actions."""
    id: Optional[int] = None
    photo_id: Optional[str] = None
    batch_id: Optional[str] = None
    action: str
    details: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchReport(BaseModel):
    """Report generated when finalizing a batch."""
    batch_name: str
    created: datetime
    finalized: datetime
    photo_count: int
    date_range: dict[str, Optional[str]]
    location: Optional[dict] = None
    date_source_breakdown: dict[str, int]
    people_tagged: list[str]
    photos: list[dict]

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.model_dump(), indent=2, default=str)


class Bucket(BaseModel):
    """A capture bucket: a working collection of photos before batch conversion."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    helper_name: Optional[str] = None
    status: BucketStatus = BucketStatus.OPEN
    batch_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class AIJob(BaseModel):
    """A queued or running AI analysis job for a batch."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    batch_id: str
    provider: str  # e.g. "anthropic_batch"
    provider_job_id: Optional[str] = None
    status: AIJobStatus = AIJobStatus.QUEUED
    submitted_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    photo_ids: list[str] = Field(default_factory=list)
    result_summary: Optional[dict] = None

    class Config:
        use_enum_values = True


class Face(BaseModel):
    """A single detected face in a photo, with its ArcFace embedding."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    photo_id: str
    batch_id: str
    bbox: tuple[int, int, int, int]   # x, y, w, h in front-image pixels
    embedding: list[float]            # 512-d, L2-normalized
    crop_path: Optional[Path] = None
    cluster_id: Optional[str] = None
    detection_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class FaceCluster(BaseModel):
    """A group of faces believed to be the same person within one batch."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    batch_id: str
    label: Optional[str] = None
    representative_face_id: Optional[str] = None
    is_noise: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True
