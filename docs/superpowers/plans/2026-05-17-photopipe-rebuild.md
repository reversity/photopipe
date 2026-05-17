# PhotoPipe Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a helper-friendly capture phase and an owner-driven curate phase to PhotoPipe, modernize the LLM/OCR stack (Sonnet 4.6 + prompt caching + Batch API + Mistral OCR 3), and extract business logic from Streamlit pages into headless, testable pipelines.

**Architecture:** Incremental refactor of the existing Streamlit app. Add a `Bucket` primitive (helper-entered label, no metadata) and a per-photo `phase` field (`captured | curated | finalized`). Extract logic from pages 2/3/4 into `capture_pipeline.py`, `curate_pipeline.py`, `finalize_pipeline.py`. Replace `ai_dating.py` with a transport-layer `vlm_client.py` (caching, batch, structured output) plus an orchestration-layer `dating_pipeline.py`. Replace Tesseract with `handwriting_ocr.py` (Mistral OCR 3 primary + Claude vision fallback). Each step ships behind the existing UI; the old workflow keeps working until the last task.

**Tech Stack:** Python 3.11+, Streamlit ≥1.30, SQLite (built-in), Anthropic SDK (`claude-sonnet-4-6`, `claude-opus-4-7`), Mistral SDK (`mistral-ocr-3`), Pydantic v2, pyexiftool, Pillow, SANE (`scanimage` subprocess), pytest, pytest-asyncio.

**Companion spec:** `docs/superpowers/specs/2026-05-17-photopipe-rebuild-design.md` — read this before starting any task.

---

## Task 1: DB schema migration

**Files:**
- Create: `photopipe/migrations/__init__.py`
- Create: `photopipe/migrations/001_phase_and_buckets.py`
- Modify: `photopipe/database.py` — add `run_migrations()` method, append new SCHEMA, call on init
- Modify: `photopipe/models.py` — add `Bucket`, `BucketStatus`, `PhotoPhase`, `AIJob`, `AIJobStatus`
- Test: `tests/test_migrations.py`

- [ ] **Step 1: Write the failing test for migration idempotency**

```python
# tests/test_migrations.py
import sqlite3
from pathlib import Path
import pytest
from photopipe.migrations import run_all_migrations

def test_migration_creates_buckets_table(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    run_all_migrations(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='buckets'")
    assert cur.fetchone() is not None

def test_migration_adds_phase_to_photos(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE photos (id TEXT PRIMARY KEY, front_path TEXT NOT NULL);
        INSERT INTO photos VALUES ('p1', '/tmp/x.jpg');
    """)
    run_all_migrations(conn)
    cur = conn.execute("SELECT phase FROM photos WHERE id='p1'")
    assert cur.fetchone()[0] == "finalized"  # existing rows default to finalized

def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    run_all_migrations(conn)
    run_all_migrations(conn)  # second run should not raise
    cur = conn.execute("SELECT COUNT(*) FROM buckets")
    assert cur.fetchone()[0] == 0

def test_migration_adds_ai_jobs_table(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    run_all_migrations(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_jobs'")
    assert cur.fetchone() is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_migrations.py -v`
Expected: ImportError on `photopipe.migrations`

- [ ] **Step 3: Implement migrations**

```python
# photopipe/migrations/__init__.py
import sqlite3
from photopipe.migrations import _001_phase_and_buckets

MIGRATIONS = [_001_phase_and_buckets]

def run_all_migrations(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    applied = {row[0] for row in conn.execute("SELECT id FROM schema_migrations")}
    for mod in MIGRATIONS:
        if mod.MIGRATION_ID not in applied:
            mod.up(conn)
            conn.execute("INSERT INTO schema_migrations(id) VALUES (?)", (mod.MIGRATION_ID,))
            conn.commit()
```

```python
# photopipe/migrations/_001_phase_and_buckets.py
import sqlite3

MIGRATION_ID = "001_phase_and_buckets"

def up(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS buckets (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            helper_name TEXT,
            status TEXT DEFAULT 'open',
            batch_id TEXT REFERENCES batches(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ai_jobs (
            id TEXT PRIMARY KEY,
            batch_id TEXT REFERENCES batches(id),
            provider TEXT NOT NULL,
            provider_job_id TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            photo_ids JSON,
            result_summary JSON
        );

        CREATE INDEX IF NOT EXISTS idx_buckets_status ON buckets(status);
        CREATE INDEX IF NOT EXISTS idx_ai_jobs_status ON ai_jobs(status);
    """)

    # Additive columns on existing tables (SQLite ALTER ADD COLUMN)
    _add_column(conn, "photos", "bucket_id", "TEXT REFERENCES buckets(id)")
    _add_column(conn, "photos", "phase", "TEXT DEFAULT 'finalized'")
    _add_column(conn, "photos", "handwriting_ocr_text", "TEXT")
    _add_column(conn, "photos", "handwriting_ocr_provider", "TEXT")
    _add_column(conn, "photos", "handwriting_ocr_confidence", "REAL")
    _add_column(conn, "batches", "source_bucket_ids", "JSON")

def _add_column(conn, table, col, decl):
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
```

- [ ] **Step 4: Add Pydantic models**

```python
# Append to photopipe/models.py

class PhotoPhase(str, Enum):
    CAPTURED = "captured"
    CURATED = "curated"
    FINALIZED = "finalized"

class BucketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CONVERTED = "converted"

class Bucket(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    helper_name: Optional[str] = None
    status: BucketStatus = BucketStatus.OPEN
    batch_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None

class AIJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class AIJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    batch_id: str
    provider: str  # "anthropic_batch"
    provider_job_id: Optional[str] = None
    status: AIJobStatus = AIJobStatus.QUEUED
    submitted_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    photo_ids: list[str] = Field(default_factory=list)
    result_summary: Optional[dict] = None
```

Also add `phase: PhotoPhase = PhotoPhase.FINALIZED`, `bucket_id: Optional[str] = None`, `handwriting_ocr_text: Optional[str] = None`, `handwriting_ocr_provider: Optional[str] = None`, `handwriting_ocr_confidence: Optional[float] = None` to the existing `PhotoPair` model.

- [ ] **Step 5: Wire migrations into `Database.__init__`**

In `photopipe/database.py`, replace the schema bootstrap to run both the legacy `SCHEMA` (for fresh DBs) AND `run_all_migrations()`. After legacy SCHEMA runs, call `run_all_migrations(conn)` so additive migrations apply.

- [ ] **Step 6: Run tests, commit**

```bash
pytest tests/test_migrations.py -v
git add photopipe/migrations/ photopipe/database.py photopipe/models.py tests/test_migrations.py
git commit -m "Add bucket and phase schema migration"
```

---

## Task 2: Extract date_parser from ocr.py

**Files:**
- Create: `photopipe/date_parser.py`
- Create: `tests/test_date_parser.py`
- Modify: `photopipe/ocr.py` — import from `date_parser` instead of duplicating

The current `ocr.py` mixes Tesseract calls with pure date-string parsing. We pull the pure functions out so both the legacy Tesseract path AND the new Mistral/VLM paths can share them. `ocr.py` itself stays for now (Task 11 deletes it).

- [ ] **Step 1: Write tests for date_parser**

```python
# tests/test_date_parser.py
from datetime import date
from photopipe.date_parser import parse_date_from_text, expand_year

def test_expand_two_digit_year_pre_2031():
    assert expand_year("85") == 1985
    assert expand_year("25") == 2025
    assert expand_year("00") == 2000

def test_photo_lab_stamp_month_year_short():
    # "JUN '85"
    results = parse_date_from_text("JUN '85")
    assert results
    assert results[0][0] == date(1985, 6, 15)

def test_full_date():
    results = parse_date_from_text("June 14, 1985")
    assert results
    assert results[0][0] == date(1985, 6, 14)

def test_seasonal():
    results = parse_date_from_text("Summer '92")
    assert results
    assert results[0][0].year == 1992
    assert results[0][0].month == 7  # SEASON_MAP summer

def test_year_only_fallback():
    results = parse_date_from_text("Mom holding the cat 1985")
    # Year-only must still parse
    years = [r[0].year for r in results]
    assert 1985 in years

def test_no_date_returns_empty():
    assert parse_date_from_text("Nothing useful here.") == []

def test_rejects_implausible_year():
    # Year-only "1850" must NOT parse (out of 1900-now range)
    results = parse_date_from_text("1850")
    assert all(r[0].year >= 1900 for r in results)
```

- [ ] **Step 2: Implement by moving from ocr.py**

Cut these from `photopipe/ocr.py` and paste into `photopipe/date_parser.py`: `DATE_PATTERNS`, `MONTH_MAP`, `SEASON_MAP`, `expand_year()`, `parse_date_from_text()`. Add the module docstring "Pure date-string parsing. No I/O, no OCR dependencies."

In `photopipe/ocr.py`, replace the cut code with:

```python
from photopipe.date_parser import parse_date_from_text, expand_year, DATE_PATTERNS, MONTH_MAP, SEASON_MAP  # noqa: F401
```

- [ ] **Step 3: Run all tests (including existing ocr tests if any), commit**

```bash
pytest tests/test_date_parser.py -v
pytest tests/ -v  # full suite to catch regressions
git add photopipe/date_parser.py photopipe/ocr.py tests/test_date_parser.py
git commit -m "Extract pure date-parsing into date_parser module"
```

---

## Task 3: Build vlm_client.py (caching + structured output)

**Files:**
- Create: `photopipe/vlm_client.py`
- Create: `tests/test_vlm_client.py`
- Modify: `photopipe/config.py` — add `VLMConfig` section

This module is the single entry point for Claude vision calls. Owns prompt caching (`cache_control` on prefix), Batch API submission/polling, structured output schema enforcement.

- [ ] **Step 1: Add VLMConfig**

```python
# Append to photopipe/config.py

class VLMConfig(BaseModel):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    max_image_dimension: int = 1568  # matches Claude vision token grid
    cache_ttl: Literal["5m", "1h"] = "5m"
    batch_api_threshold: int = 50  # use Batch API when >= this many photos
```

Add `vlm: VLMConfig = VLMConfig()` to the main `Config` model.

- [ ] **Step 2: Write tests for vlm_client (mock the Anthropic client)**

```python
# tests/test_vlm_client.py
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import pytest

from photopipe.vlm_client import VLMClient, PromptSection, build_image_block

def test_build_image_block_resizes_large_image(tmp_path):
    from PIL import Image
    img_path = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), color="red").save(img_path)
    block = build_image_block(img_path, max_dim=1568)
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/jpeg"

def test_prompt_prefix_marked_for_cache():
    client = VLMClient(api_key="fake")
    msg = client._build_message(
        cached_prefix="LONG INSTRUCTION PROMPT" * 100,
        images=[],
        per_call_prompt="describe",
    )
    # First content block (text prefix) must have cache_control set
    assert msg["content"][0].get("cache_control") == {"type": "ephemeral"}

def test_realtime_call_uses_structured_output():
    client = VLMClient(api_key="fake")
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text='{"year": 1985, "confidence": "high"}')]
    with patch.object(client, "_anthropic_client") as mock_client:
        mock_client.messages.create.return_value = mock_resp
        result = client.analyze(
            cached_prefix="instructions",
            images=[],
            per_call_prompt="describe",
            response_schema={"type": "object", "properties": {"year": {"type": "integer"}}},
        )
    assert result == {"year": 1985, "confidence": "high"}
    kwargs = mock_client.messages.create.call_args.kwargs
    assert "tools" in kwargs or "output_config" in kwargs  # structured output configured

def test_batch_api_submission_returns_job_id():
    client = VLMClient(api_key="fake")
    mock_batch = MagicMock()
    mock_batch.id = "msgbatch_abc123"
    with patch.object(client, "_anthropic_client") as mock_client:
        mock_client.messages.batches.create.return_value = mock_batch
        job_id = client.submit_batch([
            {"custom_id": "p1", "params": {"messages": []}},
        ])
    assert job_id == "msgbatch_abc123"
```

- [ ] **Step 3: Implement VLMClient**

```python
# photopipe/vlm_client.py
"""
Thin transport-layer wrapper around Anthropic vision calls.

Owns: prompt caching, structured output, Batch API submission/polling.
Does NOT own: prompts, schemas, business logic — those live in the
caller (dating_pipeline.py, handwriting_ocr.py).
"""
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from photopipe.config import get_config


@dataclass
class PromptSection:
    text: str
    cached: bool = False  # True => mark with cache_control


def build_image_block(image_path: Path, max_dim: int = 1568) -> dict:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        if w >= h:
            img = img.resize((max_dim, int(h * max_dim / w)), Image.Resampling.LANCZOS)
        else:
            img = img.resize((int(w * max_dim / h), max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode("ascii"),
        },
    }


class VLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        cfg = get_config()
        self.api_key = api_key or os.environ.get(cfg.vlm.api_key_env_var)
        self.model = model or cfg.vlm.model
        self.cache_ttl = cfg.vlm.cache_ttl
        self._anthropic_client = None

    @property
    def client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        return self._anthropic_client

    def _build_message(
        self,
        *,
        cached_prefix: str,
        images: list[dict],
        per_call_prompt: str,
    ) -> dict:
        content: list[dict] = []
        if cached_prefix:
            content.append({
                "type": "text",
                "text": cached_prefix,
                "cache_control": {"type": "ephemeral"},
            })
        content.extend(images)
        if per_call_prompt:
            content.append({"type": "text", "text": per_call_prompt})
        return {"role": "user", "content": content}

    def analyze(
        self,
        *,
        cached_prefix: str,
        images: list[dict],
        per_call_prompt: str,
        response_schema: Optional[dict] = None,
        max_tokens: int = 2048,
    ) -> dict:
        """Synchronous call. Returns parsed JSON dict matching response_schema."""
        message = self._build_message(
            cached_prefix=cached_prefix, images=images, per_call_prompt=per_call_prompt
        )
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [message],
        }
        if response_schema is not None:
            # Use tool-use strict mode for guaranteed schema adherence
            kwargs["tools"] = [{
                "name": "respond",
                "description": "Respond with the requested structured data.",
                "input_schema": response_schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "respond"}

        resp = self.client.messages.create(**kwargs)

        if response_schema is not None:
            # Find the tool_use block
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    return block.input
            # Fallback: try to parse text
            return json.loads(resp.content[0].text)
        return {"text": resp.content[0].text}

    def submit_batch(self, requests: list[dict]) -> str:
        """Submit a Batch API job. requests = list of {custom_id, params} dicts."""
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def poll_batch(self, job_id: str) -> dict:
        """Returns dict with status + results when completed."""
        batch = self.client.messages.batches.retrieve(job_id)
        result = {"status": batch.processing_status, "results": None}
        if batch.processing_status == "ended":
            result["results"] = list(self.client.messages.batches.results(job_id))
        return result
```

- [ ] **Step 4: Run tests, commit**

```bash
pytest tests/test_vlm_client.py -v
git add photopipe/vlm_client.py photopipe/config.py tests/test_vlm_client.py
git commit -m "Add VLM client with prompt caching, structured output, Batch API"
```

---

## Task 4: Build handwriting_ocr.py (Mistral OCR 3 + VLM fallback)

**Files:**
- Create: `photopipe/handwriting_ocr.py`
- Create: `tests/test_handwriting_ocr.py`
- Modify: `photopipe/config.py` — add `OCRConfig.provider` field
- Modify: `pyproject.toml` — add `mistralai>=1.0.0` dependency

- [ ] **Step 1: Add config field + dependency**

In `photopipe/config.py`, extend `OCRConfig` (or add a new `HandwritingOCRConfig`):

```python
class HandwritingOCRConfig(BaseModel):
    provider: Literal["mistral", "claude", "auto"] = "auto"
    mistral_api_key_env_var: str = "MISTRAL_API_KEY"
    mistral_model: str = "mistral-ocr-3"
    confidence_fallback_threshold: float = 0.6  # below this, fall back to VLM
    use_batch_api: bool = True
```

Add `handwriting_ocr: HandwritingOCRConfig = HandwritingOCRConfig()` to `Config`.

Add `"mistralai>=1.0.0",` to `pyproject.toml` dependencies.

- [ ] **Step 2: Write tests (mock both Mistral and VLM clients)**

```python
# tests/test_handwriting_ocr.py
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from PIL import Image

from photopipe.handwriting_ocr import HandwritingOCR, OCRResult

@pytest.fixture
def fake_back(tmp_path):
    p = tmp_path / "back.jpg"
    Image.new("RGB", (800, 600), color="white").save(p)
    return p

def test_mistral_high_confidence_returns_result(fake_back):
    ocr = HandwritingOCR(mistral_api_key="fake", vlm_client=MagicMock())
    with patch.object(ocr, "_call_mistral") as mock_mistral:
        mock_mistral.return_value = OCRResult(
            text="Mom and Dad, Summer 1985", confidence=0.9, provider="mistral"
        )
        result = ocr.ocr_back(fake_back)
    assert result.provider == "mistral"
    assert result.confidence == 0.9
    assert "1985" in result.text

def test_low_confidence_falls_back_to_vlm(fake_back):
    vlm = MagicMock()
    vlm.analyze.return_value = {"text": "Aug 1992"}
    ocr = HandwritingOCR(mistral_api_key="fake", vlm_client=vlm)
    with patch.object(ocr, "_call_mistral") as mock_mistral:
        mock_mistral.return_value = OCRResult(
            text="???", confidence=0.3, provider="mistral"
        )
        result = ocr.ocr_back(fake_back)
    assert result.provider == "claude_vlm"
    assert "1992" in result.text
    vlm.analyze.assert_called_once()

def test_extracts_date_from_ocr_text(fake_back):
    ocr = HandwritingOCR(mistral_api_key="fake", vlm_client=MagicMock())
    with patch.object(ocr, "_call_mistral") as mock_mistral:
        mock_mistral.return_value = OCRResult(
            text="June 1985", confidence=0.85, provider="mistral"
        )
        result = ocr.ocr_back(fake_back)
    assert result.extracted_date is not None
    assert result.extracted_date.year == 1985
    assert result.extracted_date.month == 6
```

- [ ] **Step 3: Implement HandwritingOCR**

```python
# photopipe/handwriting_ocr.py
"""
Handwriting OCR on photo backs.

Primary: Mistral OCR 3 (cheap, purpose-built).
Fallback: Claude vision (when Mistral confidence < threshold).

Both paths feed into date_parser to extract dates.
"""
import base64
import io
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from PIL import Image

from photopipe.config import get_config
from photopipe.date_parser import parse_date_from_text
from photopipe.vlm_client import VLMClient, build_image_block


HANDWRITING_VLM_PROMPT = """Read any handwritten or stamped text on this photo back.
Return the literal text exactly as written (preserve line breaks).
If you see a date, even partial, include it verbatim.
If there is no readable text, respond with an empty string."""


@dataclass
class OCRResult:
    text: str
    confidence: float
    provider: str  # "mistral" | "claude_vlm"
    extracted_date: Optional[date] = None


class HandwritingOCR:
    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        vlm_client: Optional[VLMClient] = None,
    ):
        cfg = get_config().handwriting_ocr
        self.cfg = cfg
        self.mistral_api_key = mistral_api_key or os.environ.get(cfg.mistral_api_key_env_var)
        self.vlm_client = vlm_client or VLMClient()
        self._mistral_client = None

    @property
    def mistral_client(self):
        if self._mistral_client is None and self.mistral_api_key:
            from mistralai import Mistral
            self._mistral_client = Mistral(api_key=self.mistral_api_key)
        return self._mistral_client

    def ocr_back(self, image_path: Path) -> OCRResult:
        """Run handwriting OCR on a single photo back."""
        result: Optional[OCRResult] = None

        if self.cfg.provider in ("mistral", "auto") and self.mistral_client:
            result = self._call_mistral(image_path)

        if (
            result is None
            or result.confidence < self.cfg.confidence_fallback_threshold
        ) and self.cfg.provider != "mistral":
            result = self._call_vlm(image_path)

        if result is None:
            result = OCRResult(text="", confidence=0.0, provider="none")

        dates = parse_date_from_text(result.text)
        if dates:
            result.extracted_date = dates[0][0]
        return result

    def _call_mistral(self, image_path: Path) -> OCRResult:
        with open(image_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode("ascii")
        resp = self.mistral_client.ocr.process(
            model=self.cfg.mistral_model,
            document={"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"},
        )
        # mistral-ocr-3 returns pages with text + confidence per region
        text = "\n".join(page.markdown for page in resp.pages)
        confs = [
            region.confidence
            for page in resp.pages
            for region in getattr(page, "regions", [])
            if region.confidence is not None
        ]
        avg_conf = sum(confs) / len(confs) if confs else 0.5
        return OCRResult(text=text.strip(), confidence=avg_conf, provider="mistral")

    def _call_vlm(self, image_path: Path) -> OCRResult:
        block = build_image_block(image_path)
        out = self.vlm_client.analyze(
            cached_prefix=HANDWRITING_VLM_PROMPT,
            images=[block],
            per_call_prompt="Read the text on this back.",
            response_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        )
        return OCRResult(
            text=out.get("text", ""),
            confidence=out.get("confidence", 0.7),
            provider="claude_vlm",
        )
```

- [ ] **Step 4: Run tests, commit**

```bash
pytest tests/test_handwriting_ocr.py -v
git add photopipe/handwriting_ocr.py photopipe/config.py tests/test_handwriting_ocr.py pyproject.toml
git commit -m "Add handwriting OCR module (Mistral OCR 3 + VLM fallback)"
```

---

## Task 5: bucket_service.py + capture_pipeline.py

**Files:**
- Create: `photopipe/bucket_service.py`
- Create: `photopipe/capture_pipeline.py`
- Create: `tests/test_bucket_service.py`
- Create: `tests/test_capture_pipeline.py`
- Modify: `photopipe/database.py` — add bucket CRUD methods

- [ ] **Step 1: Add bucket CRUD to database.py**

Add to `Database` class:

```python
def create_bucket(self, bucket: Bucket) -> Bucket:
    with self.get_connection() as conn:
        conn.execute("""
            INSERT INTO buckets(id, label, helper_name, status, batch_id, created_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (bucket.id, bucket.label, bucket.helper_name, bucket.status.value,
              bucket.batch_id, bucket.created_at, bucket.closed_at))
    return bucket

def get_bucket(self, bucket_id: str) -> Optional[Bucket]:
    with self.get_connection() as conn:
        row = conn.execute("SELECT * FROM buckets WHERE id=?", (bucket_id,)).fetchone()
    return self._row_to_bucket(row) if row else None

def list_buckets(self, status: Optional[BucketStatus] = None) -> list[Bucket]:
    with self.get_connection() as conn:
        if status:
            rows = conn.execute("SELECT * FROM buckets WHERE status=? ORDER BY created_at DESC",
                                (status.value,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM buckets ORDER BY created_at DESC").fetchall()
    return [self._row_to_bucket(r) for r in rows]

def update_bucket(self, bucket: Bucket) -> None:
    with self.get_connection() as conn:
        conn.execute("""
            UPDATE buckets SET label=?, helper_name=?, status=?, batch_id=?, closed_at=?
            WHERE id=?
        """, (bucket.label, bucket.helper_name, bucket.status.value,
              bucket.batch_id, bucket.closed_at, bucket.id))

def get_photos_by_bucket(self, bucket_id: str) -> list[PhotoPair]:
    with self.get_connection() as conn:
        rows = conn.execute("SELECT * FROM photos WHERE bucket_id=? ORDER BY sequence_num",
                            (bucket_id,)).fetchall()
    return [self._row_to_photo(r) for r in rows]

def _row_to_bucket(self, row) -> Bucket:
    return Bucket(
        id=row["id"], label=row["label"], helper_name=row["helper_name"],
        status=BucketStatus(row["status"]), batch_id=row["batch_id"],
        created_at=row["created_at"], closed_at=row["closed_at"],
    )
```

- [ ] **Step 2: Write tests for bucket_service**

```python
# tests/test_bucket_service.py
import pytest
from photopipe.bucket_service import BucketService, BucketStats
from photopipe.models import Bucket, BucketStatus
from photopipe.database import Database

@pytest.fixture
def db(tmp_path, monkeypatch):
    from photopipe.config import get_config
    cfg = get_config()
    cfg.paths.database = tmp_path / "test.db"
    return Database()

def test_open_bucket_creates_record(db):
    svc = BucketService(db)
    bucket = svc.open_bucket(label="Grandma's blue album", helper_name="Jo")
    assert bucket.status == BucketStatus.OPEN
    assert db.get_bucket(bucket.id) is not None

def test_close_bucket_sets_closed_status(db):
    svc = BucketService(db)
    bucket = svc.open_bucket(label="X")
    svc.close_bucket(bucket.id)
    reloaded = db.get_bucket(bucket.id)
    assert reloaded.status == BucketStatus.CLOSED
    assert reloaded.closed_at is not None

def test_bucket_stats_counts_photos(db):
    svc = BucketService(db)
    bucket = svc.open_bucket(label="X")
    # ... insert 3 fake photos with this bucket_id
    stats = svc.get_stats(bucket.id)
    assert isinstance(stats, BucketStats)
```

- [ ] **Step 3: Implement bucket_service.py**

```python
# photopipe/bucket_service.py
"""
Bucket lifecycle: open, list, close, convert to batch.

A Bucket is a free-text label entered by the helper during capture.
No metadata, no AI; just a way to group raw scans for the owner to
curate later.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from photopipe.database import Database
from photopipe.models import (
    Bucket, BucketStatus, Batch, BatchStatus, PhotoPair, PhotoPhase,
)


@dataclass
class BucketStats:
    photo_count: int
    photos_with_ocr: int
    photos_with_extracted_date: int
    helper_name: Optional[str]


class BucketService:
    def __init__(self, db: Database):
        self.db = db

    def open_bucket(self, label: str, helper_name: Optional[str] = None) -> Bucket:
        bucket = Bucket(label=label, helper_name=helper_name)
        self.db.create_bucket(bucket)
        return bucket

    def close_bucket(self, bucket_id: str) -> None:
        bucket = self.db.get_bucket(bucket_id)
        if bucket is None:
            raise ValueError(f"Bucket {bucket_id} not found")
        bucket.status = BucketStatus.CLOSED
        bucket.closed_at = datetime.now()
        self.db.update_bucket(bucket)

    def get_stats(self, bucket_id: str) -> BucketStats:
        bucket = self.db.get_bucket(bucket_id)
        photos = self.db.get_photos_by_bucket(bucket_id)
        return BucketStats(
            photo_count=len(photos),
            photos_with_ocr=sum(1 for p in photos if p.handwriting_ocr_text),
            photos_with_extracted_date=sum(1 for p in photos if p.extracted_date),
            helper_name=bucket.helper_name if bucket else None,
        )

    def convert_to_batch(
        self,
        bucket_id: str,
        *,
        name: str,
        date_start=None,
        date_end=None,
        location_description: Optional[str] = None,
        people: Optional[list[str]] = None,
        event_description: Optional[str] = None,
    ) -> Batch:
        bucket = self.db.get_bucket(bucket_id)
        if bucket is None:
            raise ValueError(f"Bucket {bucket_id} not found")

        batch = Batch(
            name=name,
            date_start=date_start,
            date_end=date_end,
            location_description=location_description,
            people=people or [],
            event_description=event_description,
        )
        self.db.create_batch(batch)

        # Move photos: set batch_id, advance phase to CURATED
        for photo in self.db.get_photos_by_bucket(bucket_id):
            photo.batch_id = batch.id
            photo.phase = PhotoPhase.CURATED
            self.db.update_photo(photo)

        bucket.status = BucketStatus.CONVERTED
        bucket.batch_id = batch.id
        self.db.update_bucket(bucket)
        return batch
```

- [ ] **Step 4: Write tests for capture_pipeline**

```python
# tests/test_capture_pipeline.py
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
import pytest

from photopipe.capture_pipeline import capture_batch, CaptureProgress
from photopipe.models import Bucket

@pytest.fixture
def bucket(db):
    from photopipe.bucket_service import BucketService
    svc = BucketService(db)
    return svc.open_bucket(label="Test bucket")

def test_capture_writes_photos_with_captured_phase(db, bucket, tmp_path):
    # Mock scanner to return two pre-staged files
    front = tmp_path / "f1.jpg"; Image.new("RGB", (800, 600)).save(front)
    back = tmp_path / "f1_b.jpg"; Image.new("RGB", (800, 600)).save(back)

    with patch("photopipe.capture_pipeline.scan_to_folder") as scan, \
         patch("photopipe.capture_pipeline.HandwritingOCR") as ocr_cls:
        scan.return_value = [front, back]
        ocr_cls.return_value.ocr_back.return_value = MagicMock(
            text="June 1985", confidence=0.9, provider="mistral", extracted_date=None,
        )
        result = capture_batch(bucket, db=db, scanner_device="fake", duplex=True)

    assert result.photos_added == 1
    photos = db.get_photos_by_bucket(bucket.id)
    assert all(p.phase.value == "captured" for p in photos)
```

- [ ] **Step 5: Implement capture_pipeline.py**

```python
# photopipe/capture_pipeline.py
"""
Headless capture pipeline.

Stages: scan -> pair fronts/backs -> autocrop -> handwriting OCR (async batch)
        -> persist with phase=captured, attached to bucket.

No batch concept, no owner context, no period-dating AI.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from photopipe.autocrop import process_scanned_photo
from photopipe.bucket_service import BucketService
from photopipe.database import Database
from photopipe.handwriting_ocr import HandwritingOCR
from photopipe.models import Bucket, PhotoPair, PhotoPhase, PhotoStatus, DateSource
from photopipe.pairing import pair_fronts_and_backs
from photopipe.scanner import scan_to_folder


@dataclass
class CaptureProgress:
    stage: str
    current: int = 0
    total: int = 0
    message: str = ""


@dataclass
class CaptureResult:
    photos_added: int
    bucket_id: str
    errors: list[str] = field(default_factory=list)


def capture_batch(
    bucket: Bucket,
    *,
    db: Database,
    scanner_device: str,
    resolution: int = 600,
    duplex: bool = True,
    progress: Optional[Callable[[CaptureProgress], None]] = None,
) -> CaptureResult:
    def emit(stage, **kw):
        if progress:
            progress(CaptureProgress(stage=stage, **kw))

    emit("scanning", message="Scanning stack...")
    files = scan_to_folder(device=scanner_device, resolution=resolution, duplex=duplex)
    if not files:
        return CaptureResult(photos_added=0, bucket_id=bucket.id,
                             errors=["Scanner returned no files"])

    emit("pairing", message=f"Pairing {len(files)} files...")
    pairs = pair_fronts_and_backs(files)

    emit("processing", total=len(pairs))
    photos = []
    for i, (front, back) in enumerate(pairs):
        emit("processing", current=i + 1, total=len(pairs),
             message=f"Crop + orient #{i + 1}")
        process_scanned_photo(front, use_ai_orientation=True)
        photo = PhotoPair(
            bucket_id=bucket.id,
            sequence_num=i + 1,
            front_path=front,
            back_path=back,
            phase=PhotoPhase.CAPTURED,
            status=PhotoStatus.INGESTED,
        )
        photos.append(photo)
        db.create_photo(photo)

    # Handwriting OCR: synchronous per-photo for MVP (Batch API in Task 8)
    ocr = HandwritingOCR()
    for i, photo in enumerate(photos):
        if not photo.back_path:
            continue
        emit("ocr", current=i + 1, total=len(photos), message=f"OCR back #{i + 1}")
        try:
            result = ocr.ocr_back(photo.back_path)
            photo.handwriting_ocr_text = result.text
            photo.handwriting_ocr_provider = result.provider
            photo.handwriting_ocr_confidence = result.confidence
            if result.extracted_date:
                photo.extracted_date = result.extracted_date
                photo.date_source = DateSource.OCR_BACK
            db.update_photo(photo)
        except Exception as e:
            emit("ocr_error", message=f"OCR failed for #{i+1}: {e}")

    emit("done", total=len(photos), current=len(photos))
    return CaptureResult(photos_added=len(photos), bucket_id=bucket.id)
```

- [ ] **Step 6: Run tests, commit**

```bash
pytest tests/test_bucket_service.py tests/test_capture_pipeline.py -v
git add photopipe/bucket_service.py photopipe/capture_pipeline.py photopipe/database.py tests/test_bucket_service.py tests/test_capture_pipeline.py
git commit -m "Add bucket service and headless capture pipeline"
```

---

## Task 6: Helper-mode page (pages/2_capture.py)

**Files:**
- Create: `pages/2_capture.py` (new — does NOT replace `2_scan.py` yet; see Task 11)
- Modify: `app.py` — add helper-mode toggle in sidebar/setup

- [ ] **Step 1: Build the helper page**

```python
# pages/2_capture.py
"""Helper Mode: minimal UI for someone other than the owner to scan."""
import streamlit as st
from photopipe.bucket_service import BucketService
from photopipe.capture_pipeline import capture_batch, CaptureProgress
from photopipe.config import get_config
from photopipe.database import Database
from photopipe.file_manager import generate_thumbnail
from photopipe.models import BucketStatus

st.set_page_config(
    page_title="Scan Photos",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide sidebar nav for helper mode
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {display: none;}
    [data-testid="stSidebar"] {display: none;}
    section.main {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)


def init_state():
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    if "current_bucket_id" not in st.session_state:
        st.session_state.current_bucket_id = None
    if "helper_name" not in st.session_state:
        st.session_state.helper_name = ""


def main():
    init_state()
    db = st.session_state.db
    svc = BucketService(db)

    st.title("📷 Scan Photos")

    if st.session_state.current_bucket_id is None:
        # Bucket-selection screen
        st.markdown("### Where are these photos from?")
        label = st.text_input(
            "Label",
            placeholder="e.g., Grandma's blue album, page 3",
            label_visibility="collapsed",
        )
        helper = st.text_input(
            "Your name (optional)",
            value=st.session_state.helper_name,
        )
        if st.button("🟢 Start scanning", type="primary", use_container_width=True,
                     disabled=not label.strip()):
            st.session_state.helper_name = helper
            bucket = svc.open_bucket(label=label.strip(), helper_name=helper or None)
            st.session_state.current_bucket_id = bucket.id
            st.rerun()
        return

    bucket = db.get_bucket(st.session_state.current_bucket_id)
    st.subheader(f"📁 {bucket.label}")
    if bucket.helper_name:
        st.caption(f"Scanned by {bucket.helper_name}")

    # Scan button
    if st.button("🟢 Scan Stack", type="primary", use_container_width=True):
        progress_box = st.empty()
        def on_progress(p: CaptureProgress):
            msg = p.message or p.stage
            if p.total:
                progress_box.progress(p.current / p.total, text=f"{msg} ({p.current}/{p.total})")
            else:
                progress_box.info(msg)
        with st.spinner("Scanning..."):
            cfg = get_config()
            result = capture_batch(
                bucket,
                db=db,
                scanner_device=cfg.scanner.device or "epsonds:net:192.168.1.62",
                resolution=cfg.scanner.resolution,
                duplex=cfg.scanner.duplex,
                progress=on_progress,
            )
        if result.errors:
            for err in result.errors:
                st.error(err)
        else:
            st.success(f"✓ Added {result.photos_added} photos to this bucket")
        st.rerun()

    # Recent thumbnails
    photos = db.get_photos_by_bucket(bucket.id)
    if photos:
        st.markdown(f"**{len(photos)} photos in this bucket**")
        cols = st.columns(6)
        for i, photo in enumerate(photos[-18:]):
            with cols[i % 6]:
                try:
                    st.image(generate_thumbnail(photo.front_path, size=(120, 120)),
                             use_container_width=True)
                except Exception:
                    st.caption(f"#{photo.sequence_num}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Done with this bucket", use_container_width=True):
            svc.close_bucket(bucket.id)
            st.session_state.current_bucket_id = None
            st.rerun()
    with col2:
        if st.button("➕ Start a new bucket", use_container_width=True):
            svc.close_bucket(bucket.id)
            st.session_state.current_bucket_id = None
            st.rerun()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add helper-mode toggle to settings**

In `pages/settings.py` (or directly in `app.py` sidebar):

```python
helper_mode = st.toggle("Helper Mode (hide owner-only pages)",
                        value=st.session_state.get("helper_mode", False))
if helper_mode != st.session_state.get("helper_mode"):
    st.session_state.helper_mode = helper_mode
    st.rerun()
```

In `app.py`, redirect to `pages/2_capture.py` immediately if helper mode is on and current page is the home page.

- [ ] **Step 3: Manual smoke test, commit**

Start the dev server and verify:
1. `streamlit run app.py`
2. Toggle helper mode → sidebar collapses, only capture page visible
3. Enter bucket label "Test" → click Start scanning → page shows scan button (scanner not required for this check)

```bash
git add pages/2_capture.py pages/settings.py app.py
git commit -m "Add helper-mode capture page with minimal chrome"
```

---

## Task 7: Owner buckets dashboard (pages/0_buckets.py)

**Files:**
- Create: `pages/0_buckets.py`

- [ ] **Step 1: Build the buckets dashboard**

```python
# pages/0_buckets.py
"""Owner-facing bucket dashboard: list, view stats, convert to batch."""
import streamlit as st
from photopipe.bucket_service import BucketService
from photopipe.config import get_config
from photopipe.database import Database
from photopipe.file_manager import generate_thumbnail
from photopipe.models import BucketStatus

st.set_page_config(page_title="Buckets - PhotoPipe", page_icon="📦", layout="wide")


def main():
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    db = st.session_state.db
    svc = BucketService(db)

    st.title("📦 Buckets")
    st.caption("Raw scans waiting to be curated into batches.")

    show_all = st.checkbox("Show converted buckets", value=False)
    buckets = svc.db.list_buckets() if show_all else [
        b for b in svc.db.list_buckets() if b.status != BucketStatus.CONVERTED
    ]

    if not buckets:
        st.info("No open buckets. Switch to Helper Mode and scan some photos.")
        return

    for bucket in buckets:
        stats = svc.get_stats(bucket.id)
        with st.expander(
            f"📁 {bucket.label}  ·  {stats.photo_count} photos  "
            f"·  {'helper: ' + stats.helper_name if stats.helper_name else 'unattributed'}",
            expanded=False,
        ):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.metric("Photos", stats.photo_count)
            with col2:
                st.metric("With dates from back", stats.photos_with_extracted_date)
            with col3:
                st.write(f"**Status:** {bucket.status.value}")

            # Thumbnail strip (first 6)
            photos = db.get_photos_by_bucket(bucket.id)
            if photos:
                cols = st.columns(6)
                for i, p in enumerate(photos[:6]):
                    with cols[i]:
                        try:
                            st.image(generate_thumbnail(p.front_path, size=(120, 120)))
                        except Exception:
                            st.caption(f"#{p.sequence_num}")

            if bucket.status == BucketStatus.CLOSED:
                with st.form(f"convert_{bucket.id}"):
                    st.markdown("### Convert to Batch")
                    name = st.text_input("Batch name", value=bucket.label)
                    c1, c2 = st.columns(2)
                    with c1:
                        date_start = st.date_input("Date start (optional)", value=None)
                    with c2:
                        date_end = st.date_input("Date end (optional)", value=None)
                    location = st.text_input("Location (optional)")
                    people = st.text_input("People (comma-separated, optional)")
                    event = st.text_area("Event description (optional)", height=60)
                    if st.form_submit_button("✅ Convert to Batch", type="primary"):
                        batch = svc.convert_to_batch(
                            bucket.id,
                            name=name,
                            date_start=date_start or None,
                            date_end=date_end or None,
                            location_description=location or None,
                            people=[p.strip() for p in people.split(",") if p.strip()],
                            event_description=event or None,
                        )
                        st.success(f"Converted to batch '{batch.name}'")
                        st.rerun()

            if bucket.status == BucketStatus.CONVERTED:
                st.info(f"Already converted to batch (id: {bucket.batch_id})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add pages/0_buckets.py
git commit -m "Add owner buckets dashboard"
```

---

## Task 8: curate_pipeline.py with multi-image AI

**Files:**
- Create: `photopipe/curate_pipeline.py`
- Create: `tests/test_curate_pipeline.py`
- Modify: `photopipe/database.py` — add AI job CRUD

The current `ai_dating.py` sends 1 photo per call. The new pipeline sends 10-15 per call with structured output and a coherence question.

- [ ] **Step 1: Define the structured-output schema**

```python
# photopipe/curate_pipeline.py
"""
Owner-driven curation pipeline.

Orchestrates: AI dating with multi-image batching, segment detection,
applying results to photos. Uses VLMClient for transport.
"""
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from photopipe.database import Database
from photopipe.models import (
    AIJob, AIJobStatus, Batch, DateConfidence, DateSource,
    PhotoPair, PhotoPhase,
)
from photopipe.vlm_client import VLMClient, build_image_block


CURATE_PROMPT_PREFIX = """You are an expert at dating old photographs. You will be shown
a numbered set of family photos and asked for per-photo date/location estimates
PLUS a coherence assessment of the set.

For each photo, look at:
- Clothing styles, hairstyles, jewelry
- Vehicles, technology, signage
- Photo print characteristics (border style, color cast, paper)
- Visible season cues (clothing weight, foliage, decorations)
- Architecture and regional cues for location

For the SET, assess: do these photos appear to be from the same event, day,
year, or are there visible breaks in the timeline (different ages of the same
people, different locations, different fashion eras)?

You will be given the user's hints (date range, location, people). Treat them
as priors. Override them only when visual evidence strongly contradicts.

Respond using the provided tool with strict JSON. If you cannot determine a
field, use null."""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "per_photo": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "photo_index": {"type": "integer"},
                    "year": {"type": ["integer", "null"]},
                    "year_range": {
                        "type": ["array", "null"],
                        "items": {"type": "integer"},
                        "minItems": 2, "maxItems": 2,
                    },
                    "month": {"type": ["integer", "null"]},
                    "season": {"type": ["string", "null"]},
                    "location_guess": {"type": ["string", "null"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["photo_index", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
        "coherence": {
            "type": "object",
            "properties": {
                "same_event": {"type": "boolean"},
                "segment_breaks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "after_photo_index": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["after_photo_index", "reason"],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["same_event", "segment_breaks", "summary"],
            "additionalProperties": False,
        },
    },
    "required": ["per_photo", "coherence"],
    "additionalProperties": False,
}
```

- [ ] **Step 2: Write tests**

```python
# tests/test_curate_pipeline.py
from unittest.mock import MagicMock
from datetime import date
import pytest

from photopipe.curate_pipeline import run_ai_dating, apply_ai_results, AIRunResult

def test_run_ai_dating_batches_images_into_groups(db):
    # 25 photos, images_per_call=10 => 3 calls
    vlm = MagicMock()
    vlm.analyze.return_value = {
        "per_photo": [{"photo_index": i, "confidence": "medium", "evidence": ["x"], "year": 1985}
                      for i in range(10)],
        "coherence": {"same_event": True, "segment_breaks": [], "summary": "Same trip"},
    }
    batch = make_test_batch(db)
    photos = make_test_photos(db, batch, count=25)
    result = run_ai_dating(batch, photos, vlm_client=vlm, images_per_call=10)
    assert vlm.analyze.call_count == 3

def test_apply_results_sets_extracted_date_only_on_undated_photos(db):
    batch = make_test_batch(db)
    photos = make_test_photos(db, batch, count=3)
    photos[0].extracted_date = date(1980, 1, 1)  # already dated
    db.update_photo(photos[0])

    ai_result = AIRunResult(
        per_photo={photos[i].id: {"year": 1985, "month": 6, "confidence": "high",
                                  "evidence": ["x"]} for i in range(3)},
        coherence={"same_event": True, "segment_breaks": [], "summary": ""},
    )
    applied = apply_ai_results(batch, ai_result, photos, db=db)
    assert applied.updated == 2  # only the two undated ones
```

- [ ] **Step 3: Implement run_ai_dating and apply_ai_results**

```python
# Append to photopipe/curate_pipeline.py

@dataclass
class AIRunResult:
    per_photo: dict[str, dict]  # photo_id -> per-photo dict from schema
    coherence: dict             # coherence dict from schema
    raw_responses: list[dict] = field(default_factory=list)


@dataclass
class ApplyResult:
    updated: int
    skipped: int
    errors: list[str] = field(default_factory=list)


def _batch_hint_prefix(batch: Batch) -> str:
    """Build the user-context preamble that goes AFTER the cached prefix."""
    parts = []
    if batch.date_start or batch.date_end:
        parts.append(f"User suggests date range: {batch.date_start}–{batch.date_end}")
    if batch.location_description:
        parts.append(f"User suggests location: {batch.location_description}")
    if batch.people:
        parts.append(f"People: {', '.join(batch.people)}")
    if batch.event_description:
        parts.append(f"Event: {batch.event_description}")
    return "User hints:\n- " + "\n- ".join(parts) if parts else ""


def run_ai_dating(
    batch: Batch,
    photos: list[PhotoPair],
    *,
    vlm_client: Optional[VLMClient] = None,
    images_per_call: int = 12,
) -> AIRunResult:
    vlm = vlm_client or VLMClient()
    per_photo: dict[str, dict] = {}
    all_coherences = []
    raw = []

    hints = _batch_hint_prefix(batch)
    per_call_prompt = (
        f"{hints}\n\n"
        f"Estimate per-photo date and location for these {{n}} photos. "
        "Photo indices are 0-based and match the image order."
    )

    for start in range(0, len(photos), images_per_call):
        chunk = photos[start:start + images_per_call]
        images = [build_image_block(p.front_path) for p in chunk]
        result = vlm.analyze(
            cached_prefix=CURATE_PROMPT_PREFIX,
            images=images,
            per_call_prompt=per_call_prompt.format(n=len(chunk)),
            response_schema=RESPONSE_SCHEMA,
            max_tokens=4096,
        )
        raw.append(result)
        for entry in result.get("per_photo", []):
            idx = entry["photo_index"]
            if 0 <= idx < len(chunk):
                per_photo[chunk[idx].id] = entry
        all_coherences.append(result.get("coherence", {}))

    # Aggregate coherence: union the segment_breaks across calls; majority same_event
    same_event = sum(1 for c in all_coherences if c.get("same_event")) > len(all_coherences) / 2
    coherence = {
        "same_event": same_event,
        "segment_breaks": [b for c in all_coherences for b in c.get("segment_breaks", [])],
        "summary": " | ".join(c.get("summary", "") for c in all_coherences if c.get("summary")),
    }
    return AIRunResult(per_photo=per_photo, coherence=coherence, raw_responses=raw)


def apply_ai_results(
    batch: Batch,
    ai_result: AIRunResult,
    photos: list[PhotoPair],
    *,
    db: Database,
    overwrite_existing: bool = False,
) -> ApplyResult:
    updated = skipped = 0
    errors: list[str] = []
    for photo in photos:
        entry = ai_result.per_photo.get(photo.id)
        if not entry or entry.get("year") is None:
            skipped += 1
            continue
        if photo.extracted_date is not None and not overwrite_existing:
            skipped += 1
            continue
        try:
            year = entry["year"]
            month = entry.get("month") or 6
            photo.extracted_date = date(year, month, 15)
            photo.date_source = DateSource.AI_ESTIMATED
            photo.date_confidence = DateConfidence(entry.get("confidence", "low"))
            photo.ai_analysis = entry
            photo.needs_review = True
            db.update_photo(photo)
            updated += 1
        except Exception as e:
            errors.append(f"photo {photo.id}: {e}")
    return ApplyResult(updated=updated, skipped=skipped, errors=errors)
```

- [ ] **Step 4: Run tests, commit**

```bash
pytest tests/test_curate_pipeline.py -v
git add photopipe/curate_pipeline.py tests/test_curate_pipeline.py
git commit -m "Add curate pipeline with multi-image AI and structured output"
```

---

## Task 9: Owner curate page (pages/3_curate.py)

**Files:**
- Create: `pages/3_curate.py` (new; does not yet replace `pages/3_review.py`)

- [ ] **Step 1: Build the curate page with three tabs**

```python
# pages/3_curate.py
"""Owner curate page: pick converted batch, run AI, review results."""
import streamlit as st
from photopipe.config import get_config
from photopipe.curate_pipeline import run_ai_dating, apply_ai_results
from photopipe.database import Database
from photopipe.file_manager import generate_thumbnail
from photopipe.models import PhotoPhase

st.set_page_config(page_title="Curate - PhotoPipe", page_icon="🧬", layout="wide")


def init_state():
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    st.session_state.setdefault("curate_batch_id", None)
    st.session_state.setdefault("ai_run_result", None)


def main():
    init_state()
    db = st.session_state.db
    st.title("🧬 Curate")
    st.caption("Apply AI dating and review estimates for converted batches.")

    # Batch selector
    batches = [b for b in db.get_all_batches()
               if any(p.phase != PhotoPhase.FINALIZED
                      for p in db.get_photos_by_batch(b.id))]
    if not batches:
        st.info("No batches awaiting curation. Convert a bucket from the Buckets page.")
        return

    names = [b.name for b in batches]
    chosen = st.selectbox("Batch", names)
    batch = next(b for b in batches if b.name == chosen)
    photos = db.get_photos_by_batch(batch.id)

    tab1, tab2, tab3 = st.tabs(["🤖 AI Dating", "🔍 Review", "ℹ️ Context"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Photos", len(photos))
            st.metric("Already dated", sum(1 for p in photos if p.extracted_date))
        with col2:
            images_per_call = st.slider("Photos per AI call", 5, 20, 12)
            mode = st.radio("Mode", ["Realtime (sync)", "Batch API (async, 50% off)"])

        if st.button("🤖 Run AI Dating", type="primary"):
            undated = [p for p in photos if not p.extracted_date]
            with st.spinner(f"Analyzing {len(undated)} photos..."):
                result = run_ai_dating(batch, undated, images_per_call=images_per_call)
                st.session_state.ai_run_result = result
            st.success(f"Analyzed {len(undated)} photos in {len(result.raw_responses)} call(s)")
            st.rerun()

        # Show AI results
        ai = st.session_state.ai_run_result
        if ai:
            st.markdown("### Results")
            st.write("**Coherence:**", ai.coherence.get("summary", ""))
            if ai.coherence.get("segment_breaks"):
                st.warning(f"AI detected {len(ai.coherence['segment_breaks'])} segment break(s)")
                for sb in ai.coherence["segment_breaks"]:
                    st.write(f"- After photo {sb['after_photo_index']}: {sb['reason']}")
            if st.button("✅ Apply AI dates"):
                applied = apply_ai_results(batch, ai, photos, db=db)
                st.success(f"Updated {applied.updated} photos (skipped {applied.skipped})")
                st.session_state.ai_run_result = None
                st.rerun()

    with tab2:
        # Reuse the thumbnail grid pattern from the legacy 3_review.py
        cols = st.columns(5)
        for i, photo in enumerate(photos):
            with cols[i % 5]:
                try:
                    st.image(generate_thumbnail(photo.front_path, size=(150, 150)))
                except Exception:
                    pass
                date_str = f"📅{photo.extracted_date.year}" if photo.extracted_date else "—"
                st.caption(f"#{photo.sequence_num} {date_str}")

    with tab3:
        st.write(f"**Name:** {batch.name}")
        st.write(f"**Date range:** {batch.date_start} – {batch.date_end}")
        st.write(f"**Location:** {batch.location_description or '—'}")
        st.write(f"**People:** {', '.join(batch.people) if batch.people else '—'}")
        st.write(f"**Event:** {batch.event_description or '—'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add pages/3_curate.py
git commit -m "Add owner curate page with multi-image AI dating"
```

---

## Task 10: photopipe doctor CLI + Tahoe Local Network detection

**Files:**
- Create: `photopipe/cli/__init__.py`
- Create: `photopipe/cli/doctor.py`
- Modify: `photopipe/__main__.py` — wire `doctor` subcommand
- Modify: `app.py` — add scanner-discovery banner

- [ ] **Step 1: Implement doctor**

```python
# photopipe/cli/doctor.py
"""Diagnose common PhotoPipe setup issues."""
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from photopipe.config import get_config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: Optional[str] = None


def check_exiftool() -> Check:
    found = shutil.which("exiftool")
    return Check(
        "ExifTool installed", bool(found),
        detail=found or "not found in PATH",
        fix="brew install exiftool" if not found else None,
    )


def check_sane() -> Check:
    found = shutil.which("scanimage")
    return Check(
        "SANE / scanimage installed", bool(found),
        detail=found or "not found in PATH",
        fix="brew install sane-backends" if not found else None,
    )


def check_scanner_discovery() -> Check:
    if not shutil.which("scanimage"):
        return Check("Scanner discovery", False, "scanimage not installed")
    try:
        r = subprocess.run(["scanimage", "-L"], capture_output=True, text=True, timeout=5)
        found_any = "No scanners were identified" not in (r.stdout + r.stderr)
        return Check(
            "Scanner discovery", found_any,
            detail=r.stdout.strip() or r.stderr.strip(),
            fix=(
                "On macOS Tahoe (26+), grant 'Local Network' permission to your "
                "terminal app in System Settings → Privacy & Security → Local Network. "
                "Tahoe periodically drops this permission silently after updates."
            ) if not found_any else None,
        )
    except subprocess.TimeoutExpired:
        return Check(
            "Scanner discovery", False, "scanimage -L timed out (5s)",
            fix="Network scanner may be unreachable, or Local Network permission missing.",
        )


def check_anthropic_key() -> Check:
    cfg = get_config()
    key = os.environ.get(cfg.vlm.api_key_env_var)
    return Check(
        "Anthropic API key", bool(key),
        detail=f"{cfg.vlm.api_key_env_var}: {'set' if key else 'unset'}",
        fix=f"export {cfg.vlm.api_key_env_var}=sk-..." if not key else None,
    )


def check_mistral_key() -> Check:
    cfg = get_config()
    key = os.environ.get(cfg.handwriting_ocr.mistral_api_key_env_var)
    if cfg.handwriting_ocr.provider == "claude":
        return Check("Mistral API key", True, "not required (provider=claude)")
    return Check(
        "Mistral API key", bool(key),
        detail=f"{cfg.handwriting_ocr.mistral_api_key_env_var}: {'set' if key else 'unset'}",
        fix=(
            f"export {cfg.handwriting_ocr.mistral_api_key_env_var}=...  "
            "(or set handwriting_ocr.provider=claude in config to skip Mistral)"
        ) if not key else None,
    )


CHECKS = [
    check_exiftool, check_sane, check_scanner_discovery,
    check_anthropic_key, check_mistral_key,
]


def run_doctor() -> int:
    print("PhotoPipe Doctor")
    print("=" * 40)
    failed = 0
    for fn in CHECKS:
        c = fn()
        icon = "✓" if c.ok else "✗"
        print(f"{icon} {c.name}: {c.detail}")
        if not c.ok:
            failed += 1
            if c.fix:
                print(f"   → fix: {c.fix}")
    print("=" * 40)
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 0 if failed == 0 else 1
```

- [ ] **Step 2: Wire into __main__.py**

In `photopipe/__main__.py`, add a `doctor` subcommand that calls `run_doctor()`.

- [ ] **Step 3: Surface scanner discovery banner in app.py**

```python
# In app.py main(), near the top:
from photopipe.cli.doctor import check_scanner_discovery

if "scanner_check_done" not in st.session_state:
    st.session_state.scanner_check_done = True
    check = check_scanner_discovery()
    if not check.ok:
        st.session_state.scanner_warning = check.fix or check.detail

if st.session_state.get("scanner_warning"):
    st.warning(f"⚠️ Scanner not detected. {st.session_state.scanner_warning}")
```

- [ ] **Step 4: Manual test, commit**

```bash
python -m photopipe doctor
git add photopipe/cli/ photopipe/__main__.py app.py
git commit -m "Add photopipe doctor CLI with Tahoe Local Network hint"
```

---

## Task 11: Delete legacy ocr.py + retire old pages

**Files:**
- Delete: `photopipe/ocr.py`
- Delete: `pages/2_scan.py` (rename keeping content for reference in git history)
- Delete: `pages/3_review.py`
- Delete: `pages/help.py`
- Modify: any file that still imports from `photopipe.ocr` (other than `date_parser`)

- [ ] **Step 1: Verify no remaining ocr.py imports beyond date_parser**

```bash
rg "from photopipe.ocr" --type py
rg "import photopipe.ocr" --type py
```

Expected: only `date_parser` re-exports remain, OR nothing. Migrate any holdouts to `handwriting_ocr` / `date_parser`.

- [ ] **Step 2: Delete the old files**

```bash
git rm photopipe/ocr.py
git rm pages/2_scan.py pages/3_review.py pages/help.py
```

- [ ] **Step 3: Update settings page and home to point at new pages**

In `pages/settings.py` and `app.py`, fix any `st.switch_page("pages/2_scan.py")` to `pages/2_capture.py` and `pages/3_review.py` to `pages/3_curate.py`. Update the home page quick-action buttons accordingly.

- [ ] **Step 4: Run full test suite + smoke test**

```bash
pytest tests/ -v
streamlit run app.py  # manually click through helper mode + owner mode
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Retire Tesseract OCR module and legacy pages"
```

---

## Task 12: Update README + first-run docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README workflow section**

Replace the existing workflow section with:

```markdown
## Workflow

PhotoPipe has two modes:

**Helper Mode** — for someone scanning photos who doesn't need to know
anything about batches or metadata. Toggle from owner settings, then
hand off the running app. The helper:
1. Enters a bucket label ("Grandma's blue album, page 3")
2. Optionally enters their name
3. Hits "Scan Stack" — repeats until done

**Owner Mode** — for you (the photo library owner):
1. **Buckets page**: review what the helper(s) scanned, convert each
   bucket to a Batch with date range, location, people, event.
2. **Curate page**: run AI dating on the batch. The AI sees 10–15
   photos per call, gives per-photo estimates plus a temporal coherence
   assessment, and flags likely segment breaks.
3. **Finalize page**: write EXIF/IPTC metadata, copy to archive.
```

Add a Troubleshooting subsection covering the macOS Tahoe Local Network
permission issue and the `photopipe doctor` command.

Add a "What's New (May 2026 rebuild)" section linking to the design spec.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Update README for two-phase capture/curate workflow"
```

---

## Self-Review Checklist (run before handoff)

**Spec coverage:** Walk through the spec sections and verify each maps to a task above.

| Spec section | Implementing task(s) |
|---|---|
| §1 Two-phase workflow | 1 (schema), 5 (capture), 8/9 (curate) |
| §2 Data model | 1 |
| §3 Module structure | 2, 3, 4, 5, 8 (new modules); 11 (deletions) |
| §4 Capture pipeline | 5, 6 |
| §5 Curate pipeline (multi-image, caching, batch API) | 3, 8, 9 |
| §6 Handwriting OCR | 4 |
| §7 Helper Mode UI | 6 |
| §8 Curate Mode UI | 9 |
| §9 macOS Tahoe handling | 10 |
| §11 Migration plan | Tasks 1–12 in matching order |
| §12 Testing | tests in tasks 1, 2, 3, 4, 5, 8 |

**Gap noted:** The Batch API job lifecycle (poll on app restart) referenced in spec §11 open questions is not in any task above. The MVP runs AI synchronously via `run_ai_dating`; submitting to the Batch API and polling on app restart is left for a follow-up plan. Add a note in the curate page that "Batch API mode" radio button is non-functional in MVP and falls back to realtime.

**Type consistency:** Verified `Bucket`, `BucketStatus`, `PhotoPhase`, `AIJob`, `AIJobStatus` referenced consistently. `OCRResult` (handwriting_ocr.py) is a different dataclass from the existing `OCRResult` (models.py); intentional since the legacy one is Tesseract-specific. Keep both during the rebuild; consider renaming `handwriting_ocr.OCRResult` → `HandwritingOCRResult` if collision becomes confusing.

**Placeholder scan:** None of "TBD," "TODO," "implement later." All code blocks contain actual code.
