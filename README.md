# PhotoPipe

**Photo Scanning Metadata Pipeline for Epson FastFoto FF-680W**

PhotoPipe is a tool for managing scanned photo metadata, OCR date extraction, and AI-assisted date estimation for digitizing family photos.

## What's New (May 2026 rebuild)

PhotoPipe was rebuilt in May 2026 to support a **two-phase workflow**:

- **Capture phase** (Helper Mode): bare-bones scan UI for someone other
  than the owner — no batches, no metadata, just scan into a labeled bucket.
- **Curate phase** (Owner Mode): owner converts buckets to batches, adds
  context (date range, location, people, event), runs AI dating with
  multi-image coherence reasoning.

Other big changes:
- **Sonnet 4.6** with prompt caching + Batch API for AI dating
- **Mistral OCR 3** replaces Tesseract for handwriting on photo backs
  (fallback to Claude vision on low confidence)
- **Multi-image AI batching** (10–15 photos per call) for temporal
  coherence reasoning + segment detection
- **macOS Tahoe Local Network** handling via `photopipe doctor` + in-app banner

Design spec: `docs/superpowers/specs/2026-05-17-photopipe-rebuild-design.md`
Implementation plan: `docs/superpowers/plans/2026-05-17-photopipe-rebuild.md`

## Features

- **Two-Mode UI**: Helper Mode for scanning, Owner Mode for curation
- **Bucket → Batch Flow**: Helpers fill buckets, owner promotes to batches
- **Front/Back Pairing**: Automatically pair front and back scans from duplex scanning
- **Handwriting OCR**: Mistral OCR 3 reads dates off photo backs (Claude vision fallback)
- **Multi-Image AI Dating**: Claude Sonnet 4.6 reasons across 10–15 photos at a time for temporal coherence
- **Metadata Writing**: Embed EXIF/IPTC/XMP metadata using ExifTool
- **Web GUI**: Streamlit interface for capture, curation, and finalize
- **CLI**: `photopipe doctor` and batch automation commands

## Quick Start

### Installation

```bash
# Clone or copy the project
cd photopipe

# Run the install script (Mac)
./install.sh

# Or manually install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### System Dependencies

Install required system tools:

```bash
# macOS (Homebrew)
brew install exiftool sane-backends

# Linux (apt)
sudo apt install libimage-exiftool-perl sane-utils
```

> Tesseract is no longer required — handwriting OCR uses **Mistral OCR 3**
> (cloud API) with a Claude vision fallback.

### Running PhotoPipe

```bash
# Activate virtual environment
source .venv/bin/activate

# (Optional) Set API keys for AI dating + handwriting OCR
export ANTHROPIC_API_KEY='your-key-here'
export MISTRAL_API_KEY='your-key-here'   # for Mistral OCR 3 on photo backs

# Verify environment (scanner, ExifTool, API keys)
photopipe doctor

# Launch web interface
streamlit run app.py

# Or use CLI
photopipe --help
```

## Workflow

PhotoPipe has two modes:

**Helper Mode** — for someone scanning photos who doesn't need to know
anything about batches or metadata. Toggle from the owner sidebar, then
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

## Output Structure

```
~/Pictures/Scanned_Photos/
├── 1985/
│   └── 1985-06_Summer_1985/
│       ├── 1985-06-01_Summer_1985_0001_front.jpg
│       ├── 1985-06-01_Summer_1985_0001_back.jpg
│       └── _batch_report.json
├── _archive/
│   └── Summer_1985/
│       └── (original files)
└── _logs/
```

## Configuration

Configuration is stored in `~/.photopipe/config.yaml`:

```yaml
paths:
  input_folder: ~/Pictures/Scanner_Input
  output_folder: ~/Pictures/Scanned_Photos
  archive_folder: ~/Pictures/Scanned_Photos/_archive
  database: ~/.photopipe/photopipe.db

scanner:
  front_pattern: "IMG_{num}.jpg"
  back_pattern: "IMG_{num}_back.jpg"

handwriting_ocr:
  provider: mistral          # mistral | claude | none
  model: mistral-ocr-3
  confidence_threshold: 0.6  # below this, fall back to vlm
  api_key_env: MISTRAL_API_KEY

vlm:
  enabled: true
  model: claude-sonnet-4-6
  batch_size: 12             # photos per AI call (10–15 recommended)
  use_prompt_caching: true
  use_batch_api: false       # set true for cheaper async runs
  api_key_env: ANTHROPIC_API_KEY
```

## AI Dating

PhotoPipe uses **Claude Sonnet 4.6** with prompt caching to estimate dates
for photos without reliable OCR dates. Unlike the old one-photo-at-a-time
flow, the Curate page sends batches of 10–15 photos in a single call so
the model can reason about temporal coherence (same roll of film? same
event? where do segments break?).

1. Set your API key: `export ANTHROPIC_API_KEY='sk-...'`
2. Open the **Curate** page for a batch and click **Run AI dating**
3. The AI returns, per photo:
   - Estimated date range + confidence
   - Visual reasoning (clothing, hair, cars, photo finish, EXIF clues)
   - Segment break flags when the temporal signal shifts

For cheaper async runs across many batches, enable `vlm.use_batch_api`
in your config — PhotoPipe will submit via the Anthropic Batch API.

### Handwriting OCR (photo backs)

PhotoPipe uses **Mistral OCR 3** to read dates and notes off photo backs.
If Mistral's confidence is below `handwriting_ocr.confidence_threshold`,
the back is re-sent to Claude vision as a fallback. Set
`provider: none` to skip handwriting OCR entirely.

## Date Extraction Priority

1. **Handwriting OCR from back** (highest priority) — Mistral OCR 3 + vision fallback
2. **OCR from front** — Orange date stamps from photo labs
3. **Batch default** — Spread across date range based on sequence
4. **Multi-image AI estimation** — Claude Sonnet 4.6 coherence reasoning

## CLI Reference

```bash
# Initialize
photopipe init

# Show status
photopipe status

# Environment + scanner health check (Tahoe Local Network, ExifTool,
# SANE, API keys). See Troubleshooting.
photopipe doctor

# Batch commands
photopipe batch list
photopipe batch create --name NAME [options]
photopipe batch process --name NAME [--preview] [--ai-dating]
photopipe batch finalize --name NAME [--auto-approve-high-confidence]
photopipe batch delete --name NAME [--force]
```

> **Note:** `photopipe batch process` no longer runs Tesseract OCR — that
> module was retired in the May 2026 rebuild. Use the GUI Capture
> (Helper Mode) and Curate (Owner Mode) pages for handwriting OCR and AI
> dating. The batch CLI remains useful for headless re-runs of pairing
> and metadata writing.

## Metadata Written

PhotoPipe writes the following metadata fields:

**EXIF:**
- DateTimeOriginal
- CreateDate
- GPSLatitude/GPSLongitude

**IPTC:**
- Caption-Abstract
- Keywords
- City, Province-State, Country
- DateCreated
- CopyrightNotice

**XMP:**
- Description
- Subject
- DateCreated
- Custom: DateConfidence, DateSource, OCRText

## Portability

To move PhotoPipe to another Mac:

1. Copy the entire `photopipe` folder
2. Run `./install.sh`
3. Copy your `~/.photopipe` folder (optional, for database/config)

## Troubleshooting

### Scanner not detected (macOS Tahoe / Sequoia)

macOS 15+ requires apps to have **Local Network** permission to discover
network scanners. After macOS 26.3.1+ updates, this permission is sometimes
silently dropped from apps that previously had it.

Fix:
1. Open **System Settings → Privacy & Security → Local Network**
2. Find your terminal app (Terminal, iTerm, VS Code, etc.) and check it
3. If your terminal is missing from the list, run `scanimage -L` once to
   prompt macOS to add it.
4. Restart the app.

### Run `photopipe doctor`

The doctor CLI checks your environment and tells you what's wrong:

```bash
python -m photopipe doctor
```

It verifies: ExifTool installed, SANE/scanimage installed, scanner reachable,
Anthropic API key set, Mistral API key set (or provider configured to skip it).

### Other common issues

- **No batches showing on Curate page** — make sure you've converted at
  least one bucket to a batch from the Buckets page first.
- **Helper Mode toggle hides everything** — to exit, navigate to the home
  page URL (the toggle lives in the home page sidebar).
- **API key not detected** — `export ANTHROPIC_API_KEY=sk-...` in your
  shell before launching, OR put it in `~/.photopipe/config.yaml`.
- **ExifTool not found** — `brew install exiftool` (macOS) or
  `sudo apt install libimage-exiftool-perl` (Linux).
- **AI dating returns errors** — confirm your API key has Claude API access
  and that you haven't hit usage limits on your Anthropic account.

## License

MIT License
