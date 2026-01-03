# PhotoPipe

**Photo Scanning Metadata Pipeline for Epson FastFoto FF-680W**

PhotoPipe is a tool for managing scanned photo metadata, OCR date extraction, and AI-assisted date estimation for digitizing family photos.

## Features

- **Batch Management**: Organize photos into batches with dates, locations, and people tags
- **Front/Back Pairing**: Automatically pair front and back scans from duplex scanning
- **OCR Date Extraction**: Extract dates from photo backs using Tesseract OCR
- **AI Date Estimation**: Use Claude Vision to estimate dates from visual clues
- **Metadata Writing**: Embed EXIF/IPTC/XMP metadata using ExifTool
- **Web GUI**: Simple Streamlit interface for batch management and review
- **CLI**: Command-line interface for automation

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
brew install tesseract exiftool

# Linux (apt)
sudo apt install tesseract-ocr libimage-exiftool-perl
```

### Running PhotoPipe

```bash
# Activate virtual environment
source .venv/bin/activate

# (Optional) Set API key for AI dating
export ANTHROPIC_API_KEY='your-key-here'

# Launch web interface
streamlit run app.py

# Or use CLI
photopipe --help
```

## Workflow

### 1. Create a Batch

Batches group photos with common metadata:

```bash
photopipe batch create \
  --name "Summer_1985" \
  --date-start 1985-06-01 \
  --date-end 1985-08-31 \
  --location "Toledo, OH" \
  --people "Grandma Rose, Mom, Dad"
```

Or use the web interface at **Batch Setup**.

### 2. Scan Photos

Configure your Epson FastFoto to save scans to:
- **Mac**: `~/Pictures/Scanner_Input`

The scanner creates files like:
- `IMG_0001.jpg` (front)
- `IMG_0001_back.jpg` (back)

### 3. Ingest & Process

```bash
photopipe batch process --name "Summer_1985"
```

This will:
1. Pair front/back images
2. Run OCR on photo backs
3. Extract dates from text
4. Flag low-confidence results for review

### 4. Review

Use the web interface **Review Queue** to:
- View extracted metadata
- Approve or edit dates
- Flag photos for later

### 5. Finalize

```bash
photopipe batch finalize \
  --name "Summer_1985" \
  --auto-approve-high-confidence
```

This will:
1. Copy originals to archive
2. Write metadata to files (EXIF/IPTC/XMP)
3. Rename and organize output
4. Generate batch report

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

ocr:
  language: eng
  confidence_threshold: 70

ai_dating:
  enabled: true
  model: claude-sonnet-4-20250514
  max_samples_per_batch: 3
```

## AI Dating

PhotoPipe can use Claude Vision to estimate dates for photos without OCR dates:

1. Set your API key: `export ANTHROPIC_API_KEY='sk-...'`
2. Enable in settings or use `--ai-dating` flag
3. The AI analyzes:
   - Clothing and fashion styles
   - Hairstyles
   - Visible technology (cars, TVs, phones)
   - Photo characteristics

## Date Extraction Priority

1. **OCR from back** (highest priority) - Photo lab stamps, handwritten dates
2. **OCR from front** - Orange date stamps from photo labs
3. **Batch default** - Spread across date range based on sequence
4. **AI estimation** - Claude Vision analysis

## CLI Reference

```bash
# Initialize
photopipe init

# Show status
photopipe status

# Batch commands
photopipe batch list
photopipe batch create --name NAME [options]
photopipe batch process --name NAME [--preview] [--ai-dating]
photopipe batch finalize --name NAME [--auto-approve-high-confidence]
photopipe batch delete --name NAME [--force]
```

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

### ExifTool not found
```bash
brew install exiftool
```

### Tesseract not found
```bash
brew install tesseract
```

### AI dating not working
- Check `ANTHROPIC_API_KEY` is set
- Verify API key has Claude API access
- Check usage limits on your Anthropic account

### OCR results are poor
- Ensure photos are in focus
- Try adjusting preprocessing settings
- Photo lab stamps work better than handwriting

## License

MIT License
