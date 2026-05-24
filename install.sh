#!/bin/bash
# PhotoPipe Installation Script
# This script sets up PhotoPipe on a fresh Mac

set -e

echo "=========================================="
echo "  PhotoPipe Installation Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is designed for macOS${NC}"
    exit 1
fi

# Check for Homebrew
echo "Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ -f "/opt/homebrew/bin/brew" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo -e "${GREEN}Homebrew found${NC}"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."

# ExifTool
if ! command -v exiftool &> /dev/null; then
    echo "Installing ExifTool..."
    brew install exiftool
else
    echo -e "${GREEN}ExifTool already installed${NC}"
fi

# SANE backends (for scanner control)
if ! command -v scanimage &> /dev/null; then
    echo "Installing SANE backends (for scanner control)..."
    brew install sane-backends
else
    echo -e "${GREEN}SANE backends already installed${NC}"
fi

# Check for Python 3.11+
echo ""
echo "Checking Python version..."
PYTHON_CMD=""

# Try python3.12 first, then python3.11, then python3
for cmd in python3.12 python3.11 python3; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        if [[ $MAJOR -ge 3 ]] && [[ $MINOR -ge 11 ]]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo -e "${YELLOW}Python 3.11+ not found. Installing Python 3.12...${NC}"
    brew install python@3.12
    PYTHON_CMD="python3.12"
fi

echo -e "${GREEN}Using $PYTHON_CMD${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [[ -d ".venv" ]]; then
    echo -e "${YELLOW}Virtual environment already exists. Recreating...${NC}"
    rm -rf .venv
fi

$PYTHON_CMD -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package
echo ""
echo "Installing PhotoPipe and dependencies..."
pip install -e .

# Create default directories
echo ""
echo "Creating default directories..."
mkdir -p ~/Pictures/Scanner_Input
mkdir -p ~/Pictures/Scanned_Photos
mkdir -p ~/Pictures/Scanned_Photos/_archive
mkdir -p ~/.photopipe

# Copy default config if not exists
if [[ ! -f ~/.photopipe/config.yaml ]]; then
    echo "Creating default configuration..."
    cp config.yaml ~/.photopipe/config.yaml
fi

# Initialize database
echo ""
echo "Initializing database..."
python -m photopipe init

echo ""
echo "=========================================="
echo -e "${GREEN}  Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "To run PhotoPipe:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source $(pwd)/.venv/bin/activate"
echo ""
echo "  2. (Optional) Set API keys for AI features:"
echo "     export ANTHROPIC_API_KEY='sk-ant-...'    # AI dating + Claude vision OCR fallback"
echo "     export MISTRAL_API_KEY='...'             # primary handwriting OCR on photo backs"
echo "     (or store them in ~/.photopipe/settings.json — the in-app setup wizard can do this)"
echo ""
echo "  3. Launch the web interface:"
echo "     streamlit run app.py"
echo ""
echo "  Or use the CLI:"
echo "     photopipe --help"
echo "     photopipe doctor          # diagnose env / deps / API keys"
echo ""
echo "Notes:"
echo "  - First face-clustering run downloads the InsightFace buffalo_l model (~300 MB)"
echo "    into ~/.insightface/. The Faces page will show progress."
echo "  - macOS Tahoe (26+) may silently drop 'Local Network' permission after updates;"
echo "    if the scanner stops being detected, re-grant it in System Settings."
echo ""
echo "Scanner input folder: ~/Pictures/Scanner_Input"
echo "Output folder:        ~/Pictures/Scanned_Photos"
echo ""
