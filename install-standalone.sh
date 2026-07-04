#!/bin/bash
# PhotoPipe Standalone Installation Script
# Run this on any Mac to install PhotoPipe

set -e

echo "=========================================="
echo "  PhotoPipe Standalone Installer"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is for macOS only${NC}"
    exit 1
fi

# Check for Homebrew
echo "Checking for Homebrew..."
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    if [[ -f "/opt/homebrew/bin/brew" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo -e "${GREEN}Homebrew found${NC}"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."

for pkg in exiftool sane-backends python@3.12; do
    if ! brew list $pkg &> /dev/null; then
        echo "Installing $pkg..."
        brew install $pkg
    else
        echo -e "${GREEN}$pkg already installed${NC}"
    fi
done

# Determine install location
INSTALL_DIR="${PHOTOPIPE_INSTALL_DIR:-$HOME/.photopipe-app}"
echo ""
echo "Installing PhotoPipe to: $INSTALL_DIR"

# Create install directory
mkdir -p "$INSTALL_DIR"

# Copy application files
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Copying application files..."
cp -r "$SCRIPT_DIR/photopipe" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/pages" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/app.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/config.yaml" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/pyproject.toml" "$INSTALL_DIR/"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
cd "$INSTALL_DIR"

PYTHON_CMD=""
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
    # brew prefix differs by architecture: /opt/homebrew (Apple Silicon) vs /usr/local (Intel)
    PYTHON_CMD="$(brew --prefix)/bin/python3.12"
fi

$PYTHON_CMD -m venv .venv
source .venv/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip -q
BREW_PREFIX="$(brew --prefix)"
CFLAGS="-I${BREW_PREFIX}/include" LDFLAGS="-L${BREW_PREFIX}/lib" pip install -e . -q

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p ~/Pictures/Scanner_Input
mkdir -p ~/Pictures/Scanned_Photos
mkdir -p ~/Pictures/Scanned_Photos/_archive
mkdir -p ~/.photopipe

# Copy default config if not exists
if [[ ! -f ~/.photopipe/config.yaml ]]; then
    cp config.yaml ~/.photopipe/config.yaml
fi

# Create launcher script
echo ""
echo "Creating launcher..."
cat > "$INSTALL_DIR/run.sh" << 'LAUNCHER'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run app.py --server.headless true
LAUNCHER
chmod +x "$INSTALL_DIR/run.sh"

# Create desktop shortcut (macOS app)
APP_DIR="$HOME/Applications/PhotoPipe.app"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cat > "$APP_DIR/Contents/MacOS/PhotoPipe" << APPSCRIPT
#!/bin/bash
cd "$INSTALL_DIR"
source .venv/bin/activate
open "http://localhost:8501"
streamlit run app.py --server.headless true
APPSCRIPT
chmod +x "$APP_DIR/Contents/MacOS/PhotoPipe"

cat > "$APP_DIR/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>PhotoPipe</string>
    <key>CFBundleIdentifier</key>
    <string>com.photopipe.app</string>
    <key>CFBundleName</key>
    <string>PhotoPipe</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
PLIST

echo ""
echo "=========================================="
echo -e "${GREEN}  Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "To run PhotoPipe:"
echo ""
echo "  Option 1: Double-click PhotoPipe in ~/Applications"
echo ""
echo "  Option 2: Run from terminal:"
echo "     $INSTALL_DIR/run.sh"
echo ""
echo "  Option 3: Manual:"
echo "     cd $INSTALL_DIR"
echo "     source .venv/bin/activate"
echo "     streamlit run app.py"
echo ""
echo "Scanner input:  ~/Pictures/Scanner_Input"
echo "Photo output:   ~/Pictures/Scanned_Photos"
echo ""
echo "Next steps:"
echo "  - Set API keys (in your shell or via the in-app setup wizard):"
echo "      export ANTHROPIC_API_KEY='sk-ant-...'  # AI dating + Claude vision"
echo "      export MISTRAL_API_KEY='...'           # primary handwriting OCR"
echo "  - First Faces-page run downloads the InsightFace buffalo_l model (~300 MB)"
echo "    into ~/.insightface/."
echo "  - Run 'photopipe doctor' inside the venv to diagnose env / scanner / keys."
echo ""
