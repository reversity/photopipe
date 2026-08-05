#!/bin/bash
# Build PhotoPipe.app — a real macOS app bundle that runs the PhotoPipe server.
#
# WHY A BUNDLE: macOS 15+ gates local-network access (needed to reach a network
# scanner) per *responsible process*. A bare LaunchAgent running an unsigned
# binary (Homebrew's `scanimage`) has no app identity, never gets prompted, and
# is silently denied — the scanner is unreachable from the server even though it
# works from Terminal. A signed .app bundle has a stable identity: macOS prompts
# once, the user approves, and the grant sticks (and is visible/toggleable in
# System Settings > Privacy & Security > Local Network).
#
# Usage: scripts/build_app_bundle.sh [install-dir]   (default: ~/Applications)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HOME/Applications}"
APP="$DEST/PhotoPipe.app"
MACOS="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"

VENV_PY="$REPO/.venv/bin/streamlit"
[[ -x "$VENV_PY" ]] || { echo "error: $VENV_PY not found — run install.sh first" >&2; exit 3; }

echo "Building $APP"
rm -rf "$APP"
mkdir -p "$MACOS" "$RES"

# The launcher: starts the server in the bundle's own process tree, so the
# Local Network grant macOS associates with PhotoPipe.app covers the scanner
# subprocesses (scanimage) it spawns.
cat > "$MACOS/PhotoPipe" <<LAUNCHER
#!/bin/bash
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"
cd "$REPO"
LOG="\$HOME/Library/Logs/photopipe"
mkdir -p "\$LOG"

# Touch the local network from the app's own process tree at launch. macOS
# only prompts for (and lists) the Local Network permission when an app
# actually attempts local-network access — without this the app never appears
# in System Settings, and the unsigned scanner binaries stay silently blocked.
"$REPO/.venv/bin/python" - <<'PROBE' >> "\$LOG/server.out.log" 2>&1 || true
import socket
from contextlib import suppress
host = None
with suppress(Exception):
    import yaml, pathlib
    cfg = yaml.safe_load(open(pathlib.Path.home() / ".photopipe" / "config.yaml"))
    host = (cfg.get("scanner") or {}).get("mdns_host")
with suppress(Exception):
    ip = socket.gethostbyname(host) if host else None
    if ip:
        s = socket.socket(); s.settimeout(3)
        with suppress(OSError):
            s.connect((ip, 1865))
        s.close()
PROBE

# Already running? Just focus the browser instead of starting a second server.
if curl -s --max-time 3 http://localhost:8501/_stcore/health >/dev/null 2>&1; then
  open "http://localhost:8501/"
  exit 0
fi

"$REPO/.venv/bin/streamlit" run "$REPO/app.py" >> "\$LOG/server.out.log" 2>> "\$LOG/server.err.log" &
SERVER_PID=\$!

# Wait for it to answer, then open the UI.
for i in \$(seq 1 40); do
  if curl -s --max-time 2 http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    open "http://localhost:8501/"
    break
  fi
  sleep 0.5
done

wait \$SERVER_PID
LAUNCHER
chmod +x "$MACOS/PhotoPipe"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>PhotoPipe</string>
    <key>CFBundleIdentifier</key><string>com.photopipe.app</string>
    <key>CFBundleName</key><string>PhotoPipe</string>
    <key>CFBundleDisplayName</key><string>PhotoPipe</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <!-- Must be a normal foreground app: a background-only (LSUIElement) app
         cannot display the macOS Local Network permission prompt, and without
         that grant the unsigned scanner binaries are silently refused. -->
    <!-- Explains the Local Network prompt the user will see once -->
    <key>NSLocalNetworkUsageDescription</key>
    <string>PhotoPipe needs to reach your photo scanner on this network.</string>
    <key>NSBonjourServices</key>
    <array>
        <string>_scanner._tcp</string>
        <string>_uscan._tcp</string>
    </array>
</dict>
</plist>
PLIST

# Ad-hoc sign so the bundle has a stable identity for TCC. Unsigned bundles get
# a new identity whenever they change, which loses the user's permission grant.
codesign --force --deep --sign - "$APP" 2>/dev/null \
  && echo "  signed (ad-hoc)" \
  || echo "  WARNING: codesign failed — permission may not persist across changes" >&2

echo "Built $APP"
echo
echo "Next:"
echo "  1. Open it once:  open '$APP'"
echo "  2. Approve the macOS 'Local Network' prompt (needed to reach the scanner)."
echo "  3. Optional autostart: System Settings > General > Login Items > add PhotoPipe."
