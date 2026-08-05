#!/bin/bash
# Double-click this to start PhotoPipe (opens in Terminal and stays running).
#
# WHY TERMINAL: macOS gates local-network access (needed to reach the network
# scanner) per app. Homebrew's `scanimage` is unsigned, so it inherits the
# permission of whatever launched it. Terminal can be granted that permission —
# a background LaunchAgent or ad-hoc-signed .app cannot, which is why the app
# previously reported "Scanner not detected" while the scanner was fine.
#
# The FIRST time you run this, macOS may ask to let Terminal find devices on
# your local network — say YES, or the scanner won't be reachable.
cd "$(dirname "$0")"
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"
LOG="$HOME/Library/Logs/photopipe"
mkdir -p "$LOG"

echo "Starting PhotoPipe…"

# Already running? Just open the browser.
if curl -s --max-time 3 http://localhost:8501/_stcore/health >/dev/null 2>&1; then
  echo "PhotoPipe is already running."
  open "http://localhost:8501/"
  echo "Leave this window open (or close it — the server keeps running)."
  exit 0
fi

# Keep it alive: restart the server if it ever dies, until this window is closed.
while true; do
  if ! curl -s --max-time 3 http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    echo "$(date '+%H:%M:%S') starting server…"
    ./.venv/bin/streamlit run app.py >> "$LOG/server.out.log" 2>> "$LOG/server.err.log" &
    # Wait for it to answer, then open the UI (first launch only).
    for _ in $(seq 1 40); do
      if curl -s --max-time 2 http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        [ -n "${OPENED:-}" ] || { open "http://localhost:8501/"; OPENED=1; }
        echo "PhotoPipe is running at http://localhost:8501/"
        echo "Keep this window open. Press Ctrl-C to stop PhotoPipe."
        break
      fi
      sleep 0.5
    done
  fi
  sleep 20
done
