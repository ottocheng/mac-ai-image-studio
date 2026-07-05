#!/bin/bash
# Double-click this file in Finder to start AI Image Studio.
# A Terminal window opens, the model server boots, and your browser
# launches at http://127.0.0.1:7860. Close the Terminal window to stop.

set -e

# Always operate from the directory this script lives in
cd "$(dirname "$0")"

# uv lives in ~/.local/bin on a default install; make sure that's on PATH
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found on PATH."
  echo "Install it from https://docs.astral.sh/uv/ and try again."
  echo "Press Return to close..."
  read
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "First run: installing dependencies (one-time, ~2 minutes)..."
  uv sync
fi

echo "------------------------------------------------------------"
echo " AI Image Studio"
echo " Browser will open at http://127.0.0.1:7860 automatically."
echo " Close this Terminal window to stop the server."
echo "------------------------------------------------------------"

# inbrowser=True in app.py opens the browser when the server is ready
exec uv run python webui/app.py
