#!/bin/zsh
# Launcher: start the layered EEG collector GUI module with the project venv
# Double-click this file in Finder to run.

# cd to the folder containing this script (handles spaces in path)
cd "$(dirname "$0")" || exit 1

VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "[Error] .venv not found or python missing."
    echo "Recreate it with:"
    echo "  python3 -m venv .venv"
    echo "  .venv/bin/pip install -r requirements.lock.txt -r requirements-gui.txt"
    echo ""
    echo "Press any key to close..."
    read -k 1
    exit 1
fi

echo "Starting EEG collector GUI..."
"$VENV_PY" -m eeg_mi.gui
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo ""
    echo "[Exit] eeg_mi.gui exited with code $STATUS"
    echo "Press any key to close..."
    read -k 1
fi
