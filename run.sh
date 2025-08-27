#!/usr/bin/env bash
# run.sh — simple launcher for the project (Git Bash friendly)
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# create a local venv if not present (optional)
if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi

# activate venv
# shellcheck disable=SC1091
source ".venv/Scripts/activate" 2>/dev/null || source ".venv/bin/activate"

python -m pip install --upgrade pip
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
fi

# entrypoint — adjust if your main is different
if [[ -f "src/main.py" ]]; then
  python -u src/main.py
else
  echo "No src/main.py found. Add your pipeline entrypoint to src/ and re-run."
fi
