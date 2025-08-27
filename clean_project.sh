#!/usr/bin/env bash
# clean_project.sh — safe cleanup for "Machine-Learning-for-Identifying-Fraudulent-Calls"
# Default: dry-run (prints actions). Use --apply to actually delete.
# Optional: --delete-plots to delete PNG plots in root/ instead of archiving them.

set -Eeuo pipefail

APPLY=0
DELETE_PLOTS=0
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --delete-plots) DELETE_PLOTS=1 ;;
    *) echo "Unknown arg: $a"; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say(){ echo "• $*"; }
doit(){ if [[ $APPLY -eq 1 ]]; then eval "$1"; else echo "[dry-run] $1"; fi; }

# --- candidates to delete (only if they exist) ---
CANDIDATES=(
  "$ROOT/temp.txt"
  "$ROOT/auto_git_sync.py"
  "$ROOT/check_sync.py"
  "$ROOT/Machine-Learning-for-Identifying-Fraudulent-Calls.docx"
  "$ROOT/venv"
  "$ROOT/myvenv"
  "$ROOT/.ipynb_checkpoints"
  "$ROOT/root/.ipynb_checkpoints"
)

if [[ -f "$ROOT/eda.ipynb" && -f "$ROOT/root/eda.ipynb" ]]; then
  CANDIDATES+=("$ROOT/root/eda.ipynb")
fi
if [[ -f "$ROOT/requirements.txt" && -f "$ROOT/root/requirements.txt" ]]; then
  CANDIDATES+=("$ROOT/root/requirements.txt")
fi

say "Cleaning project at: $ROOT"
say "Mode: $([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)"

# 1) Remove files/folders
for path in "${CANDIDATES[@]}"; do
  [[ -e "$path" ]] || continue
  if [[ -d "$path" ]]; then
    doit "rm -rf \"$path\""
  else
    doit "rm -f \"$path\""
  fi
done

# 2) Recursively remove any stray .ipynb_checkpoints anywhere
while IFS= read -r d; do
  doit "rm -rf \"$d\""
done < <(find "$ROOT" -type d -name ".ipynb_checkpoints" 2>/dev/null || true)

# 3) Plot handling inside root/ (PNG images)
PLOT_DIR="$ROOT/root"
if [[ -d "$PLOT_DIR" ]]; then
  mapfile -t PLOTS < <(find "$PLOT_DIR" -maxdepth 1 -type f -name "*.png" 2>/dev/null || true)
  if (( ${#PLOTS[@]} > 0 )); then
    if [[ $DELETE_PLOTS -eq 1 ]]; then
      say "Deleting plots in root/:"
      for p in "${PLOTS[@]}"; do doit "rm -f \"$p\""; done
    else
      ARCHIVE="$ROOT/reports/figures"
      doit "mkdir -p \"$ARCHIVE\""
      say "Archiving plots from root/ -> reports/figures (use --delete-plots to delete instead):"
      for p in "${PLOTS[@]}"; do doit "mv -f \"$p\" \"$ARCHIVE/\""; done
    fi
  fi
fi

say "Done. Re-run with --apply to perform changes."
