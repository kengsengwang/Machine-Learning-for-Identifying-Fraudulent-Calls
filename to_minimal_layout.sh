#!/usr/bin/env bash
# to_minimal_layout.sh — enforce examiner’s minimal repo layout
# Default: DRY RUN (prints actions). Use --apply to perform.

set -Eeuo pipefail

APPLY=0
while (( "$#" )); do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --help)  echo "Usage: ./to_minimal_layout.sh [--apply]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ROOT/archive_$TS"

say(){ echo "• $*"; }
run(){ if [[ $APPLY -eq 1 ]]; then eval "$*"; else echo "[dry-run] $*"; fi; }

say "Repo: $ROOT"
say "Mode: $([[ $APPLY -eq 1 ]] && echo APPLY || echo DRY-RUN)"
run "mkdir -p \"$ARCHIVE\""

# 0) Ensure required dirs exist
for d in ".github" "src"; do
  [[ -d "$ROOT/$d" ]] || run "mkdir -p \"$ROOT/$d\""
done

# 1) Move any top-level *.py (except this script) into src/
shopt -s nullglob
for f in "$ROOT"/*.py; do
  base="$(basename "$f")"
  [[ "$base" == "to_minimal_layout.sh" ]] && continue
  say "Move $base -> src/"
  run "mv -f \"$f\" \"$ROOT/src/\""
done
shopt -u nullglob

# 2) If you had a scripts/ folder, move its .py into src/
if [[ -d "$ROOT/scripts" ]]; then
  while IFS= read -r p; do
    base="$(basename "$p")"
    say "Move scripts/$base -> src/"
    run "mv -f \"$p\" \"$ROOT/src/\""
  done < <(find "$ROOT/scripts" -type f -name "*.py" 2>/dev/null || true)
fi

# 3) Keep exactly one EDA notebook at root named eda.ipynb
if [[ ! -f "$ROOT/eda.ipynb" ]]; then
  CAND=""
  if [[ -f "$ROOT/root/eda.ipynb" ]]; then CAND="$ROOT/root/eda.ipynb"; fi
  if [[ -z "$CAND" ]]; then CAND="$(find "$ROOT" -iname 'eda*.ipynb' | head -n1 || true)"; fi
  if [[ -n "$CAND" ]]; then
    say "Promote $(basename "$CAND") -> ./eda.ipynb"
    run "mv -f \"$CAND\" \"$ROOT/eda.ipynb\""
  else
    say "NOTE: No eda.ipynb found; create one later if required."
  fi
else
  # archive any other *eda*.ipynb duplicates
  while IFS= read -r dup; do
    [[ "$dup" == "$ROOT/eda.ipynb" ]] && continue
    rel="${dup#$ROOT/}"
    say "Archive duplicate notebook: $rel"
    run "mv -f \"$dup\" \"$ARCHIVE/\""
  done < <(find "$ROOT" -iname 'eda*.ipynb' 2>/dev/null || true)
fi

# 4) Merge requirements to a single root requirements.txt
TMP="$ARCHIVE/_req_merged.txt"; : > "$TMP"
[[ -f "$ROOT/requirements.txt" ]] && cat "$ROOT/requirements.txt" >> "$TMP"
[[ -f "$ROOT/root/requirements.txt" ]] && cat "$ROOT/root/requirements.txt" >> "$TMP"
[[ -f "$ROOT/src/requirements.txt"  ]] && cat "$ROOT/src/requirements.txt"  >> "$TMP"
if [[ -s "$TMP" ]]; then
  say "Write merged requirements -> ./requirements.txt"
  if [[ $APPLY -eq 1 ]]; then
    awk 'NF && !seen[tolower($0)]++' "$TMP" | sort -f > "$ROOT/requirements.txt"
  else
    echo "[dry-run] would write merged requirements"
  fi
fi
# archive any extras
for f in "$ROOT/root/requirements.txt" "$ROOT/src/requirements.txt"; do
  [[ -f "$f" ]] && run "mv -f \"$f\" \"$ARCHIVE/\""
done

# 5) Ensure run.sh exists (simple runner)
if [[ ! -f "$ROOT/run.sh" ]]; then
  say "Create run.sh"
  if [[ $APPLY -eq 1 ]]; then
    cat > "$ROOT/run.sh" <<'EOSH'
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# Optional local venv
if [[ ! -d ".venv" ]]; then python -m venv .venv; fi
# shellcheck disable=SC1091
source ".venv/Scripts/activate" 2>/dev/null || source ".venv/bin/activate"
python -m pip install --upgrade pip
[[ -f requirements.txt ]] && pip install -r requirements.txt
if [[ -f "src/main.py" ]]; then python -u src/main.py; else echo "Add your entrypoint at src/main.py"; fi
EOSH
    chmod +x "$ROOT/run.sh"
  else
    echo "[dry-run] would create run.sh"
  fi
fi

# 6) Remove ipynb checkpoints everywhere
while IFS= read -r d; do
  say "Remove $d"
  run "rm -rf \"$d\""
done < <(find "$ROOT" -type d -name ".ipynb_checkpoints" 2>/dev/null || true)

# 7) Archive everything in root that is NOT one of the required items
REQUIRED=( ".git" ".github" "src" "README.md" "eda.ipynb" "requirements.txt" "run.sh" ".gitignore" "to_minimal_layout.sh" )
shopt -s dotglob nullglob
for item in "$ROOT"/*; do
  name="$(basename "$item")"
  [[ "$name" == "." || "$name" == ".." ]] && continue

  keep=0
  for r in "${REQUIRED[@]}"; do [[ "$name" == "$r" ]] && keep=1 && break; done
  [[ "$name" == archive_* ]] && keep=1   # never move archives into themselves

  if [[ $keep -eq 0 ]]; then
    say "Archive: $name"
    run "mv -f \"$item\" \"$ARCHIVE/\""
  fi
done
shopt -u dotglob nullglob

say "Done. Re-run with --apply to perform changes."
