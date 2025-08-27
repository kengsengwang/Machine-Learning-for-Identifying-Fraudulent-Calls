#!/usr/bin/env bash
# fix_github_actions.sh — make CI green by using Python 3.11 and platform markers

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ROOT/archive_${TS}"
mkdir -p "$ARCHIVE" ".github/workflows"

echo "• Repo: $ROOT"
echo "• Archive: $ARCHIVE"

# 1) Add platform markers for Windows-only deps in requirements.txt
if [[ -f "$ROOT/requirements.txt" ]]; then
  cp "$ROOT/requirements.txt" "$ARCHIVE/requirements.txt.bak"
  awk '
    BEGIN{IGNORECASE=1}
    # leave comments/blank lines untouched
    /^[[:space:]]*($|#)/ {print; next}
    # normalize whitespace
    {gsub(/\r/,"")}
    # add platform markers to pywin32/pywinpty if they are bare
    /^pywin32([[:space:]]*$|[<>=!~ ]).*/ && $0 !~ /platform_system/ {
      print $0 " ; platform_system==\"Windows\""; next
    }
    /^pywinpty([[:space:]]*$|[<>=!~ ]).*/ && $0 !~ /platform_system/ {
      print $0 " ; platform_system==\"Windows\""; next
    }
    {print}
  ' "$ROOT/requirements.txt" > "$ROOT/requirements.txt.tmp"
  mv -f "$ROOT/requirements.txt.tmp" "$ROOT/requirements.txt"
  echo "• requirements.txt updated with platform markers (backup at archive_${TS})"
fi

# 2) Disable/remove the old auto-sync workflow (if present)
for f in "$ROOT/.github/workflows/"*.yml "$ROOT/.github/workflows/"*.yaml; do
  [[ -e "$f" ]] || continue
  if grep -qiE 'Automated Git Sync|auto_git_sync' "$f"; then
    mv -f "$f" "$ARCHIVE/$(basename "$f").disabled"
    echo "• archived old workflow: $(basename "$f")"
  fi
done

# 3) Create a clean CI workflow using Python 3.11
cat > "$ROOT/.github/workflows/ci.yml" <<'YAML'
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Cache pip
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install deps
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Smoke test
      run: |
        python - <<'PY'
        import sys
        print("Python:", sys.version)
        try:
            import pandas, sklearn
            print("Imported pandas", pandas.__version__, "sklearn", sklearn.__version__)
        except Exception as e:
            raise SystemExit("Import check failed: " + str(e))
        PY
YAML

echo "• wrote .github/workflows/ci.yml (Python 3.11)"

# 4) Optional: ensure .gitattributes to avoid CRLF noise
if [[ ! -f "$ROOT/.gitattributes" ]]; then
  cat > "$ROOT/.gitattributes" <<'EOF'
*.sh text eol=lf
*.py text eol=lf
*.md text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.txt text eol=lf
*.ipynb -text
EOF
  echo "• added .gitattributes"
fi

echo "• Done. Now commit & push:"
echo "    git add -A"
echo "    git commit -m 'ci: fix workflows (py311, platform markers, archive old auto sync)'"
echo "    git push origin main"
