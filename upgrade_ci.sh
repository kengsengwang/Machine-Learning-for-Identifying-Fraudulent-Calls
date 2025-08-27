#!/usr/bin/env bash
# upgrade_ci.sh — replace failing GH Action with a clean CI (py311 + lint + tests)

set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ROOT/archive_$TS"
mkdir -p "$ARCHIVE" ".github/workflows" "tests"

echo "• Repo: $ROOT"
echo "• Archive: $ARCHIVE"

# 0) Make requirements friendlier to Linux runners (idempotent)
if [[ -f "$ROOT/requirements.txt" ]]; then
  cp "$ROOT/requirements.txt" "$ARCHIVE/requirements.txt.bak"
  awk '
    BEGIN{IGNORECASE=1}
    /^[[:space:]]*($|#)/ {print; next}
    {gsub(/\r/,"")}
    /^pywin32([[:space:]]*$|[<>=!~ ]).*/ && $0 !~ /platform_system/ {print $0 " ; platform_system==\"Windows\""; next}
    /^pywinpty([[:space:]]*$|[<>=!~ ]).*/ && $0 !~ /platform_system/ {print $0 " ; platform_system==\"Windows\""; next}
    {print}
  ' "$ROOT/requirements.txt" > "$ROOT/requirements.txt.tmp"
  mv -f "$ROOT/requirements.txt.tmp" "$ROOT/requirements.txt"
  echo "• requirements.txt: added platform markers for Windows-only deps (backup in $(basename "$ARCHIVE"))"
fi

# 1) Archive any old "Automated Git Sync" workflows
for f in "$ROOT/.github/workflows/"*.yml "$ROOT/.github/workflows/"*.yaml; do
  [[ -e "$f" ]] || continue
  if grep -qiE 'Automated Git Sync|auto_git_sync' "$f"; then
    mv -f "$f" "$ARCHIVE/$(basename "$f").disabled"
    echo "• archived old workflow: $(basename "$f")"
  fi
done

# 2) Write a new CI workflow (py311 + cache + lint + tests)
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

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip setuptools wheel
        pip install -r requirements.txt
        pip install pytest flake8

    - name: Lint
      run: |
        flake8 .

    - name: Tests
      run: |
        pytest -q
YAML
echo "• wrote .github/workflows/ci.yml"

# 3) Add a lenient flake8 config (so style doesn’t fail)
cat > "$ROOT/.flake8" <<'EOF'
[flake8]
max-line-length = 120
extend-ignore = E203, W503
exclude = .git, .venv, venv, myenv, archive_*, __pycache__
EOF
echo "• wrote .flake8"

# 4) Add a minimal pytest config + a smoke test
cat > "$ROOT/pytest.ini" <<'EOF'
[pytest]
addopts = -q
pythonpath = .
testpaths = tests
EOF
echo "• wrote pytest.ini"

# don’t overwrite if user already has tests
if [[ ! -f "$ROOT/tests/test_imports.py" ]]; then
  cat > "$ROOT/tests/test_imports.py" <<'PY'
def test_imports():
    import importlib
    importlib.import_module("src.main")

def test_libs_available():
    import pandas, sklearn
    assert pandas.__version__
    assert sklearn.__version__
PY
  echo "• wrote tests/test_imports.py"
fi

# 5) Add .gitattributes if missing (stops CRLF noise on CI)
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
  echo "• wrote .gitattributes"
fi

echo "• Done. Now commit & push:"
echo "    git add -A"
echo "    git commit -m 'ci: py311 + lint + tests; archive old sync workflow'"
echo "    git push origin main"
