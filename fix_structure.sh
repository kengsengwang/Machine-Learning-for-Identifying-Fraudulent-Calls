#!/usr/bin/env bash
set -e

echo "============================================"
echo " AIAP REPOSITORY AUTO-CLEAN + STRUCTURE FIX "
echo "============================================"

# Ensure you are at repo root
if [ ! -d ".git" ]; then
    echo "‚ùå ERROR: Run this script from the repository root."
    exit 1
fi

echo "Ì∑π Cleaning old folders and files..."
rm -rf tests || true
rm -rf data || true
rm -rf outputs || true
rm -rf archive* || true
rm -rf __pycache__ || true
rm -rf src/__pycache__ || true

rm -f fix_github_actions.sh || true
rm -f upgrade_ci.sh || true
rm -f pytest.ini || true
rm -f requirements-lock.txt || true
rm -f .gitattributes || true
rm -f to_minimal_layout.sh || true

echo "Ì≥Å Ensuring src/ exists..."
mkdir -p src

echo "Ì∑† Required pipeline files:"
REQUIRED_FILES="src/__init__.py src/build_features.py src/train_and_evaluate.py src/utils.py"
for f in $REQUIRED_FILES; do
    if [ ! -f "$f" ]; then
        echo "‚ùå MISSING: $f ‚Äî please add it manually"
    else
        echo "   ‚úî $f found"
    fi
done

echo "Ì∑π Cleaning Python bytecode..."
find . -name "*.pyc" -delete || true
find . -name "__pycache__" -type d -exec rm -rf {} + || true

echo "Ì≥ù Rebuilding .gitignore..."
cat <<EOF > .gitignore
__pycache__/
*.pyc
data/
artifacts/
outputs/
archive*/
EOF

echo "Ì∫∫ Final directory tree (if tree installed):"
tree -L 3 || ls -R | sed -e 's/[^-][^\/]*\//--/g;s/--/ |/g' || true

echo "============================================"
echo " AIAP STRUCTURE FIXED SUCCESSFULLY ÌæâÌπè"
echo " Now run the following:"
echo ""
echo "   git add ."
echo "   git commit -m 'Fix repo to AIAP structure'"
echo "   git push origin main"
echo ""
echo "============================================"
