#!/usr/bin/env bash
set -e

echo "============================================"
echo " AIAP REPOSITORY AUTO-CLEAN + STRUCTURE FIX "
echo "============================================"

# Ensure you are at repo root
if [ ! -d ".git" ]; then
    echo "❌ ERROR: Run this script from the repository root."
    exit 1
fi

echo "🧹 Cleaning old folders and files..."
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

echo "📁 Ensuring src/ exists..."
mkdir -p src

echo "🧠 Required pipeline files:"
REQUIRED_FILES="src/__init__.py src/build_features.py src/train_and_evaluate.py src/utils.py"
for f in $REQUIRED_FILES; do
    if [ ! -f "$f" ]; then
        echo "❌ MISSING: $f — please add it manually"
    else
        echo "   ✔ $f found"
    fi
done

echo "🧹 Cleaning Python bytecode..."
find . -name "*.pyc" -delete || true
find . -name "__pycache__" -type d -exec rm -rf {} + || true

echo "📝 Ensuring .gitignore contains correct rules..."
cat <<EOF > .gitignore
__pycache__/
*.pyc
data/
artifacts/
outputs/
archive*/
EOF

echo "🪺 Final directory tree:"
tree -L 3 || ls -R | sed -e 's/[^-][^\/]*\//--/g;s/--/ |/g' || true

echo "============================================"
echo " AIAP STRUCTURE FIXED SUCCESSFULLY 🎉🙏"
echo " Now run the following:"
echo ""
echo "   git add ."
echo "   git commit -m 'Fix repo to AIAP structure'"
echo "   git push origin main"
echo ""
echo "============================================"
