#!/usr/bin/env bash
set -euo pipefail

echo "======================================="
echo " Step 1: Build processed dataset"
echo "======================================="
python -m src.build_features

echo "======================================="
echo " Step 2: Train and evaluate models"
echo "======================================="
python -m src.train_and_evaluate
