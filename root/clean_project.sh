# 1) save file
nano clean_project_v2.sh
# paste, save (Ctrl+O, Enter), exit (Ctrl+X)

# 2) make executable
chmod +x clean_project_v2.sh

# 3) preview (no changes)
./clean_project_v2.sh

# 4) apply changes
./clean_project_v2.sh --apply

# 5) also auto-commit & push after cleanup
./clean_project_v2.sh --apply --git "chore: clean repo & archive temp/duplicates"

# 6) keep virtualenvs (don’t delete .venv/venv/myenv)
./clean_project_v2.sh --apply --keep-venv

# 7) delete PNG plots instead of archiving
./clean_project_v2.sh --apply --delete-plots
