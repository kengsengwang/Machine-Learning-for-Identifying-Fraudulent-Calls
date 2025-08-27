# 8) Move everything *not* required by the rubric out of root (to archive),
#    except required items and any archive_* folders
REQUIRED=( ".git" ".github" "src" "README.md" "eda.ipynb" "requirements.txt" "run.sh" ".gitignore" )
say "Archiving extraneous top-level items (keeping only rubric-required files/folders)"
shopt -s dotglob nullglob
for item in "$ROOT"/*; do
  name="$(basename "$item")"
  [[ "$name" == "." || "$name" == ".." ]] && continue

  # skip required
  skip=0
  for r in "${REQUIRED[@]}"; do
    [[ "$name" == "$r" ]] && skip=1 && break
  done

  # skip any archive_* folders (including the current one)
  [[ "$name" == archive_* ]] && skip=1

  # skip our helper scripts
  [[ "$name" == "enforce_structure.sh" || "$name" == "clean_project.sh" ]] && skip=1

  if [[ $skip -eq 0 ]]; then
    say "Archive: $name"
    run "mv -f \"$item\" \"$ARCHIVE/\""
  fi
done
shopt -u dotglob nullglob
