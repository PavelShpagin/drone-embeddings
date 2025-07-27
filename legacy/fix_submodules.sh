#!/bin/bash
set -e

for dir in third_party/GLVL third_party/pytorch-superpoint; do
  if [ -d "$dir/.git" ]; then
    echo "Removing nested git repo in $dir"
    rm -rf "$dir/.git"
  fi
  # Remove submodule references if they exist
  git submodule deinit -f "$dir" 2>/dev/null || true
  git rm -f "$dir" 2>/dev/null || true
  rm -rf ".git/modules/$dir"
  # Re-add the directory as a regular folder
  git add "$dir"
done

git commit -m "Convert GLVL and pytorch-superpoint to regular directories tracked by main repo"
echo "Done! Both directories are now regular folders tracked by your main git repository."
