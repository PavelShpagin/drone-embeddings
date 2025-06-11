#!/bin/bash

# This is a universal script to initialize Git LFS on any machine for this project.
# Run it once after cloning the repository on a new machine (local or server).

echo "--- Git LFS Initializer ---"

# --- Step 1: Install Git LFS if it's not present ---
# This command checks if 'git-lfs' is not in the system's command path.
if ! command -v git-lfs &> /dev/null
then
    echo "Git LFS command not found. Attempting to install..."
    # This assumes a Debian/Ubuntu-based system (like your VM and WSL).
    sudo apt-get update
    sudo apt-get install -y git-lfs
    echo "Git LFS installed successfully."
else
    echo "Git LFS is already installed."
fi

# --- Step 2: Initialize LFS for the user and repository ---
# 'git lfs install' configures your global Git config to use LFS. It's safe to run multiple times.
echo "Initializing Git LFS..."
git lfs install

# --- Step 3: Set which file types to track ---
# This command tells Git LFS to manage any file ending in '.pth'.
# It modifies the '.gitattributes' file. This is also safe to run multiple times.
echo "Ensuring .pth files are tracked by LFS..."
git lfs track "*.pth"

# --- Step 4: Add the configuration file to Git ---
# The '.gitattributes' file contains the tracking rules and MUST be part of the repository.
# This ensures anyone who clones the repo automatically uses LFS for the right files.
echo "Staging .gitattributes file..."
git add .gitattributes

echo "---"
echo "✅ LFS initialization complete."
echo "The .gitattributes file has been staged. Please commit and push it if it has changed."
echo "You can now use 'git add/commit/push/pull' as normal, and LFS will work automatically." 