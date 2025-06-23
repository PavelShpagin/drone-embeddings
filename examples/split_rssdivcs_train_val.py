import os
import shutil
import random
from pathlib import Path

# Set the path to your RSSDIVCS dataset root (with images in subfolders)
RSSDIVCS_ROOT = "third_party/pytorch-superpoint/datasets/RSSDIVCS"
TRAIN_RATIO = 0.9  # 90% train, 10% val
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Create train/ and val/ subfolders, including parents if needed
train_dir = Path(RSSDIVCS_ROOT) / 'train'
val_dir = Path(RSSDIVCS_ROOT) / 'val'
train_dir.mkdir(exist_ok=True, parents=True)
val_dir.mkdir(exist_ok=True, parents=True)

# Recursively find all image files in all subfolders (excluding train/ and val/)
all_files = [f for f in Path(RSSDIVCS_ROOT).rglob('*')
             if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and f.parent.name not in ('train', 'val')]

random.shuffle(all_files)

num_train = int(len(all_files) * TRAIN_RATIO)
train_files = all_files[:num_train]
val_files = all_files[num_train:]

print(f"Total images: {len(all_files)}")
print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")

# Move files to train/ and val/ (flattening the structure)
for f in train_files:
    dest = train_dir / f.name
    if dest.exists():
        print(f"Warning: {dest} already exists. Skipping.")
        continue
    shutil.move(str(f), str(dest))
for f in val_files:
    dest = val_dir / f.name
    if dest.exists():
        print(f"Warning: {dest} already exists. Skipping.")
        continue
    shutil.move(str(f), str(dest))

print("Split complete.") 