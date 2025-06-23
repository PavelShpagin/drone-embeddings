import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# Input directories containing original large images
EARTH_IMAGERY_DIRS = [f"data/earth_imagery/loc{i}" for i in range(1, 11)]
# Output directory for generated crops
OUTPUT_CROPS_DIR = "superpoint_training/crops"

# Crop settings
CROP_SIZE = 256
CROP_STRIDE = 128  # 50% overlap for dense coverage

def generate_crops():
    print(f"Generating {CROP_SIZE}x{CROP_SIZE} crops...")
    Path(OUTPUT_CROPS_DIR).mkdir(parents=True, exist_ok=True)

    total_crops_generated = 0

    for loc_dir in EARTH_IMAGERY_DIRS:
        loc_path = Path(loc_dir)
        if not loc_path.exists():
            print(f"Warning: Location directory {loc_path} not found. Skipping.")
            continue

        image_files = list(loc_path.glob('*.jpg'))
        if not image_files:
            print(f"No .jpg images found in {loc_path}. Skipping.")
            continue

        for img_path in tqdm(image_files, desc=f"Processing {loc_path.name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                width, height = img.size

                crop_idx = 0
                # Iterate with stride to cover the entire image uniformly
                for y in range(0, height - CROP_SIZE + 1, CROP_STRIDE):
                    for x in range(0, width - CROP_SIZE + 1, CROP_STRIDE):
                        crop = img.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))
                        # Save crops with a unique name
                        crop_name = f"{img_path.stem}_crop{crop_idx:05d}.png"
                        crop_save_path = Path(OUTPUT_CROPS_DIR) / crop_name
                        crop.save(crop_save_path)
                        crop_idx += 1
                        total_crops_generated += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    print(f"Finished generating crops. Total crops: {total_crops_generated}")

if __name__ == '__main__':
    generate_crops() 