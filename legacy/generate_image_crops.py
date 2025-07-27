import os
import json
from PIL import Image
import math
from tqdm import tqdm
import random
import tyro
from dataclasses import dataclass
from typing import Optional

@dataclass
class CropArgs:
    num_target_crops: int = 1000
    crop_size_pixels: int = 210
    source_dir: str = "data/earth_imagery"
    output_base_dir: str = "data/cropped_loc_images"
    output_subdir: Optional[str] = None

def generate_image_crops(args: CropArgs):
    all_potential_crops = []
    pixel_to_meter_ratio = 100 / args.crop_size_pixels

    # Approximate meters per degree latitude/longitude at ~45 degrees latitude
    METERS_PER_DEG_LAT = 111000
    
    for i in range(1, 11): # loc1 to loc10
        loc_folder = os.path.join(args.source_dir, f"loc{i}")
        if not os.path.exists(loc_folder):
            print(f"Warning: Folder {loc_folder} not found, skipping.")
            continue

        for filename in tqdm(os.listdir(loc_folder), desc=f"Scanning {loc_folder} for crops"):
            if filename.lower().endswith(".jpg"):
                original_image_path = os.path.join(loc_folder, filename)
                
                filename_no_ext = filename.replace(".jpg", "")
                
                if ',' not in filename_no_ext:
                    print(f"Warning: Skipping {filename} due to missing comma in coordinates.")
                    continue

                lat_str, lon_season_str = filename_no_ext.split(',', 1)
                
                lon_parts = lon_season_str.rsplit('.', 1)
                if len(lon_parts) != 2:
                    print(f"Warning: Skipping {filename} due to unexpected format in longitude or season.")
                    continue
                
                lon_str, season = lon_parts[0], lon_parts[1]

                try:
                    center_lat = float(lat_str.strip())
                    center_lon = float(lon_str.strip())
                except ValueError:
                    print(f"Warning: Skipping {filename} due to invalid coordinates or season.")
                    continue

                try:
                    img = Image.open(original_image_path)
                    img_width, img_height = img.size

                    num_crops_x = img_width // args.crop_size_pixels
                    num_crops_y = img_height // args.crop_size_pixels

                    for y_idx in range(num_crops_y):
                        for x_idx in range(num_crops_x):
                            left = x_idx * args.crop_size_pixels
                            top = y_idx * args.crop_size_pixels
                            right = left + args.crop_size_pixels
                            bottom = top + args.crop_size_pixels

                            # Calculate crop center in pixels relative to original image center
                            crop_center_pixel_x = left + args.crop_size_pixels / 2
                            crop_center_pixel_y = top + args.crop_size_pixels / 2

                            # Offset from image center in pixels
                            offset_x_pixels = crop_center_pixel_x - img_width / 2
                            offset_y_pixels = crop_center_pixel_y - img_height / 2

                            # Convert pixel offsets to meters
                            offset_x_meters = offset_x_pixels * pixel_to_meter_ratio
                            offset_y_meters = offset_y_pixels * pixel_to_meter_ratio

                            # Convert meter offsets to lat/lon degrees
                            delta_lat_deg = -offset_y_meters / METERS_PER_DEG_LAT # Negative because Y increases downwards
                            
                            meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(center_lat))
                            if meters_per_deg_lon == 0:
                                delta_lon_deg = 0
                            else:
                                delta_lon_deg = offset_x_meters / meters_per_deg_lon

                            crop_center_lat = center_lat + delta_lat_deg
                            crop_center_lon = center_lon + delta_lon_deg
                            
                            all_potential_crops.append({
                                "original_image_path": original_image_path,
                                "crop_bbox": (left, top, right, bottom),
                                "crop_center_lat": crop_center_lat,
                                "crop_center_lon": crop_center_lon,
                                "season": season,
                                "original_filename": filename_no_ext
                            })
                except Exception as e:
                    print(f"Error processing {original_image_path}: {e}")
                    continue
    
    print(f"Found {len(all_potential_crops)} potential crops. Selecting {min(args.num_target_crops, len(all_potential_crops))}.")
    
    # Randomly sample the target number of crops
    selected_crops = random.sample(all_potential_crops, min(args.num_target_crops, len(all_potential_crops)))

    output_dir = os.path.join(args.output_base_dir, args.output_subdir if args.output_subdir else f"{args.num_target_crops}_crops")
    os.makedirs(output_dir, exist_ok=True) # Create specific output directory

    final_crops_metadata = []
    for idx, crop_info in enumerate(tqdm(selected_crops, desc="Saving selected crops")):
        original_image_path = crop_info["original_image_path"]
        img = Image.open(original_image_path)
        crop_img = img.crop(crop_info["crop_bbox"])

        crop_filename = f"{crop_info['original_filename']}_crop{idx}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        crop_img.save(crop_path)

        final_crops_metadata.append({
            "original_image_path": original_image_path,
            "crop_filename": crop_filename,
            "crop_center_lat": crop_info["crop_center_lat"],
            "crop_center_lon": crop_info["crop_center_lon"],
            "season": crop_info["season"]
        })

    metadata_output_path = os.path.join(output_dir, "cropped_images_metadata.json")
    with open(metadata_output_path, "w") as f:
        json.dump(final_crops_metadata, f, indent=4)
    
    print(f"Generated {len(final_crops_metadata)} crops. Metadata saved to {metadata_output_path}")

if __name__ == "__main__":
    args = tyro.cli(CropArgs)
    generate_image_crops(args) 