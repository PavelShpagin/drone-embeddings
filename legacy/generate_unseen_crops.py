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
class UnseenCropArgs:
    num_target_crops: int = 1000
    crop_size_pixels: int = 210
    source_file: str = "data/test/test.jpg"
    output_dir: str = "data/unseen_crops"
    test_lat: float = 50.4162  # Approximate test location latitude
    test_lon: float = 30.8906  # Approximate test location longitude

def generate_unseen_crops(args: UnseenCropArgs):
    os.makedirs(args.output_dir, exist_ok=True)
    
    pixel_to_meter_ratio = 100 / args.crop_size_pixels
    METERS_PER_DEG_LAT = 111000
    
    # Load the test image
    try:
        img = Image.open(args.source_file)
        img_width, img_height = img.size
        print(f"Loaded test image: {img_width}x{img_height} pixels")
    except Exception as e:
        print(f"Error loading test image: {e}")
        return

    # Calculate number of possible crops
    num_crops_x = img_width // args.crop_size_pixels
    num_crops_y = img_height // args.crop_size_pixels
    total_possible_crops = num_crops_x * num_crops_y
    
    print(f"Possible crops: {num_crops_x} x {num_crops_y} = {total_possible_crops}")
    print(f"Target crops: {min(args.num_target_crops, total_possible_crops)}")
    
    all_potential_crops = []
    
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
            delta_lat_deg = -offset_y_meters / METERS_PER_DEG_LAT
            meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(args.test_lat))
            delta_lon_deg = offset_x_meters / meters_per_deg_lon if meters_per_deg_lon != 0 else 0

            crop_center_lat = args.test_lat + delta_lat_deg
            crop_center_lon = args.test_lon + delta_lon_deg
            
            all_potential_crops.append({
                "crop_bbox": (left, top, right, bottom),
                "crop_center_lat": crop_center_lat,
                "crop_center_lon": crop_center_lon,
                "x_idx": x_idx,
                "y_idx": y_idx
            })
    
    # Randomly sample the target number of crops
    selected_crops = random.sample(all_potential_crops, min(args.num_target_crops, len(all_potential_crops)))

    final_crops_metadata = []
    for idx, crop_info in enumerate(tqdm(selected_crops, desc="Saving unseen crops")):
        crop_img = img.crop(crop_info["crop_bbox"])

        crop_filename = f"test_unseen_crop_{idx:04d}.jpg"
        crop_path = os.path.join(args.output_dir, crop_filename)
        crop_img.save(crop_path)

        final_crops_metadata.append({
            "crop_filename": crop_filename,
            "crop_center_lat": crop_info["crop_center_lat"],
            "crop_center_lon": crop_info["crop_center_lon"],
            "source_image": "test.jpg"
        })

    metadata_output_path = os.path.join(args.output_dir, "unseen_crops_metadata.json")
    with open(metadata_output_path, "w") as f:
        json.dump(final_crops_metadata, f, indent=4)
    
    print(f"Generated {len(final_crops_metadata)} unseen crops.")
    print(f"Metadata saved to {metadata_output_path}")

if __name__ == "__main__":
    args = tyro.cli(UnseenCropArgs)
    generate_unseen_crops(args) 