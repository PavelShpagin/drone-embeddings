import os
import random
from PIL import Image, ImageDraw

# --- Configuration ---
DATA_DIR = "data/earth_imagery"
OUTPUT_DIR = "output"
M_PER_PIXEL = 0.487      # The new value you calculated
CROP_METERS = 100        # The size of each crop in meters
NUM_CROPS = 60           # The number of crops to extract in the flight
# --------------------


def generate_flight_path(img_w, img_h, crop_px, num_steps):
    """
    Generates a random walk flight path within the image boundaries.
    Returns a list of (left, top) coordinates for each crop.
    """
    # Start the flight near the center of the image
    x = img_w // 2
    y = img_h // 2
    
    path = [(x, y)]
    visited = {(x, y)}

    for _ in range(num_steps - 1):
        # Get all possible moves (up, down, left, right)
        possible_moves = [
            (x + crop_px, y), (x - crop_px, y),
            (x, y + crop_px), (x, y - crop_px)
        ]
        
        # Filter for valid moves (within bounds and not previously visited)
        valid_moves = []
        for move_x, move_y in possible_moves:
            if 0 <= move_x <= img_w - crop_px and \
               0 <= move_y <= img_h - crop_px and \
               (move_x, move_y) not in visited:
                valid_moves.append((move_x, move_y))
        
        if not valid_moves:
            break  # Stop if the path gets stuck

        # Choose the next step randomly from the valid moves
        x, y = random.choice(valid_moves)
        path.append((x, y))
        visited.add((x, y))
        
    print(f"Generated a flight path with {len(path)} steps.")
    return path


def simulate_flight_and_visualize(image_path, m_per_pixel, crop_m, num_crops):
    """
    Simulates a flight, creating a trajectory map and a stitched map of the flight path.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    crop_px = int(crop_m / m_per_pixel)
    img_w, img_h = img.size
    print(f"Image size: {img_w}x{img_h}px. Crop size: {crop_px}x{crop_px}px.")

    # 1. Generate the flight path
    path_coords = generate_flight_path(img_w, img_h, crop_px, num_crops)
    if not path_coords:
        print("Could not generate a flight path.")
        return

    # 2. Create the stitched map of explored areas
    min_x = min(c[0] for c in path_coords)
    max_x = max(c[0] for c in path_coords)
    min_y = min(c[1] for c in path_coords)
    max_y = max(c[1] for c in path_coords)

    stitched_w = (max_x - min_x) + crop_px
    stitched_h = (max_y - min_y) + crop_px
    stitched_map = Image.new('RGB', (stitched_w, stitched_h), 'black')

    for x, y in path_coords:
        crop = img.crop((x, y, x + crop_px, y + crop_px))
        # Paste the crop onto the stitched map using translated coordinates
        stitched_map.paste(crop, (x - min_x, y - min_y))

    # 3. Draw the trajectory on the original map
    trajectory_map = img.copy()
    draw = ImageDraw.Draw(trajectory_map)
    
    # Get the center points of each crop box
    path_centers = [(x + crop_px/2, y + crop_px/2) for x, y in path_coords]
    
    # Draw lines connecting the centers
    draw.line(path_centers, fill="yellow", width=8)
    # Draw a green circle at the start and a red one at the end
    if path_centers:
        draw.ellipse((path_centers[0][0]-15, path_centers[0][1]-15, path_centers[0][0]+15, path_centers[0][1]+15), fill="lime", outline="black")
        draw.ellipse((path_centers[-1][0]-15, path_centers[-1][1]-15, path_centers[-1][0]+15, path_centers[-1][1]+15), fill="red", outline="black")

    # 4. Save the output images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_subdir = os.path.join(OUTPUT_DIR, f"flight_{base_name}")
    os.makedirs(output_subdir, exist_ok=True)

    trajectory_map.save(os.path.join(output_subdir, "trajectory_map.png"))
    stitched_map.save(os.path.join(output_subdir, "stitched_map.png"))

    print(f"\nSimulation complete. Outputs saved to: {output_subdir}")


def find_random_image(base_dir):
    """Finds a random image file in the nested directory structure."""
    try:
        locations = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not locations: return None
        random_loc = random.choice(locations)
        loc_path = os.path.join(base_dir, random_loc)
        images = [f for f in os.listdir(loc_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        if not images: return None
        random_img_name = random.choice(images)
        return os.path.join(loc_path, random_img_name)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    random_image_path = find_random_image(DATA_DIR)
    if random_image_path:
        print(f"Found random image: {random_image_path}")
        simulate_flight_and_visualize(random_image_path, M_PER_PIXEL, CROP_METERS, NUM_CROPS)
    else:
        print(f"Error: No images found in the '{DATA_DIR}' directory.")
        print("Please ensure your directory structure is like 'data/earth_imagery/<location_name>/<image_file>'.") 