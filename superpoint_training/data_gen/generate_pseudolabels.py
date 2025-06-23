import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
import argparse
import os

from superpoint_training.superpoint_model import SuperPointNet # Assuming this path is correct

# --- Configuration ---
CROPS_DIR = "superpoint_training/crops"
PSEUDOLABELS_DIR = "superpoint_training/pseudolabels"

# Model settings
IMAGE_SIZE = 256
CELL_SIZE = 8  # SuperPoint outputs feature maps at 1/8 resolution

# Homographic Adaptation settings
NUM_HOMOGRAPHIES = 100 # Nh from paper

# Homography augmentation parameters (from original SuperPoint/rpautrat implementation)
# These define the range of random transformations for homography generation
PARAMS = {
    'translation': True, 'rotation': True, 'scaling': True, 'perspective': True,
    'max_scale': 0.5, 'min_scale': 0.5, 'max_angle': 30, 'perspective_amplitude_x': 0.2,
    'perspective_amplitude_y': 0.2, 'allow_artifacts': True,
    'patch_ratio': 0.85 # Ratio of original image to use for homography
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CropsDataset(Dataset):
    def __init__(self, crops_dir):
        self.image_paths = list(Path(crops_dir).glob('*.png'))
        print(f"Found {len(self.image_paths)} crops in {crops_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L") # Grayscale
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        # Convert to numpy array for OpenCV ops later
        img_np = np.array(img, dtype=np.float32) / 255.0 # Normalize to [0,1]
        return img_np, img_path.stem

def generate_homography(image_shape, params):
    """Generates a random homography matrix H and its inverse H_inv."""
    height, width = image_shape
    # Generate a random homography.
    # The patch_ratio parameter controls the size of the image region affected.
    # Original corners of the image
    pts1 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # Perturb the corners to create the destination points
    # This matches the logic in rpautrat's SuperPoint implementation
    s = width * params['patch_ratio']
    center = np.array([width/2., height/2.], dtype=np.float32)
    pts2 = center + np.random.normal(size=(4, 2)) * s / 2

    # Apply random transformations (translation, rotation, scale, perspective)
    if params['translation']:
        t = np.array([width, height]) * np.random.normal(size=2) * 0.1
        pts2 += t
    if params['rotation']:
        angle = np.deg2rad(np.random.uniform(-params['max_angle'], params['max_angle']))
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        pts2 = (pts2 - center) @ R.T + center
    if params['scaling']:
        s = np.exp(np.random.uniform(np.log(params['min_scale']), np.log(params['max_scale']))) * 0.5 + 0.5 # original has 0.5-1.5, my params have 0.5
        pts2 = (pts2 - center) * s + center
    if params['perspective']:
        h = np.random.normal(size=2) * params['perspective_amplitude_x']
        w = np.random.normal(size=2) * params['perspective_amplitude_y']
        pts2[0] += [w[0], h[0]]
        pts2[1] += [-w[1], h[1]]
        pts2[2] += [w[1], -h[1]]
        pts2[3] += [-w[0], -h[0]]

    # Compute homography and its inverse
    H, _ = cv2.findHomography(pts1, pts2)
    H_inv, _ = cv2.findHomography(pts2, pts1)
    return H, H_inv

def generate_pseudolabels(model_path=None):
    print("Starting pseudo-label generation...")
    
    # Load or initialize SuperPoint model
    model = SuperPointNet().to(DEVICE)
    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path} for pseudo-labeling.")
        # Load checkpoint (assuming it's just state_dict for simplicity)
        # For full checkpoint, you'd extract 'model_state_dict'
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("Initializing new model for pseudo-labeling (first round or no checkpoint provided).")
    model.eval()

    Path(PSEUDOLABELS_DIR).mkdir(parents=True, exist_ok=True)
    dataset = CropsDataset(CROPS_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count()//2 or 1)

    total_images_processed = 0
    with torch.no_grad():
        for img_np, img_name in tqdm(dataloader, desc="Generating pseudo-labels"):
            if isinstance(img_name, (list, tuple)): # Ensure img_name is a string, not a list/tuple
                img_name = img_name[0]
            img_np = img_np.squeeze(0).cpu().numpy() # [H, W]
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, H, W]

            # Accumulator for warped heatmaps
            aggregated_semi = torch.zeros(1, 65, IMAGE_SIZE // CELL_SIZE, IMAGE_SIZE // CELL_SIZE, device=DEVICE)

            for i in range(NUM_HOMOGRAPHIES):
                # Generate random homography and warped image
                H, H_inv = generate_homography(img_np.shape, PARAMS)
                warped_img_np = cv2.warpPerspective(img_np, H, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR)
                warped_img_tensor = torch.from_numpy(warped_img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

                # Get heatmap from warped image
                semi_warped, _ = model(warped_img_tensor)

                # Warp heatmap back to original image space using H_inv
                # Need to convert heatmap to numpy for cv2.warpPerspective then back to tensor
                # Note: This is simpler for semi_np directly, but for full heatmap it's complex
                # A simpler approach is to warp the corner points and then interpolate, or direct torch warp

                # Let's directly warp the heatmap using F.grid_sample for efficiency on GPU
                # Create a grid for warping
                grid = F.affine_grid(torch.tensor(H_inv[:2], dtype=torch.float32).unsqueeze(0).to(DEVICE), 
                                     semi_warped.size(), align_corners=True)
                warped_semi = F.grid_sample(semi_warped, grid, mode='bilinear', 
                                            padding_mode='zeros', align_corners=True)
                
                aggregated_semi += warped_semi
            
            # Average the aggregated heatmaps
            pseudo_label_semi = aggregated_semi / NUM_HOMOGRAPHIES

            # Save pseudo-label. Store as .pt (PyTorch tensor file)
            save_path = Path(PSEUDOLABELS_DIR) / f"{img_name}.pt"
            torch.save(pseudo_label_semi.cpu(), save_path)
            total_images_processed += 1

    print(f"Finished generating pseudo-labels. Total: {total_images_processed}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate SuperPoint pseudo-labels using Homographic Adaptation.")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a SuperPoint model checkpoint to use for generating pseudo-labels (for iterative adaptation).')
    args = parser.parse_args()
    generate_pseudolabels(args.model_path) 