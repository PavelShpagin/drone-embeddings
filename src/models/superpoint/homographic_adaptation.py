import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def generate_homography(shape, perspective=0.0009, rotation=30, translation=32, scaling=0.2):
    """Generate a random homography transformation matrix."""
    pts1 = np.array([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]], dtype=np.float32)
    pts2 = pts1.copy()
    
    # Apply random perspective
    pts2 += np.random.uniform(-perspective * shape[0], perspective * shape[0], pts2.shape)
    
    # Apply random rotation
    angle = np.random.uniform(-rotation, rotation)
    center = np.mean(pts2, axis=0)
    R = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    pts2 = cv2.transform(pts2.reshape(-1, 1, 2), R).reshape(-1, 2)
    
    # Apply random translation
    pts2 += np.random.uniform(-translation, translation, pts2.shape)
    
    # Apply random scaling
    scale = np.random.uniform(1 - scaling, 1 + scaling)
    pts2 = (pts2 - center) * scale + center
    
    # Compute homography
    H = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
    return H

def warp_points(points, H):
    """Warp points using homography matrix H."""
    if len(points) == 0:
        return points
        
    points = points.reshape(-1, 1, 2)
    points = cv2.perspectiveTransform(points.astype(np.float32), H)
    return points.reshape(-1, 2)

def filter_points(points, shape):
    """Remove points outside image boundaries."""
    if len(points) == 0:
        return points
        
    mask = (points[:, 0] >= 0) & (points[:, 0] < shape[1]) & \
           (points[:, 1] >= 0) & (points[:, 1] < shape[0])
    return points[mask]

def process_image(model, img_path, output_dir, n_views, batch_size, device):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    H, W = img.shape
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    point_prob = np.zeros((H//8, W//8), dtype=np.float32)
    total_views = 0
    for _ in range(n_views // batch_size):
        batch_imgs = []
        batch_homographies = []
        for _ in range(batch_size):
            H_mat = generate_homography((H, W))
            warped = cv2.warpPerspective(img, H_mat, (W, H), flags=cv2.INTER_LINEAR)
            warped_tensor = torch.from_numpy(warped).float() / 255.0
            warped_tensor = warped_tensor.unsqueeze(0).unsqueeze(0)
            batch_imgs.append(warped_tensor)
            batch_homographies.append(H_mat)
        batch_imgs = torch.cat(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            outputs = model(batch_imgs)
            prob = F.softmax(outputs['keypoint_scores'], dim=1)
            prob = prob[:,:-1,:,:]
        for i in range(batch_size):
            view_prob = prob[i].sum(dim=0).cpu().numpy()
            H_inv = np.linalg.inv(batch_homographies[i])
            warped_prob = cv2.warpPerspective(view_prob, H_inv, (W//8, H//8), flags=cv2.INTER_LINEAR)
            point_prob += warped_prob
            total_views += 1
    point_prob /= total_views
    point_prob = (point_prob > 0.01).astype(np.float32)
    out_path = os.path.join(output_dir, Path(img_path).stem)
    np.save(f"{out_path}_points.npy", point_prob)
    cv2.imwrite(f"{out_path}_image.png", img)

def generate_pseudo_labels(model, image_paths, output_dir, n_views=100, batch_size=1, num_workers=26):
    """
    Generate pseudo ground truth labels using homographic adaptation.
    
    Args:
        model: Trained MagicPoint model
        image_paths: List of paths to real images
        output_dir: Directory to save results
        n_views: Number of homographic views per image
        batch_size: Batch size for processing
        num_workers: Number of threads for parallel processing
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda img_path: process_image(model, img_path, output_dir, n_views, batch_size, device), image_paths), total=len(image_paths), desc="Processing images"))

if __name__ == "__main__":
    import argparse
    from glob import glob
    from .superpoint_model import SuperPoint
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing real images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained MagicPoint model')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save pseudo labels')
    parser.add_argument('--n_views', type=int, default=100, help='Number of homographic views per image')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=26, help='Number of threads for parallel processing')
    args = parser.parse_args()
    
    # Load model
    model = SuperPoint(pretrained_path=args.model_path)
    
    # Get image paths recursively
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(glob(os.path.join(args.image_dir, '**', ext), recursive=True))
    
    # Generate pseudo labels
    generate_pseudo_labels(model, image_paths, args.output_dir, args.n_views, args.batch_size, args.num_workers) 