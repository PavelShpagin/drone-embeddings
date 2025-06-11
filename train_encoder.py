import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import time
from pathlib import Path
from dotenv import load_dotenv
import timm
import kornia.augmentation as K
import kornia.geometry.transform as T

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
class Config:
    # Execution
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    DATA_DIR = os.getenv("DATA_DIR", "data/earth_imagery")
    TEST_IMG_PATH = os.getenv("TEST_IMG_PATH", "data/test/test.jpg")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "training_results")

    # Models to train (using timm model names)
    BACKBONES = ['efficientnet_b0']

    # Training Hyperparameters
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
    SAMPLES_PER_EPOCH = 20000  # Full scale
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
    LR = float(os.getenv("LEARNING_RATE", 1e-6))
    OPTIMIZER = torch.optim.Adam
    TRIPLET_MARGIN = 0.2

    # Sampling and Augmentation
    CROP_SIZE_PIXELS = 100
    M_PER_PIXEL = 1.0  # Assuming 1 meter per pixel
    POS_RADIUS_M = 100
    NEG_RADIUS_M = 400
    
    # Evaluation
    EVAL_SAMPLES = 500
    EVAL_RECALL_K = [1, 5, 10]


# --- AUGMENTATIONS ---
def get_cpu_augmentations():
    """Returns basic augmentations to be run on the CPU."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def get_gpu_augmentations(device: torch.device) -> nn.Module:
    """Returns a sequential module of augmentations to be run on the GPU."""
    return nn.Sequential(
        # Note: Kornia expects (B, C, H, W) and values in range [0, 1]
        K.RandomInvert(p=0.1),
        K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5), p=0.3),
        K.RandomSharpness(sharpness=2, p=0.2),
        K.RandomAutoContrast(p=0.2),
        K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.8),
        K.RandomRotation(degrees=180, p=0.5),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        # Normalize after all other augmentations
        K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
        K.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ).to(device)

def get_eval_transforms():
    """Returns basic transforms for evaluation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- LOSS FUNCTION ---
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_ap = F.pairwise_distance(anchor, positive)
        dist_an = F.pairwise_distance(anchor, negative)
        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()

# --- NetVLAD Implementation ---
class NetVLAD(nn.Module):
    """
    NetVLAD layer implementation.
    This version is vectorized for efficiency and avoids Python loops.
    """
    def __init__(self, num_clusters=32, dim=1024, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x):
        N, C, H, W = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # L2 normalize features

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each cluster
        # residuals shape: (N, num_clusters, C, H*W)
        vlad_residuals = x_flatten.unsqueeze(1) - self.centroids.unsqueeze(0).unsqueeze(-1)
        
        # weight residuals by soft-assignment
        # soft_assign shape (N, num_clusters, H*W) -> unsqueezed for broadcasting
        weighted_residuals = soft_assign.unsqueeze(2) * vlad_residuals
        
        # sum over all descriptors for each cluster
        vlad = weighted_residuals.sum(dim=-1)

        # intra-normalization and flattening
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)               # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

# --- Backbone Networks ---
def get_backbone(backbone_name: str) -> (nn.Module, int):
    """
    Creates a backbone model using the timm library, configured to output a
    4D feature map instead of a classification vector.

    Args:
        backbone_name (str): The name of the model in timm's library.
        
    Returns:
        A tuple containing the model (nn.Module) and the number of output
        feature channels (int).
    """
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        features_only=True,
    )
    
    # Get the number of output channels from the model's feature_info
    feature_dim = model.feature_info.channels()[-1]
    
    return model, feature_dim

class SiameseNet(nn.Module):
    def __init__(self, backbone_name):
        super(SiameseNet, self).__init__()
        self.feature_extractor, feature_dim = get_backbone(backbone_name)
        
        # Intermediate projection layer, as per CVM-Net style
        self.projection_dim = 1024 
        self.projection = nn.Sequential(
            nn.Conv2d(feature_dim, self.projection_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.projection_dim),
            nn.ReLU(inplace=True)
        )

        self.netvlad = NetVLAD(num_clusters=32, dim=self.projection_dim)
        
    def forward(self, x):
        # When using features_only=True, the extractor returns a list of feature maps.
        # We want the last one, which is the richest feature representation.
        features = self.feature_extractor(x)[-1]
        projected_features = self.projection(features)
        embedding = self.netvlad(projected_features)
        return embedding

# --- DATASET ---
class SatelliteDataset(Dataset):
    def __init__(self, data_dir, stage, num_samples, augmentations):
        super().__init__()
        self.stage = stage
        self.num_samples = num_samples
        self.augmentations = augmentations
        self.crop_size_px = Config.CROP_SIZE_PIXELS
        self.m_per_pixel = Config.M_PER_PIXEL
        self.pos_radius_px = Config.POS_RADIUS_M / self.m_per_pixel
        self.neg_radius_px = Config.NEG_RADIUS_M / self.m_per_pixel

        self.locations = self._find_locations(data_dir)
        if not self.locations:
            raise FileNotFoundError(f"No valid images found in {data_dir}")

    def _find_locations(self, data_dir):
        locations = {}
        if not os.path.exists(data_dir): return locations
        for loc_dir in os.listdir(data_dir):
            loc_path = os.path.join(data_dir, loc_dir)
            if not os.path.isdir(loc_path): continue
            
            seasons = {}
            for fname in os.listdir(loc_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    parts = fname.rsplit('.', 2)
                    if len(parts) == 3:
                        season = parts[1].lower()
                        seasons[season] = os.path.join(loc_path, fname)
            
            if seasons:
                locations[loc_dir] = seasons
        return locations

    def __len__(self):
        return self.num_samples

    def _get_random_crop_coords(self, img_w, img_h, center_x=None, center_y=None):
        if center_x is not None:
            # Positive sampling
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0, self.pos_radius_px)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
        else:
            # Anchor or Negative sampling
            x = random.randint(0, img_w - self.crop_size_px - 1)
            y = random.randint(0, img_h - self.crop_size_px - 1)
        
        # Clamp to bounds
        x = max(0, min(x, img_w - self.crop_size_px))
        y = max(0, min(y, img_h - self.crop_size_px))
        return x, y

    def __getitem__(self, index):
        loc_name = random.choice(list(self.locations.keys()))
        loc_seasons = self.locations[loc_name]

        if self.stage == 1:
            # All crops from one random image
            season = random.choice(list(loc_seasons.keys()))
            img_path = loc_seasons[season]
            large_img = Image.open(img_path).convert('RGB')
            w, h = large_img.size

            # Get anchor
            ax, ay = self._get_random_crop_coords(w, h)
            anchor_crop = self.augmentations(large_img.crop((ax, ay, ax + self.crop_size_px, ay + self.crop_size_px)))

            # Get positives
            pos_crops = []
            for _ in range(4):
                px, py = self._get_random_crop_coords(w, h, center_x=ax, center_y=ay)
                pos_crops.append(self.augmentations(large_img.crop((px, py, px + self.crop_size_px, py + self.crop_size_px))))

            # Get negatives
            neg_crops = []
            for _ in range(4):
                while True:
                    nx, ny = self._get_random_crop_coords(w, h)
                    if np.sqrt((nx-ax)**2 + (ny-ay)**2) > self.neg_radius_px:
                        neg_crops.append(self.augmentations(large_img.crop((nx, ny, nx + self.crop_size_px, ny + self.crop_size_px))))
                        break
            
            large_img.close()
            return anchor_crop, *pos_crops, *neg_crops

        elif self.stage == 2:
            # Crops can come from different seasons of the same location
            if len(loc_seasons) < 2: # Need at least 2 seasons to learn invariance
                # Fallback to stage 1 logic for this sample
                return self.__getitem__(index) # This is a simplification
            
            # Anchor
            anchor_season = random.choice(list(loc_seasons.keys()))
            anchor_img = Image.open(loc_seasons[anchor_season]).convert('RGB')
            w, h = anchor_img.size
            ax, ay = self._get_random_crop_coords(w, h)
            anchor_crop = self.augmentations(anchor_img.crop((ax, ay, ax + self.crop_size_px, ay + self.crop_size_px)))
            anchor_img.close()
            
            # Positives
            pos_crops = []
            for _ in range(4):
                pos_season = random.choice(list(loc_seasons.keys()))
                pos_img = Image.open(loc_seasons[pos_season]).convert('RGB')
                w, h = pos_img.size
                px, py = self._get_random_crop_coords(w, h, center_x=ax, center_y=ay)
                pos_crops.append(self.augmentations(pos_img.crop((px, py, px + self.crop_size_px, py + self.crop_size_px))))
                pos_img.close()

            # Negatives
            neg_crops = []
            for _ in range(4):
                neg_season = random.choice(list(loc_seasons.keys()))
                neg_img = Image.open(loc_seasons[neg_season]).convert('RGB')
                w, h = neg_img.size
                while True:
                    nx, ny = self._get_random_crop_coords(w, h)
                    if np.sqrt((nx-ax)**2 + (ny-ay)**2) > self.neg_radius_px:
                        neg_crops.append(self.augmentations(neg_img.crop((nx, ny, nx + self.crop_size_px, ny + self.crop_size_px))))
                        break
                neg_img.close()
                
            return anchor_crop, *pos_crops, *neg_crops

# --- TRAINING & EVALUATION ---

def train_one_epoch(model, dataloader, criterion, optimizer, device, gpu_augmentations):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_data in progress_bar:
        # The dataloader returns a list of 9 tensors.
        # We unpack them into anchor, a list of positives, and a list of negatives.
        anchor = batch_data[0]
        positives = list(batch_data[1:5])
        negatives = list(batch_data[5:9])

        # Move all tensors to the GPU first and combine into a single batch
        # The structure is [anchor, p1, p2, p3, p4, n1, n2, n3, n4]
        # where each is a batch of size BATCH_SIZE
        batch_size = anchor.size(0)
        num_pos = len(positives)
        num_neg = len(negatives)

        # Create a single large batch for augmentation
        # The tensors from dataloader are on CPU, move them to GPU
        all_images = torch.cat([anchor] + positives + negatives, dim=0).to(device)

        # Apply GPU augmentations to the entire batch at once
        all_images_aug = gpu_augmentations(all_images)
        
        optimizer.zero_grad()
        
        # Get embeddings for the augmented batch
        embeddings = model(all_images_aug)
        
        # Split embeddings back
        anchor_emb, pos_embs, neg_embs = torch.split(
            embeddings, 
            [batch_size, batch_size * num_pos, batch_size * num_neg]
        )
        
        # Reshape and average the embeddings for positives and negatives
        pos_embs = pos_embs.view(num_pos, batch_size, -1).mean(dim=0)
        neg_embs = neg_embs.view(num_neg, batch_size, -1).mean(dim=0)

        loss = criterion(anchor_emb, pos_embs, neg_embs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, test_img_path, device, output_dir, epoch=None):
    model.eval()
    print("Running evaluation...")

    # --- Load and prepare test image ---
    try:
        test_img = Image.open(test_img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Evaluation test image not found at {test_img_path}. Skipping evaluation.")
        # Return dummy values
        return {f'R@{k}': 0 for k in Config.EVAL_RECALL_K}

    w, h = test_img.size
    
    eval_transforms = get_eval_transforms()
    
    # Create gallery of database embeddings
    db_crops = []
    db_coords = []
    while len(db_crops) < Config.EVAL_SAMPLES:
        x = random.randint(0, w - Config.CROP_SIZE_PIXELS - 1)
        y = random.randint(0, h - Config.CROP_SIZE_PIXELS - 1)
        # Simple non-overlap check
        is_overlap = any(np.sqrt((x-cx)**2 + (y-cy)**2) < Config.CROP_SIZE_PIXELS for cx, cy in db_coords)
        if not is_overlap:
            crop = test_img.crop((x, y, x + Config.CROP_SIZE_PIXELS, y + Config.CROP_SIZE_PIXELS))
            db_crops.append(eval_transforms(crop))
            db_coords.append((x, y))

    if not db_crops:
        print("Could not generate any database crops for evaluation.")
        return {f'R@{k}': 0 for k in Config.EVAL_RECALL_K}

    db_embs = model(torch.stack(db_crops).to(device))
    
    # Create query embeddings (perturbed versions of db)
    query_embs_list = []
    gt_indices = list(range(len(db_crops)))
    for i, (x, y) in enumerate(db_coords):
        # Small perturbation
        px = int(x + random.uniform(-20, 20))
        py = int(y + random.uniform(-20, 20))
        px = max(0, min(px, w - Config.CROP_SIZE_PIXELS))
        py = max(0, min(py, h - Config.CROP_SIZE_PIXELS))
        crop = test_img.crop((px, py, px + Config.CROP_SIZE_PIXELS, py + Config.CROP_SIZE_PIXELS))
        query_embs_list.append(eval_transforms(crop))
    
    if not query_embs_list:
        print("Could not generate any query crops for evaluation.")
        return {f'R@{k}': 0 for k in Config.EVAL_RECALL_K}
        
    query_embs = model(torch.stack(query_embs_list).to(device))

    # Calculate recall and mAP
    recalls = {k: 0 for k in Config.EVAL_RECALL_K}
    mAPs = {k: 0 for k in Config.EVAL_RECALL_K}
    
    for i in range(len(query_embs)):
        q_emb = query_embs[i].unsqueeze(0)
        dists = F.pairwise_distance(q_emb, db_embs)
        sorted_indices = torch.argsort(dists)
        
        # Find rank of the ground truth item
        try:
            gt_rank = (sorted_indices == gt_indices[i]).nonzero(as_tuple=True)[0].item() + 1
        except IndexError:
            continue # Should not happen if gt_indices are correct

        for k in Config.EVAL_RECALL_K:
            if gt_rank <= k:
                recalls[k] += 1
                mAPs[k] += 1.0 / gt_rank  # Add Average Precision for this query
    
    for k in Config.EVAL_RECALL_K:
        if len(query_embs) > 0:
            recalls[k] /= len(query_embs)
            mAPs[k] /= len(query_embs)
        print(f"Recall@{k}: {recalls[k]:.4f}, mAP@{k}: {mAPs[k]:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(len(Config.EVAL_RECALL_K))
    
    plt.bar(index, recalls.values(), bar_width, label='Recall@K')
    plt.bar(index + bar_width, mAPs.values(), bar_width, label='mAP@K (Precision)')
    
    plt.title(f"Evaluation Metrics at Epoch {epoch}" if epoch is not None else "Final Evaluation Metrics")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.xticks(index + bar_width / 2, [str(k) for k in Config.EVAL_RECALL_K])
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save the plot
    fname = f"evaluation_metrics_epoch_{epoch}.png" if epoch is not None else "evaluation_metrics_final.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    
    return recalls


def run_training_pipeline():
    # Create output directories
    output_path = Path(Config.OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    for backbone_name in Config.BACKBONES:
        print(f"--- Starting Training for {backbone_name} ---")
        
        # --- Model Setup ---
        model = SiameseNet(backbone_name).to(Config.DEVICE)
        optimizer = Config.OPTIMIZER(model.parameters(), lr=Config.LR)
        criterion = TripletLoss(margin=Config.TRIPLET_MARGIN)
        gpu_augmentations = get_gpu_augmentations(torch.device(Config.DEVICE))
        
        # --- Checkpoint Loading ---
        start_epoch = 0
        best_recall_at_1 = -1.0
        checkpoint_dir = output_path / backbone_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # LEGACY: Try to load from old 'stage_1' directory first
        legacy_checkpoint_path = output_path / backbone_name / "stage_1" / "efficientnet_b0_stage1_best_recall.pth"
        if legacy_checkpoint_path.exists():
            print(f"Found legacy checkpoint at {legacy_checkpoint_path}. Loading weights...")
            # Set weights_only=False to load older checkpoints saved with different pickle protocols.
            model.load_state_dict(torch.load(legacy_checkpoint_path, weights_only=False))
            # Since we don't know the epoch or optimizer state, we start fresh but with the right weights
            start_epoch = 0 
            print("Legacy weights loaded. New checkpoints will be saved to the new directory structure.")

        # Find the latest epoch to resume from in the new checkpoint directory
        latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint_path:
            print(f"Resuming from checkpoint: {latest_checkpoint_path}")
            # Set weights_only=False as we are loading a dictionary, not just weights.
            checkpoint = torch.load(latest_checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            # Scan all previous checkpoints to find the true best recall
            best_recall_at_1 = find_best_recall_from_checkpoints(checkpoint_dir)
        elif not legacy_checkpoint_path.exists():
            print("No new or legacy checkpoints found, starting from scratch.")

        # --- Data Setup ---
        cpu_augmentations = get_cpu_augmentations()

        # --- Training Loop ---
        for epoch in range(start_epoch, Config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            
            # Determine current training stage
            current_stage = 2 if epoch >= (Config.NUM_EPOCHS // 2) else 1
            print(f"Current training stage: {current_stage}")

            # Update dataset based on stage
            if 'dataset' not in locals() or dataset.stage != current_stage:
                 dataset = SatelliteDataset(
                    data_dir=Config.DATA_DIR,
                    stage=current_stage,
                    num_samples=Config.SAMPLES_PER_EPOCH,
                    augmentations=cpu_augmentations
                )
                 dataloader = DataLoader(
                    dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=True, 
                    num_workers=4, 
                    pin_memory=True
                )

            epoch_loss = train_one_epoch(model, dataloader, criterion, optimizer, Config.DEVICE, gpu_augmentations)
            print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")

            # --- Evaluation ---
            recall_results = evaluate(
                model=model, 
                test_img_path=Config.TEST_IMG_PATH, 
                device=Config.DEVICE,
                output_dir=(output_path / backbone_name),
                epoch=epoch + 1
            )
            
            # --- Model Saving ---
            current_recall_at_1 = recall_results['R@1']
            
            # Save checkpoint after every epoch
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'recall': recall_results
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            if current_recall_at_1 > best_recall_at_1:
                best_recall_at_1 = current_recall_at_1
                final_model_path = output_path / backbone_name / "final_model.pth"
                torch.save(model.state_dict(), final_model_path)
                print(f"New best model saved to {final_model_path} with R@1: {best_recall_at_1:.4f}")

    print("--- Training complete for all backbones ---")

def find_latest_checkpoint(checkpoint_dir: Path):
    """Finds the latest checkpoint file in a directory."""
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    return latest_checkpoint

def find_best_recall_from_checkpoints(checkpoint_dir: Path) -> float:
    """Scans all checkpoints in a directory and returns the highest R@1 found."""
    best_recall = -1.0
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        return best_recall
    
    print("Scanning existing checkpoints to find best historical recall...")
    for ckpt_path in checkpoints:
        try:
            # Load to CPU to avoid using GPU memory just for a number
            # Set weights_only=False as we are loading a dictionary, not just weights.
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # The recall dict is saved under the 'recall' key
            recall_at_1 = ckpt.get('recall', {}).get('R@1', -1.0)
            if recall_at_1 > best_recall:
                best_recall = recall_at_1
        except Exception as e:
            print(f"Warning: Could not read recall from checkpoint {ckpt_path}: {e}")
    
    print(f"Found best historical Recall@1: {best_recall:.4f}")
    return best_recall

if __name__ == '__main__':
    run_training_pipeline()
