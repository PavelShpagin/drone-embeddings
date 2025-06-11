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

# --- CONFIGURATION ---
class Config:
    # Execution
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    DATA_DIR = "data/earth_imagery"
    TEST_IMG_PATH = "data/test/test.jpg"
    OUTPUT_DIR = "training_results"

    # Models to train
    BACKBONES = ['resnet50', 'efficientnet_b0', 'mobilenet_v2'] # mobilenet_v3 and shufflenet are more complex to adapt

    # Training Hyperparameters
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 500))
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
def get_augmentations():
    return transforms.Compose([
        transforms.RandomInvert(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
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

# --- MODEL COMPONENTS ---
class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=128, alpha=100.0, normalize_input=True):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clus, traindescs):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            -self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C, _, _ = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for i in range(self.num_clusters):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[i:i+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, i:i+1, :].expand(N, C, -1).unsqueeze(0)
            vlad[:, i:i+1, :] = residual.sum(dim=-1, keepdim=True)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad

def get_backbone(backbone_name):
    if backbone_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        out_channels = model.fc.in_features
        model.fc = nn.Identity()
        # Return model and a hookable layer
        return model, out_channels
    elif backbone_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        out_channels = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, out_channels
    elif backbone_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        out_channels = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, out_channels
    else:
        raise ValueError(f"Backbone {backbone_name} not supported.")

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
        features = self.feature_extractor(x)
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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Epoch Loss: ?")
    for batch in pbar:
        optimizer.zero_grad()
        
        # Unpack and send to device
        images = [img.to(device) for img in batch]
        anchor_img, pos_imgs, neg_imgs = images[0], images[1:5], images[5:9]

        # Get embeddings
        all_imgs = torch.cat([anchor_img] + pos_imgs + neg_imgs, dim=0)
        all_embs = model(all_imgs)
        
        anchor_emb, pos_embs, neg_embs = torch.split(all_embs, [Config.BATCH_SIZE, Config.BATCH_SIZE*4, Config.BATCH_SIZE*4])
        pos_embs = pos_embs.view(Config.BATCH_SIZE, 4, -1)
        neg_embs = neg_embs.view(Config.BATCH_SIZE, 4, -1)
        
        # Calculate triplet loss across all combinations
        loss = 0
        for i in range(4): # 4 positives
            for j in range(4): # 4 negatives
                loss += criterion(anchor_emb, pos_embs[:, i, :], neg_embs[:, j, :])
        
        loss /= 16.0 # Average over all pairs
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Epoch Loss: {total_loss / (pbar.n + 1):.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, test_img_path, device, output_dir):
    if not os.path.exists(test_img_path):
        print(f"Test image not found at {test_img_path}. Skipping evaluation.")
        return

    print("Running evaluation...")
    model.eval()
    
    test_img = Image.open(test_img_path).convert("RGB")
    w, h = test_img.size
    
    # Create a gallery of database embeddings
    db_crops = []
    db_coords = []
    while len(db_crops) < Config.EVAL_SAMPLES:
        x = random.randint(0, w - Config.CROP_SIZE_PIXELS - 1)
        y = random.randint(0, h - Config.CROP_SIZE_PIXELS - 1)
        # Ensure non-overlap (simplified check)
        is_overlap = any(np.sqrt((x-cx)**2 + (y-cy)**2) < Config.CROP_SIZE_PIXELS for cx, cy in db_coords)
        if not is_overlap:
            crop = test_img.crop((x, y, x + Config.CROP_SIZE_PIXELS, y + Config.CROP_SIZE_PIXELS))
            db_crops.append(get_augmentations()(crop))
            db_coords.append((x,y))

    db_embs = model(torch.stack(db_crops).to(device))
    
    # Create a set of query embeddings (perturbed versions of db)
    query_embs = []
    gt_indices = list(range(len(db_crops)))
    for i, (x, y) in enumerate(db_coords):
        px = int(x + random.uniform(-10, 10))
        py = int(y + random.uniform(-10, 10))
        px = max(0, min(px, w - Config.CROP_SIZE_PIXELS))
        py = max(0, min(py, h - Config.CROP_SIZE_PIXELS))
        crop = test_img.crop((px, py, px + Config.CROP_SIZE_PIXELS, py + Config.CROP_SIZE_PIXELS))
        query_embs.append(get_augmentations()(crop))
        
    query_embs = model(torch.stack(query_embs).to(device))

    # Calculate recall
    recalls = {k: 0 for k in Config.EVAL_RECALL_K}
    for i in range(len(query_embs)):
        q_emb = query_embs[i].unsqueeze(0)
        dists = F.pairwise_distance(q_emb, db_embs)
        sorted_indices = torch.argsort(dists)
        
        for k in Config.EVAL_RECALL_K:
            if gt_indices[i] in sorted_indices[:k]:
                recalls[k] += 1
    
    for k in Config.EVAL_RECALL_K:
        recalls[k] /= len(query_embs)
        print(f"Recall@{k}: {recalls[k]:.4f}")
    
    # Save recall plot
    plt.figure()
    plt.bar([str(k) for k in recalls.keys()], recalls.values())
    plt.title("Recall@K")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, "evaluation_recall.png"))
    plt.close()


def run_training_pipeline():
    # Create output directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    augmentations = get_augmentations()

    for backbone in Config.BACKBONES:
        print(f"\n{'='*20} Training Backbone: {backbone.upper()} {'='*20}")
        
        backbone_dir = os.path.join(Config.OUTPUT_DIR, backbone)
        os.makedirs(backbone_dir, exist_ok=True)
        
        # --- Model Initialization ---
        model = SiameseNet(backbone_name=backbone).to(Config.DEVICE)
        optimizer = Config.OPTIMIZER(model.parameters(), lr=Config.LR)
        criterion = TripletLoss(margin=Config.TRIPLET_MARGIN).to(Config.DEVICE)
        
        for stage in [1, 2]:
            print(f"\n--- Stage {stage}: {'Pre-training' if stage == 1 else 'Season Invariance'} ---")
            stage_dir = os.path.join(backbone_dir, f"stage_{stage}")
            os.makedirs(stage_dir, exist_ok=True)

            dataset = SatelliteDataset(
                data_dir=Config.DATA_DIR,
                stage=stage,
                num_samples=Config.SAMPLES_PER_EPOCH,
                augmentations=augmentations
            )
            # Use a persistent worker DataLoader to speed up
            dataloader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True
            )

            loss_history = []
            for epoch in range(Config.NUM_EPOCHS):
                print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
                epoch_loss = train_one_epoch(model, dataloader, criterion, optimizer, Config.DEVICE)
                loss_history.append(epoch_loss)
            
            # --- Save Loss Plot ---
            plt.figure()
            plt.plot(loss_history)
            plt.title(f"Stage {stage} Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(stage_dir, "loss_curve.png"))
            plt.close()
            
            # --- Save Model ---
            model_path = os.path.join(stage_dir, f"{backbone}_stage{stage}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

            # --- Evaluation ---
            evaluate(model, Config.TEST_IMG_PATH, Config.DEVICE, stage_dir)

    print("\nTraining complete for all backbones.")

if __name__ == '__main__':
    # Add a check for data directory
    if not os.path.exists(Config.DATA_DIR) or not os.listdir(Config.DATA_DIR):
        print(f"Error: Data directory '{Config.DATA_DIR}' is empty or does not exist.")
        print("Please ensure you have downloaded the satellite imagery into subfolders like 'data/earth_imagery/loc1/'.")
    else:
        run_training_pipeline()
