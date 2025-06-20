import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import timm

# Settings
IMAGE_DIRS = [f'data/earth_imagery/loc{i}' for i in range(1, 11)]
OUTPUT_DIR = 'foundloc_repro/features'
MODEL_NAME = 'vit_base_patch16_224_dino'
IMAGE_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load DINOvIT model
model = timm.create_model(MODEL_NAME, pretrained=True)
model.eval()
model.to(DEVICE)

# Feature extraction function
def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    img = img.to(DEVICE)
    with torch.no_grad():
        feats = model.forward_features(img)
        if isinstance(feats, dict):
            feats = feats['x_norm_patchtokens'] if 'x_norm_patchtokens' in feats else list(feats.values())[0]
        feats = feats.squeeze(0).cpu().numpy()  # (num_patches, dim)
    return feats

# Main loop
all_images = []
for d in IMAGE_DIRS:
    all_images.extend(sorted(glob.glob(os.path.join(d, '*.jpg'))))

for img_path in tqdm(all_images, desc='Extracting features'):
    feats = extract_features(img_path)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path) + '.npy')
    np.save(out_path, feats)

print(f'Features saved to {OUTPUT_DIR}')

"""
This script extracts DINOvIT features from all images in data/earth_imagery/loc1 ... loc10.
- Features are saved as .npy files in foundloc_repro/features.
- Each .npy file contains a (num_patches, feature_dim) array.
- Uses timm to load a pretrained DINOvIT model.
""" 