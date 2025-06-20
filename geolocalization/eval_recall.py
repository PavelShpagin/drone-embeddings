import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from geolocalization.database import EmbeddingDatabase
from geolocalization import config
from scipy.spatial.distance import cdist

# Haversine distance for lat/lon in degrees
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def load_metadata(metadata_path, crops_dir):
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    # Add full path to each crop
    for entry in meta:
        entry['full_path'] = os.path.join(crops_dir, entry['crop_filename'])
    return meta

def main():
    # Paths
    db_dir = 'data/cropped_loc_images/1000_crops'
    db_meta_path = os.path.join(db_dir, 'cropped_images_metadata.json')
    query_dir = 'data/cropped_loc_images/unseen_queries'
    query_meta_path = os.path.join(query_dir, 'cropped_images_metadata.json')

    # Load metadata
    db_meta = load_metadata(db_meta_path, db_dir)
    query_meta = load_metadata(query_meta_path, query_dir)

    # Build embedding model
    db = EmbeddingDatabase()
    model = db.model
    eval_transforms = db.eval_transforms
    device = db.device
    patch_size = db.patch_size_px

    # Build database embeddings
    print(f"Extracting embeddings for {len(db_meta)} database images...")
    db_embeddings = []
    db_gps = []
    db_paths = []
    for entry in tqdm(db_meta):
        img = Image.open(entry['full_path']).convert('RGB').resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        with torch.no_grad():
            emb = model.get_embedding(eval_transforms(img).unsqueeze(0).to(device)).squeeze().cpu().numpy()
        db_embeddings.append(emb)
        db_gps.append([entry['crop_center_lat'], entry['crop_center_lon']])
        db_paths.append(entry['full_path'])
    db_embeddings = np.stack(db_embeddings)
    db_gps = np.array(db_gps)

    # Build FAISS index
    import faiss
    index = faiss.IndexFlatL2(db_embeddings.shape[1])
    index.add(db_embeddings.astype('float32'))

    # Query loop
    print(f"Evaluating {len(query_meta)} queries...")
    recall1 = 0.0
    recall5 = 0.0
    Q = len(query_meta)
    for entry in tqdm(query_meta):
        img = Image.open(entry['full_path']).convert('RGB').resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        with torch.no_grad():
            q_emb = model.get_embedding(eval_transforms(img).unsqueeze(0).to(device)).squeeze().cpu().numpy()
        D, I = index.search(q_emb[None, :].astype('float32'), 5)
        # Get GPS of query and top-5
        q_gps = np.array([entry['crop_center_lat'], entry['crop_center_lon']])
        top5_gps = db_gps[I[0]]
        # Compute GPS distances
        dists = np.array([haversine(q_gps[0], q_gps[1], lat, lon) for lat, lon in top5_gps])
        # Find the closest in the whole DB (for ground truth)
        all_dists = np.array([haversine(q_gps[0], q_gps[1], lat, lon) for lat, lon in db_gps])
        gt_idx = np.argmin(all_dists)
        # Is the closest GPS match in top-1/top-5?
        if gt_idx == I[0][0]:
            recall1 += 1.0 / Q
        if gt_idx in I[0]:
            recall5 += 1.0 / Q
    print(f"Recall@1: {recall1:.3f}")
    print(f"Recall@5: {recall5:.3f}")

if __name__ == '__main__':
    import torch
    main() 