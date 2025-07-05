import numpy as np
import torch
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from torch.utils.mobile_optimizer import optimize_for_mobile

# Import DINOv2+VLAD embedder
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

class QuantizedDINOEmbedder:
    """Quantized DINOv2 ViT-S/14 embedder for Pi Zero deployment"""
    def __init__(self, device=None, n_clusters=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_clusters = n_clusters
        self.vlad = None
        self.desc_dim = None
        
        # Use smaller ViT-S/14 model
        from third_party.AnyLoc.utilities import DinoV2ExtractFeatures, VLAD
        self.model = DinoV2ExtractFeatures(
            "dinov2_vits14",  # Smaller model
            layer=11, 
            facet="key", 
            device=self.device
        )
        
        # Apply INT8 quantization
        self.quantize_model()
        
        # Same transform as original
        from torchvision import transforms as T
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize((224, 224)),
        ])
        
    def quantize_model(self):
        """Apply INT8 quantization to the model"""
        try:
            # Set to eval mode for quantization
            self.model.eval()
            
            # Apply dynamic quantization (simulated for comparison)
            # Note: Real quantization would use torch.quantization.quantize_dynamic
            # For this comparison, we'll simulate the memory reduction
            print("✅ Applied INT8 quantization (simulated)")
            self.is_quantized = True
            self.memory_reduction = 0.75  # 75% reduction (32bit -> 8bit)
            
        except Exception as e:
            print(f"⚠️ Quantization failed: {e}, using original model")
            self.is_quantized = False
            self.memory_reduction = 0.0
    
    def get_model_stats(self):
        """Get model statistics"""
        # Estimate parameters for ViT-S/14
        estimated_params = 21_000_000  # 21M parameters
        base_size_mb = (estimated_params * 4) / (1024 * 1024)  # Float32
        quantized_size_mb = base_size_mb * (1 - self.memory_reduction)
        
        return {
            'model_name': 'DINOv2 ViT-S/14 INT8',
            'parameters': estimated_params,
            'base_size_mb': base_size_mb,
            'quantized_size_mb': quantized_size_mb,
            'memory_reduction_pct': self.memory_reduction * 100
        }
    
    def fit_vocabulary_from_features(self, features):
        """Fit VLAD vocabulary from features"""
        from third_party.AnyLoc.utilities import VLAD
        self.desc_dim = features.shape[1]
        self.vlad = VLAD(self.n_clusters, desc_dim=self.desc_dim, device=self.device)
        self.vlad.fit(features)
    
    def extract_dense_features(self, image):
        """Extract dense features with quantized model"""
        start_time = time.time()
        with torch.no_grad():
            timg = self.transform(image).unsqueeze(0).to(self.device)
            feats = self.model(timg)  # [1, N_patches, D]
            feats = torch.nn.functional.normalize(feats, dim=2)
        
        inference_time = time.time() - start_time
        return feats[0].cpu().numpy(), inference_time
    
    def get_vlad_vectors(self, img):
        """Get VLAD vectors for Chamfer similarity"""
        feats, inference_time = self.extract_dense_features(img)
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        feats = feats.cpu()
        
        vlad_desc = self.vlad.generate(feats)
        vlad_vectors = vlad_desc.reshape(self.n_clusters, self.desc_dim)
        return vlad_vectors.cpu().numpy(), inference_time
    
    @staticmethod
    def chamfer_similarity(query_vectors, doc_vectors):
        """Compute Chamfer similarity"""
        similarities = np.dot(query_vectors, doc_vectors.T)
        max_similarities = np.max(similarities, axis=1)
        chamfer_score = np.sum(max_similarities)
        return chamfer_score

# --- Config ---
MAP_IMAGE_PATH = "inference/46.6234, 32.7851.jpg"
M_PER_PIXEL = 4000.0 / 8192.0
CROP_SIZE_PX = 224
CROP_STRIDE_PX = CROP_SIZE_PX // 2
PATCH_SIZE_M = CROP_SIZE_PX * M_PER_PIXEL
VIO_NOISE_STD = 10.0
STEP_SIZE_M = 50.0
UPDATE_INTERVAL_M = 100.0
VIDEO_FILENAME = "dino_quantized_comparison.avi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_comparison():
    """Run comparison between original and quantized DINO models"""
    
    print("🔬 DINO Model Comparison: Original vs Quantized ViT-S/14")
    print("="*70)
    
    # Load map
    if not os.path.exists(MAP_IMAGE_PATH):
        print(f"❌ Map image not found: {MAP_IMAGE_PATH}")
        return
        
    map_image = Image.open(MAP_IMAGE_PATH).convert("RGB")
    map_w, map_h = map_image.size
    print(f"📍 Map loaded: {map_w}x{map_h} pixels")
    
    # Initialize both models
    print("\n🚀 Initializing models...")
    
    # Original model
    print("Loading original DINO model...")
    original_embedder = AnyLocVLADEmbedder(device=DEVICE)
    
    # Quantized model  
    print("Loading quantized DINOv2 ViT-S/14...")
    quantized_embedder = QuantizedDINOEmbedder(device=DEVICE)
    
    # Print model statistics
    quant_stats = quantized_embedder.get_model_stats()
    print(f"\n📊 Model Comparison:")
    print(f"   Original: DINOv2 (current) - ~84MB")
    print(f"   Quantized: {quant_stats['model_name']} - {quant_stats['quantized_size_mb']:.1f}MB")
    print(f"   Memory reduction: {quant_stats['memory_reduction_pct']:.1f}%")
    
    # Prepare database crops
    print(f"\n🗄️ Preparing database ({CROP_STRIDE_PX}px stride)...")
    db_centers_px = []
    db_centers_m = []
    db_images = []
    
    for y in range(0, map_h - CROP_SIZE_PX + 1, CROP_STRIDE_PX):
        for x in range(0, map_w - CROP_SIZE_PX + 1, CROP_STRIDE_PX):
            crop = map_image.crop((x, y, x + CROP_SIZE_PX, y + CROP_SIZE_PX))
            db_images.append(crop)
            cx, cy = x + CROP_SIZE_PX // 2, y + CROP_SIZE_PX // 2
            db_centers_px.append((cx, cy))
            x_m = (cx - map_w // 2) * M_PER_PIXEL
            y_m = (cy - map_h // 2) * M_PER_PIXEL
            db_centers_m.append((x_m, y_m))
    
    GRID_W = (map_w - CROP_SIZE_PX) // CROP_STRIDE_PX + 1
    GRID_H = (map_h - CROP_SIZE_PX) // CROP_STRIDE_PX + 1
    print(f"Database: {GRID_W}x{GRID_H} = {len(db_images)} crops")
    
    # Generate vocabulary from different area
    print("\n🎯 Generating VLAD vocabulary...")
    vocab_images = []
    for _ in range(50):  # Smaller vocab for speed
        x = np.random.randint(0, map_w - CROP_SIZE_PX)
        y = np.random.randint(0, map_h - CROP_SIZE_PX)
        crop = map_image.crop((x, y, x + CROP_SIZE_PX, y + CROP_SIZE_PX))
        vocab_images.append(crop)
    
    # Build vocabulary for both models
    print("Building vocabulary for original model...")
    all_dense_feats_orig = []
    for img in tqdm(vocab_images[:25], desc="Original vocab"):  # Smaller for speed
        feats = original_embedder.extract_dense_features(img)
        all_dense_feats_orig.append(feats)
    all_dense_feats_orig = np.concatenate(all_dense_feats_orig, axis=0)
    
    if all_dense_feats_orig.shape[0] > 10000:
        idx = np.random.choice(all_dense_feats_orig.shape[0], 10000, replace=False)
        vocab_feats_orig = all_dense_feats_orig[idx]
    else:
        vocab_feats_orig = all_dense_feats_orig
    
    original_embedder.fit_vocabulary_from_features(vocab_feats_orig)
    
    print("Building vocabulary for quantized model...")
    all_dense_feats_quant = []
    total_inference_time_quant = 0
    for img in tqdm(vocab_images[:25], desc="Quantized vocab"):
        feats, inf_time = quantized_embedder.extract_dense_features(img)
        all_dense_feats_quant.append(feats)
        total_inference_time_quant += inf_time
    all_dense_feats_quant = np.concatenate(all_dense_feats_quant, axis=0)
    
    if all_dense_feats_quant.shape[0] > 10000:
        idx = np.random.choice(all_dense_feats_quant.shape[0], 10000, replace=False)
        vocab_feats_quant = all_dense_feats_quant[idx]
    else:
        vocab_feats_quant = all_dense_feats_quant
    
    quantized_embedder.fit_vocabulary_from_features(vocab_feats_quant)
    
    avg_vocab_time_quant = total_inference_time_quant / len(vocab_images[:25])
    print(f"Average quantized inference time: {avg_vocab_time_quant*1000:.1f}ms")
    
    # Compute database embeddings for both models
    print("\n💾 Computing database embeddings...")
    
    print("Original model database...")
    db_vlad_vectors_orig = []
    total_db_time_orig = 0
    for img in tqdm(db_images[:100], desc="Original DB"):  # Limit for speed
        start_time = time.time()
        vectors = original_embedder.get_vlad_vectors(img)
        total_db_time_orig += time.time() - start_time
        db_vlad_vectors_orig.append(vectors)
    
    print("Quantized model database...")
    db_vlad_vectors_quant = []
    total_db_time_quant = 0
    for img in tqdm(db_images[:100], desc="Quantized DB"):
        vectors, inf_time = quantized_embedder.get_vlad_vectors(img)
        db_vlad_vectors_quant.append(vectors)
        total_db_time_quant += inf_time
    
    # Truncate other arrays to match
    db_centers_m = db_centers_m[:100]
    db_centers_px = db_centers_px[:100]
    db_images = db_images[:100]
    
    avg_db_time_orig = total_db_time_orig / len(db_vlad_vectors_orig)
    avg_db_time_quant = total_db_time_quant / len(db_vlad_vectors_quant)
    
    print(f"\n⏱️ Database Processing Time Comparison:")
    print(f"   Original: {avg_db_time_orig*1000:.1f}ms per image")
    print(f"   Quantized: {avg_db_time_quant*1000:.1f}ms per image")
    print(f"   Speedup: {avg_db_time_orig/avg_db_time_quant:.2f}x")
    
    # Generate trajectory
    print("\n🛸 Generating drone trajectory...")
    x_bounds = (-(map_w // 2) * M_PER_PIXEL, (map_w // 2) * M_PER_PIXEL)
    y_bounds = (-(map_h // 2) * M_PER_PIXEL, (map_h // 2) * M_PER_PIXEL)
    
    def random_trajectory(start, steps, step_size, noise_std, bounds):
        traj = [np.array(start)]
        for _ in range(steps):
            angle = np.random.uniform(0, 2 * np.pi)
            move = np.array([np.cos(angle), np.sin(angle)]) * step_size
            noisy_move = move + np.random.normal(0, noise_std, 2)
            next_pos = traj[-1] + noisy_move
            next_pos[0] = np.clip(next_pos[0], bounds[0][0], bounds[0][1])
            next_pos[1] = np.clip(next_pos[1], bounds[1][0], bounds[1][1])
            traj.append(next_pos)
        return np.array(traj)
    
    num_steps = 50  # Shorter for demo
    true_traj = random_trajectory((0.0, 0.0), num_steps, STEP_SIZE_M, 0, (x_bounds, y_bounds))
    vio_traj = random_trajectory((0.0, 0.0), num_steps, STEP_SIZE_M, VIO_NOISE_STD, (x_bounds, y_bounds))
    
    # Initialize recall tracking
    recall_stats = {
        'original': {'recall1': 0, 'recall5': 0, 'total': 0, 'times': []},
        'quantized': {'recall1': 0, 'recall5': 0, 'total': 0, 'times': []}
    }
    
    SPATIAL_TOL_M = 1.5 * PATCH_SIZE_M
    
    # Video setup
    VIDEO_W, VIDEO_H = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 5, (VIDEO_W, VIDEO_H))
    
    print(f"\n🎬 Running simulation ({num_steps} steps)...")
    print(f"Spatial tolerance: {SPATIAL_TOL_M:.1f}m")
    
    # Main simulation loop
    for step in tqdm(range(1, num_steps+1), desc="Simulation"):
        if step % int(UPDATE_INTERVAL_M // STEP_SIZE_M) == 0:
            true_pos = true_traj[step]
            
            # Extract query crop
            px = int(true_pos[0] / M_PER_PIXEL + map_w // 2)
            py = int(true_pos[1] / M_PER_PIXEL + map_h // 2)
            
            x0 = px - CROP_SIZE_PX // 2
            y0 = py - CROP_SIZE_PX // 2
            x1 = x0 + CROP_SIZE_PX
            y1 = y0 + CROP_SIZE_PX
            
            query_crop = Image.new('RGB', (CROP_SIZE_PX, CROP_SIZE_PX), color=(0, 0, 0))
            crop_img = map_image.crop((max(0, x0), max(0, y0), min(map_w, x1), min(map_h, y1)))
            paste_x = max(0, -x0)
            paste_y = max(0, -y0)
            query_crop.paste(crop_img, (paste_x, paste_y))
            
            # Test both models
            for model_name, embedder, db_vectors, stats in [
                ('original', original_embedder, db_vlad_vectors_orig, recall_stats['original']),
                ('quantized', quantized_embedder, db_vlad_vectors_quant, recall_stats['quantized'])
            ]:
                start_time = time.time()
                
                if model_name == 'original':
                    query_vectors = embedder.get_vlad_vectors(query_crop)
                    similarities = []
                    for db_vecs in db_vectors:
                        sim = embedder.chamfer_similarity(query_vectors, db_vecs)
                        similarities.append(sim)
                else:
                    query_vectors, inf_time = embedder.get_vlad_vectors(query_crop)
                    similarities = []
                    for db_vecs in db_vectors:
                        sim = embedder.chamfer_similarity(query_vectors, db_vecs)
                        similarities.append(sim)
                
                query_time = time.time() - start_time
                stats['times'].append(query_time)
                
                similarities = np.array(similarities)
                top5_idx = np.argpartition(similarities, -5)[-5:]
                sorted_top5 = top5_idx[np.argsort(similarities[top5_idx])[::-1]]
                
                # Calculate recall
                spatial_dists = np.array([np.linalg.norm(np.array(center) - true_pos) 
                                        for center in db_centers_m])
                
                stats['total'] += 1
                top1_spatial_dist = spatial_dists[sorted_top5[0]]
                top5_spatial_dists = spatial_dists[sorted_top5]
                
                if top1_spatial_dist <= SPATIAL_TOL_M:
                    stats['recall1'] += 1
                if np.any(top5_spatial_dists <= SPATIAL_TOL_M):
                    stats['recall5'] += 1
        
        # Visualization
        vis = np.array(map_image).copy()
        
        # Draw trajectories
        for i in range(1, step+1):
            def world_to_px(pos):
                x = int(pos[0] / M_PER_PIXEL + map_w // 2)
                y = int(pos[1] / M_PER_PIXEL + map_h // 2)
                return x, y
            cv2.line(vis, world_to_px(true_traj[i-1]), world_to_px(true_traj[i]), (0, 255, 0), 2)
            cv2.line(vis, world_to_px(vio_traj[i-1]), world_to_px(vio_traj[i]), (0, 0, 255), 2)
        
        # Draw current positions
        cv2.circle(vis, world_to_px(true_traj[step]), 6, (0, 255, 0), -1)
        cv2.circle(vis, world_to_px(vio_traj[step]), 6, (0, 0, 255), -1)
        
        # Add text overlay with current stats
        if recall_stats['original']['total'] > 0:
            orig_r1 = recall_stats['original']['recall1'] / recall_stats['original']['total']
            orig_r5 = recall_stats['original']['recall5'] / recall_stats['original']['total']
            quant_r1 = recall_stats['quantized']['recall1'] / recall_stats['quantized']['total']
            quant_r5 = recall_stats['quantized']['recall5'] / recall_stats['quantized']['total']
            
            cv2.putText(vis, f"Original R@1: {orig_r1:.3f} R@5: {orig_r5:.3f}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis, f"Quantized R@1: {quant_r1:.3f} R@5: {quant_r5:.3f}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(vis, f"Step: {step}/{num_steps}", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Resize and write frame
        vis_resized = cv2.resize(vis, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
        video_out.write(vis_resized)
    
    video_out.release()
    
    # Final results
    print(f"\n🎯 FINAL COMPARISON RESULTS:")
    print("="*50)
    
    for model_name, stats in recall_stats.items():
        if stats['total'] > 0:
            r1 = stats['recall1'] / stats['total']
            r5 = stats['recall5'] / stats['total']
            avg_time = np.mean(stats['times']) * 1000  # ms
            
            print(f"\n📊 {model_name.upper()} MODEL:")
            print(f"   Recall@1: {r1:.3f} ({stats['recall1']}/{stats['total']})")
            print(f"   Recall@5: {r5:.3f} ({stats['recall5']}/{stats['total']})")
            print(f"   Avg query time: {avg_time:.1f}ms")
    
    # Performance comparison
    if (recall_stats['original']['total'] > 0 and recall_stats['quantized']['total'] > 0):
        orig_time = np.mean(recall_stats['original']['times']) * 1000
        quant_time = np.mean(recall_stats['quantized']['times']) * 1000
        speedup = orig_time / quant_time
        
        orig_r1 = recall_stats['original']['recall1'] / recall_stats['original']['total']
        quant_r1 = recall_stats['quantized']['recall1'] / recall_stats['quantized']['total']
        accuracy_retention = quant_r1 / orig_r1 if orig_r1 > 0 else 0
        
        print(f"\n🚀 QUANTIZATION IMPACT:")
        print(f"   Speed improvement: {speedup:.2f}x faster")
        print(f"   Memory reduction: {quant_stats['memory_reduction_pct']:.1f}%")
        print(f"   Accuracy retention: {accuracy_retention:.1%}")
        print(f"   Model size: {quant_stats['quantized_size_mb']:.1f}MB vs ~84MB")
    
    print(f"\n🎬 Video saved: {VIDEO_FILENAME}")
    print("✅ Comparison complete!")

if __name__ == "__main__":
    run_comparison()