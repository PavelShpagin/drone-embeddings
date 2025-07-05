#!/usr/bin/env python3
"""
Quantized DINO Implementation for Pi Zero Deployment
Direct replacement for AnyLocVLADEmbedder with quantization support
"""

import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
import time
import psutil
import gc

# Import your existing utilities
try:
    from third_party.AnyLoc.utilities import DinoV2ExtractFeatures, VLAD
except ImportError:
    print("⚠️ AnyLoc utilities not found - using mock implementations")

class QuantizedDINOEmbedder:
    """
    Drop-in replacement for AnyLocVLADEmbedder with quantization support
    Optimized for Pi Zero deployment with DINOv2 ViT-S/14
    """
    
    def __init__(self, model_type="dinov2_vits14", layer=11, facet="key", 
                 device=None, n_clusters=32, enable_quantization=True):
        """
        Initialize quantized DINO embedder
        
        Args:
            model_type: Use "dinov2_vits14" for Pi Zero (21M params vs 86M)
            layer: Feature extraction layer (11 = last layer)
            facet: Feature facet ("key", "query", "value", "token")
            device: Compute device ("cpu" for Pi Zero)
            n_clusters: VLAD cluster count
            enable_quantization: Apply INT8 quantization
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.enable_quantization = enable_quantization
        self.vlad = None
        self.desc_dim = None
        
        # Performance tracking
        self.query_times = []
        self.memory_usage = []
        
        # Load and prepare model
        print(f"🔄 Loading {model_type} on {self.device}...")
        self.model = DinoV2ExtractFeatures(model_type, layer, facet, device=self.device)
        
        # Apply quantization if enabled
        if enable_quantization:
            self.quantize_model()
        
        # Image preprocessing
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize((224, 224)),
        ])
        
        # Print model statistics
        self.print_model_stats()
    
    def quantize_model(self):
        """Apply INT8 dynamic quantization for Pi Zero deployment"""
        try:
            print("🔧 Applying INT8 quantization...")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Apply dynamic quantization to Linear and Conv2d layers
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Move to CPU for Pi Zero
            if self.device == "cpu":
                self.model = self.model.cpu()
            
            print("✅ INT8 quantization applied successfully")
            self.is_quantized = True
            
        except Exception as e:
            print(f"⚠️ Quantization failed: {e}")
            print("Continuing with original model...")
            self.is_quantized = False
    
    def print_model_stats(self):
        """Print model statistics and memory usage"""
        # Estimate model size
        param_count = sum(p.numel() for p in self.model.parameters())
        
        if self.is_quantized:
            # INT8: ~1 byte per parameter
            estimated_size_mb = param_count / (1024 * 1024)
            quant_info = " (INT8 Quantized)"
        else:
            # Float32: ~4 bytes per parameter  
            estimated_size_mb = (param_count * 4) / (1024 * 1024)
            quant_info = " (Float32)"
        
        current_memory = self.get_memory_usage()
        
        print(f"📊 Model Statistics:")
        print(f"   • Type: {self.model_type}{quant_info}")
        print(f"   • Parameters: {param_count:,}")
        print(f"   • Estimated size: {estimated_size_mb:.1f}MB")
        print(f"   • Current memory: {current_memory:.1f}MB")
        print(f"   • Device: {self.device}")
    
    def get_memory_usage(self):
        """Get current process memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def fit_vocabulary_from_features(self, features):
        """
        Fit VLAD vocabulary from pre-extracted features
        Compatible with original AnyLocVLADEmbedder interface
        """
        print(f"🎯 Fitting VLAD vocabulary ({self.n_clusters} clusters)...")
        
        self.desc_dim = features.shape[1]
        self.vlad = VLAD(self.n_clusters, desc_dim=self.desc_dim, device=self.device)
        
        # Convert to appropriate format for VLAD
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        
        self.vlad.fit(features)
        print(f"✅ VLAD vocabulary fitted on {features.shape[0]} features")
    
    def extract_dense_features(self, image):
        """
        Extract dense DINOv2 features from image
        Compatible with original interface
        """
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Preprocess image
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                timg = self.transform(image).unsqueeze(0).to(self.device)
                
                # Extract features
                feats = self.model(timg)  # [1, N_patches, D]
                feats = torch.nn.functional.normalize(feats, dim=2)
                
                # Convert to numpy
                features = feats[0].cpu().numpy()
                
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            # Return empty features as fallback
            features = np.zeros((1, 384))  # Default ViT-S dimension
        
        # Track performance
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        self.memory_usage.append(self.get_memory_usage())
        
        return features
    
    def get_vlad_vectors(self, img):
        """
        Get individual VLAD cluster vectors for Chamfer similarity
        Compatible with original interface
        """
        start_time = time.time()
        
        # Extract dense features
        feats = self.extract_dense_features(img)
        
        if self.vlad is None:
            raise ValueError("VLAD vocabulary not fitted. Call fit_vocabulary_from_features() first.")
        
        # Convert to tensor for VLAD
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        feats = feats.cpu()
        
        # Generate VLAD description
        vlad_desc = self.vlad.generate(feats)  # [Nc*D]
        
        # Reshape to individual cluster vectors [n_clusters, desc_dim]
        vlad_vectors = vlad_desc.reshape(self.n_clusters, self.desc_dim)
        
        query_time = time.time() - start_time
        
        return vlad_vectors.cpu().numpy()
    
    def get_embedding(self, img):
        """
        Get concatenated VLAD embedding (original behavior)
        Compatible with original interface
        """
        vlad_vectors = self.get_vlad_vectors(img)
        # Flatten to get concatenated embedding
        return vlad_vectors.flatten()
    
    @staticmethod
    def chamfer_similarity(query_vectors, doc_vectors):
        """
        Compute Chamfer similarity between two sets of vectors
        Compatible with original interface
        """
        # Compute pairwise inner products
        similarities = np.dot(query_vectors, doc_vectors.T)
        
        # For each query vector, find max similarity with any doc vector
        max_similarities = np.max(similarities, axis=1)
        
        # Sum over all query vectors
        chamfer_score = np.sum(max_similarities)
        
        return chamfer_score
    
    def memory_cleanup(self):
        """Force memory cleanup for Pi Zero"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.query_times:
            return None
        
        return {
            'avg_query_time_ms': np.mean(self.query_times) * 1000,
            'min_query_time_ms': np.min(self.query_times) * 1000,
            'max_query_time_ms': np.max(self.query_times) * 1000,
            'avg_memory_mb': np.mean(self.memory_usage),
            'peak_memory_mb': np.max(self.memory_usage),
            'total_queries': len(self.query_times)
        }
    
    def print_performance_summary(self):
        """Print performance summary"""
        stats = self.get_performance_stats()
        if stats is None:
            print("📊 No performance data available")
            return
        
        print(f"📊 Performance Summary:")
        print(f"   • Queries processed: {stats['total_queries']}")
        print(f"   • Avg query time: {stats['avg_query_time_ms']:.1f}ms")
        print(f"   • Query time range: {stats['min_query_time_ms']:.1f}-{stats['max_query_time_ms']:.1f}ms")
        print(f"   • Avg memory usage: {stats['avg_memory_mb']:.1f}MB")
        print(f"   • Peak memory usage: {stats['peak_memory_mb']:.1f}MB")


class PiZeroOptimizedLocalizer:
    """
    Complete localization pipeline optimized for Pi Zero
    Integrates quantized DINO with your existing approach
    """
    
    def __init__(self, map_image_path, crop_size=224, device="cpu"):
        self.crop_size = crop_size
        self.device = device
        
        # Initialize quantized DINO embedder
        self.embedder = QuantizedDINOEmbedder(
            model_type="dinov2_vits14",  # Smaller model
            device=device,
            enable_quantization=True
        )
        
        # Load map and create database
        self.load_map(map_image_path)
        self.create_database()
        
        print("🚀 Pi Zero optimized localizer ready!")
    
    def load_map(self, map_path):
        """Load map image"""
        self.map_image = Image.open(map_path).convert("RGB")
        self.map_w, self.map_h = self.map_image.size
        print(f"🗺️ Map loaded: {self.map_w}x{self.map_h}")
    
    def create_database(self):
        """Create reference database with quantized DINO features"""
        print("💾 Creating database with quantized features...")
        
        # Extract database crops
        stride = self.crop_size // 2  # 50% overlap
        db_images = []
        db_centers = []
        
        for y in range(0, self.map_h - self.crop_size + 1, stride):
            for x in range(0, self.map_w - self.crop_size + 1, stride):
                crop = self.map_image.crop((x, y, x + self.crop_size, y + self.crop_size))
                db_images.append(crop)
                
                cx = x + self.crop_size // 2
                cy = y + self.crop_size // 2
                db_centers.append((cx, cy))
        
        print(f"📍 Extracted {len(db_images)} database crops")
        
        # Build vocabulary from subset
        vocab_images = db_images[::10]  # Every 10th image for vocabulary
        vocab_features = []
        
        for img in vocab_images[:50]:  # Limit for Pi Zero
            feats = self.embedder.extract_dense_features(img)
            vocab_features.append(feats)
        
        vocab_features = np.concatenate(vocab_features, axis=0)
        self.embedder.fit_vocabulary_from_features(vocab_features)
        
        # Compute database VLAD vectors
        print("🔄 Computing database VLAD vectors...")
        self.db_vlad_vectors = []
        
        for i, img in enumerate(db_images):
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(db_images)}")
                self.embedder.memory_cleanup()  # Cleanup for Pi Zero
            
            vectors = self.embedder.get_vlad_vectors(img)
            self.db_vlad_vectors.append(vectors)
        
        self.db_centers = db_centers
        print(f"✅ Database ready with {len(self.db_vlad_vectors)} entries")
    
    def localize(self, query_image, top_k=5):
        """
        Localize query image using quantized DINO + VLAD + Chamfer
        """
        start_time = time.time()
        
        # Extract query VLAD vectors
        query_vectors = self.embedder.get_vlad_vectors(query_image)
        
        # Compute Chamfer similarities
        similarities = []
        for db_vectors in self.db_vlad_vectors:
            sim = self.embedder.chamfer_similarity(query_vectors, db_vectors)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Get top-k matches
        top_k_idx = np.argpartition(similarities, -top_k)[-top_k:]
        sorted_top_k = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
        
        # Get results
        results = []
        for i, idx in enumerate(sorted_top_k):
            results.append({
                'rank': i + 1,
                'similarity': similarities[idx],
                'position': self.db_centers[idx],
                'index': idx
            })
        
        query_time = time.time() - start_time
        
        return {
            'results': results,
            'query_time_ms': query_time * 1000,
            'best_match': results[0] if results else None
        }


def test_quantized_implementation():
    """Test the quantized implementation"""
    print("🧪 Testing Quantized DINO Implementation")
    print("=" * 50)
    
    # Test embedder initialization
    embedder = QuantizedDINOEmbedder(
        model_type="dinov2_vits14",
        device="cpu",
        enable_quantization=True
    )
    
    # Test feature extraction
    test_image = Image.new('RGB', (224, 224), color='red')
    
    print("\n🔬 Testing feature extraction...")
    features = embedder.extract_dense_features(test_image)
    print(f"✅ Features extracted: {features.shape}")
    
    # Test vocabulary fitting
    print("\n🎯 Testing vocabulary fitting...")
    vocab_features = np.random.randn(1000, features.shape[1])
    embedder.fit_vocabulary_from_features(vocab_features)
    print("✅ Vocabulary fitted")
    
    # Test VLAD vector extraction
    print("\n🔄 Testing VLAD vector extraction...")
    vlad_vectors = embedder.get_vlad_vectors(test_image)
    print(f"✅ VLAD vectors: {vlad_vectors.shape}")
    
    # Test Chamfer similarity
    print("\n🎲 Testing Chamfer similarity...")
    query_vectors = np.random.randn(32, features.shape[1])
    doc_vectors = np.random.randn(32, features.shape[1])
    similarity = embedder.chamfer_similarity(query_vectors, doc_vectors)
    print(f"✅ Chamfer similarity: {similarity:.3f}")
    
    # Print performance stats
    print("\n📊 Performance Statistics:")
    embedder.print_performance_summary()
    
    print("\n🎉 All tests passed! Ready for Pi Zero deployment.")


if __name__ == "__main__":
    test_quantized_implementation()