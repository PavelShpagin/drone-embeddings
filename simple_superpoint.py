#!/usr/bin/env python3
"""
Simple SuperPoint implementation compatible with proven pretrained weights.
Based on: https://github.com/shaofengzeng/SuperPoint-Pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path

class SuperPointNet(nn.Module):
    """SuperPoint network architecture compatible with pretrained weights."""
    
    def __init__(self):
        super().__init__()
        
        # Shared encoder
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder layers
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Detector head
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor head  
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """Forward pass."""
        # Shared encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)  # [B, 65, H/8, W/8]
        
        # Descriptor head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)  # [B, 256, H/8, W/8]
        
        # FIXED: Proper descriptor normalization using F.normalize
        desc = F.normalize(desc, p=2, dim=1)  # L2 normalize along channel dimension
        
        return semi, desc

class SuperPoint:
    """SuperPoint wrapper for easy inference."""
    
    def __init__(self, weights_path=None, device='cuda'):
        self.device = device
        self.net = SuperPointNet().to(device)
        
        if weights_path and Path(weights_path).exists():
            print(f"Loading SuperPoint weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            self.net.load_state_dict(state_dict)
            print("✓ SuperPoint weights loaded successfully")
        else:
            print("⚠️ No weights loaded - using random initialization")
            
        self.net.eval()
        
    def detect(self, image, conf_thresh=0.015, nms_dist=4, cell=8):
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: Grayscale image (H, W) or (H, W, 1)
            conf_thresh: Confidence threshold for keypoint detection
            nms_dist: Non-maximum suppression distance
            cell: Cell size (8 for SuperPoint)
            
        Returns:
            keypoints: (N, 2) array of keypoint coordinates
            scores: (N,) array of keypoint confidence scores  
            descriptors: (N, 256) array of descriptors
        """
        if image.ndim == 3:
            image = image[:, :, 0]  # Take first channel if RGB
            
        H, W = image.shape
        
        # Prepare input tensor
        inp = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            semi, desc = self.net(inp)
            
        # Convert outputs to numpy
        semi = semi.squeeze().cpu().numpy()  # [65, H/8, W/8]
        desc = desc.squeeze().cpu().numpy()  # [256, H/8, W/8]
        
        # Extract keypoints
        dense = np.exp(semi)  # Softmax
        dense = dense / (np.sum(dense, axis=0) + 1e-6)  # Normalize
        
        # Remove dustbin (no-keypoint class)
        nodust = dense[:-1, :, :]  # [64, H/8, W/8]
        
        # Reshape to get keypoint probabilities per cell
        Hc, Wc = nodust.shape[1], nodust.shape[2]
        nodust = nodust.transpose(1, 2, 0)  # [H/8, W/8, 64]
        heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])  # [H, W]
        
        # Find keypoints above threshold
        xs, ys = np.where(heatmap >= conf_thresh)
        if len(xs) == 0:
            return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 256))
            
        pts = np.zeros((len(xs), 2))
        pts[:, 0] = ys  # x coordinates
        pts[:, 1] = xs  # y coordinates
        scores = heatmap[xs, ys]
        
        # Apply NMS
        if nms_dist > 0:
            keep = self._nms(pts, scores, nms_dist)
            pts = pts[keep]
            scores = scores[keep]
        
        # Extract descriptors at keypoint locations
        if len(pts) > 0:
            # Convert keypoint coordinates to descriptor map coordinates
            samp_pts = pts / cell  # Scale to descriptor map resolution
            samp_pts[:, [0, 1]] = samp_pts[:, [1, 0]]  # Swap x,y to y,x for indexing
            
            # Bilinear interpolation to get descriptors
            descriptors = self._sample_descriptors(desc, samp_pts)
        else:
            descriptors = np.zeros((0, 256))
            
        return pts, scores, descriptors
    
    def _nms(self, pts, scores, nms_dist):
        """Non-maximum suppression."""
        if len(pts) == 0:
            return []
            
        # Sort by score (descending)
        order = np.argsort(scores)[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
                
            # Compute distances to remaining points
            dists = np.sqrt(np.sum((pts[order[1:]] - pts[i]) ** 2, axis=1))
            
            # Keep points that are far enough
            inds = np.where(dists >= nms_dist)[0]
            order = order[inds + 1]
            
        return keep
    
    def _sample_descriptors(self, desc, pts):
        """Sample descriptors at given points using bilinear interpolation."""
        H, W = desc.shape[1], desc.shape[2]
        descriptors = []
        
        for pt in pts:
            y, x = pt
            
            # Get integer coordinates
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1
            
            # Clamp to valid range
            x0 = max(0, min(x0, W - 1))
            x1 = max(0, min(x1, W - 1))
            y0 = max(0, min(y0, H - 1))
            y1 = max(0, min(y1, H - 1))
            
            # Get fractional parts
            dx = x - x0
            dy = y - y0
            
            # Bilinear interpolation
            d00 = desc[:, y0, x0]
            d01 = desc[:, y1, x0]
            d10 = desc[:, y0, x1]
            d11 = desc[:, y1, x1]
            
            d = (1 - dx) * (1 - dy) * d00 + dx * (1 - dy) * d10 + (1 - dx) * dy * d01 + dx * dy * d11
            
            # Normalize
            d = d / (np.linalg.norm(d) + 1e-8)
            descriptors.append(d)
            
        return np.array(descriptors) if descriptors else np.zeros((0, 256))

def test_superpoint():
    """Test SuperPoint with a sample image."""
    # Try to find pretrained weights
    weight_paths = [
        "pretrained_weights/superpoint_v1.pth",
        "pretrained_weights/superpoint_pytorch.pth",
        "pretrained_weights/superpoint_pretrained.pth"
    ]
    
    weights_path = None
    for path in weight_paths:
        if Path(path).exists():
            weights_path = path
            break
    
    if weights_path is None:
        print("No pretrained weights found. Run download_pretrained_superpoint.py first.")
        return
        
    # Load SuperPoint
    sp = SuperPoint(weights_path)
    
    # Test with a sample image
    test_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    keypoints, scores, descriptors = sp.detect(test_img)
    
    print(f"Detected {len(keypoints)} keypoints")
    print(f"Descriptor shape: {descriptors.shape}")
    print("✓ SuperPoint test successful!")

if __name__ == "__main__":
    test_superpoint() 