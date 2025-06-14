import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class SuperPointNet(nn.Module):
    """SuperPoint network architecture"""
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            conv_bn_relu(1, 64),  # Conv1a
            conv_bn_relu(64, 64),  # Conv1b
            nn.MaxPool2d(2, 2),
            conv_bn_relu(64, 64),  # Conv2a
            conv_bn_relu(64, 64),  # Conv2b
            nn.MaxPool2d(2, 2),
            conv_bn_relu(64, 128),  # Conv3a
            conv_bn_relu(128, 128),  # Conv3b
            nn.MaxPool2d(2, 2),
            conv_bn_relu(128, 128),  # Conv4a
            conv_bn_relu(128, 128)   # Conv4b
        )
        
        # Detector head
        self.detector = nn.Sequential(
            conv_bn_relu(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)  # 65 = 64 cells + 1 no keypoint
        )
        
        # Descriptor head
        self.descriptor = nn.Sequential(
            conv_bn_relu(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # 256-dim descriptors
        )
        
    def forward(self, x, labels=None):
        """
        Forward pass of SuperPoint network.
        
        Args:
            x: Input image tensor (B,1,H,W)
            labels: Optional keypoint map labels (B,H/8,W/8)
            
        Returns:
            Dictionary containing:
            - keypoint_scores: (B,65,H/8,W/8) raw scores for keypoint detection
            - descriptors: (B,256,H/8,W/8) descriptor maps
            - loss: Training loss if labels provided
        """
        # Shared encoder
        features = self.encoder(x)
        
        # Detector head
        keypoint_scores = self.detector(features)  # (B,65,H/8,W/8)
        
        # Descriptor head
        descriptors = self.descriptor(features)  # (B,256,H/8,W/8)
        # L2 normalize descriptors, but avoid normalizing zero vectors
        descriptor_norms = torch.norm(descriptors, p=2, dim=1, keepdim=True)
        descriptors = descriptors / (descriptor_norms + 1e-8)  # Add small epsilon to avoid division by zero
        
        outputs = {
            'keypoint_scores': keypoint_scores,
            'descriptors': descriptors
        }
        
        if labels is not None:
            # Compute detector loss
            B, H, W = labels.shape
            labels = labels.long()  # Convert to long for cross entropy
            
            # Reshape predictions and labels for loss computation
            pred = keypoint_scores.permute(0, 2, 3, 1).reshape(-1, 65)  # (B*H*W, 65)
            target = labels.reshape(-1)  # (B*H*W)
            
            # Binary classification loss for each cell
            keypoint_loss = F.cross_entropy(pred, target)
            
            # Add loss to outputs
            outputs['loss'] = keypoint_loss
            
        return outputs

    def detect_keypoints(self, scores, threshold=0.015, nms_dist=4):
        """
        Post-process keypoint scores to get keypoint locations and descriptors.
        
        Args:
            scores: Raw keypoint scores (B,65,H,W)
            threshold: Detection threshold
            nms_dist: Non-maxima suppression distance
            
        Returns:
            keypoints: List of (N,2) keypoint arrays
            scores: List of (N,) score arrays
        """
        # Convert to probability distribution
        prob = F.softmax(scores, dim=1)
        
        # Get keypoint location probabilities (removing no-keypoint)
        prob = prob[:,:-1,:,:]  # (B,64,H,W)
        
        batch_keypoints = []
        batch_scores = []
        
        B, C, H, W = prob.shape
        for b in range(B):
            prob_np = prob[b].detach().cpu().numpy()
            
            # Simple non-maximum suppression
            keypoints = []
            kp_scores = []
            
            for i in range(H):
                for j in range(W):
                    score = np.max(prob_np[:,i,j])
                    if score > threshold:
                        # Check if maximum in nms_dist x nms_dist neighborhood
                        left = max(0, j-nms_dist)
                        right = min(W, j+nms_dist+1)
                        top = max(0, i-nms_dist)
                        bottom = min(H, i+nms_dist+1)
                        
                        patch = prob_np[:,top:bottom,left:right]
                        if score == np.max(patch):
                            # Convert back to original image coordinates
                            keypoints.append([j*8 + 4, i*8 + 4])  # +4 for center of cell
                            kp_scores.append(score)
            
            batch_keypoints.append(np.array(keypoints))
            batch_scores.append(np.array(kp_scores))
            
        return batch_keypoints, batch_scores

class SuperPoint(nn.Module):
    """Wrapper class for SuperPoint model with both training and inference modes"""
    def __init__(self, pretrained_path=None, device='cuda'):
        super().__init__()
        self.device = device
        self.net = SuperPointNet().to(device)
        
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            # Handle both direct state dict and checkpoint format
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Try to load state dict as is first
            try:
                self.net.load_state_dict(state_dict)
            except:
                # If that fails, try removing 'net.' prefix if it exists
                if all(k.startswith('net.') for k in state_dict.keys()):
                    state_dict = {k[4:]: v for k, v in state_dict.items()}
                    self.net.load_state_dict(state_dict)
                # If that fails too, try adding 'net.' prefix
                else:
                    state_dict = {f'net.{k}': v for k, v in state_dict.items()}
                    self.load_state_dict(state_dict)
            
    def train(self, mode=True):
        self.net.train(mode)
        return self
        
    def eval(self):
        self.net.eval()
        return self
        
    def detect(self, image):
        """
        Detect keypoints in a single image.
        
        Args:
            image: uint8 image array (H,W) or (H,W,1/3)
            
        Returns:
            keypoints: (N,2) array of keypoint coordinates
            scores: (N,) array of keypoint scores
            descriptors: (N,256) array of L2-normalized descriptors
        """
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[...,0]
            
        # Normalize and convert to tensor
        image = torch.from_numpy(image).float() / 255.0
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        
        # Run inference
        with torch.no_grad():
            self.eval()
            outputs = self.net(image)
            
            # Get keypoint detections
            keypoints, scores = self.net.detect_keypoints(outputs['keypoint_scores'])
            
            # Get descriptors for keypoints
            if len(keypoints[0]) > 0:
                kp = torch.from_numpy(keypoints[0]).float().to(self.device)
                desc_map = outputs['descriptors'][0]  # (256,H/8,W/8)
                
                # Sample descriptors at keypoint locations
                kp_scaled = kp / 8  # Scale to feature map resolution
                H_feat, W_feat = desc_map.shape[1], desc_map.shape[2]
                
                # Normalize coordinates for grid_sample (from [0, H/8] to [-1, 1])
                kp_norm = kp_scaled.clone()
                kp_norm[:, 0] = 2.0 * kp_scaled[:, 0] / (W_feat - 1) - 1.0  # x coordinate
                kp_norm[:, 1] = 2.0 * kp_scaled[:, 1] / (H_feat - 1) - 1.0  # y coordinate
                
                # grid_sample expects (x,y) format
                kp_grid = kp_norm[:, [0, 1]].view(1, -1, 1, 2)  # (1, N, 1, 2)
                
                descriptors = F.grid_sample(
                    desc_map.unsqueeze(0),
                    kp_grid,
                    mode='bilinear',
                    align_corners=True
                )
                descriptors = descriptors.squeeze().t()  # (N,256)
            else:
                descriptors = torch.zeros((0, 256), device=self.device)
            
            return keypoints[0], scores[0], descriptors.cpu().numpy()
            
    def forward(self, x, labels=None):
        return self.net(x, labels) 