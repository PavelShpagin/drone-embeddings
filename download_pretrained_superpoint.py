#!/usr/bin/env python3
"""
Download pretrained SuperPoint weights from the proven PyTorch implementation.
Source: https://github.com/shaofengzeng/SuperPoint-Pytorch
"""

import os
import requests
import torch
from pathlib import Path

def download_file(url, filepath):
    """Download a file from URL to filepath with progress."""
    print(f"Downloading {url}")
    print(f"Saving to {filepath}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    print(f"\n✓ Downloaded successfully: {filepath}")

def download_pretrained_superpoint():
    """Download pretrained SuperPoint weights."""
    
    # Create directory for pretrained weights
    weights_dir = Path("pretrained_weights")
    weights_dir.mkdir(exist_ok=True)
    
    # URLs for pretrained weights (these are common locations for SuperPoint weights)
    urls = [
        {
            "name": "superpoint_v1.pth",
            "url": "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth",
            "description": "Original MagicLeap SuperPoint weights"
        },
        # Alternative sources if the above doesn't work
        {
            "name": "superpoint_pytorch.pth", 
            "url": "https://drive.google.com/uc?id=1hieWKvyAWyffHVgcSg2RugQDcsQHqvg&export=download",
            "description": "PyTorch SuperPoint implementation weights"
        }
    ]
    
    success = False
    
    for weight_info in urls:
        filepath = weights_dir / weight_info["name"]
        
        if filepath.exists():
            print(f"✓ {weight_info['name']} already exists")
            success = True
            continue
            
        try:
            print(f"\nTrying to download: {weight_info['description']}")
            download_file(weight_info["url"], filepath)
            
            # Verify the downloaded file is a valid PyTorch checkpoint
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                print(f"✓ Verified valid PyTorch checkpoint")
                print(f"  Keys: {list(checkpoint.keys())}")
                success = True
                break
            except Exception as e:
                print(f"✗ Invalid checkpoint: {e}")
                os.remove(filepath)
                
        except Exception as e:
            print(f"✗ Download failed: {e}")
            if filepath.exists():
                os.remove(filepath)
    
    if not success:
        print("\n❌ Could not download pretrained weights automatically.")
        print("\nManual download instructions:")
        print("1. Go to: https://github.com/shaofengzeng/SuperPoint-Pytorch")
        print("2. Download their pretrained weights")
        print("3. Place the .pth file in: pretrained_weights/")
        print("4. Rename it to: superpoint_pretrained.pth")
        return False
    
    print(f"\n🎉 Pretrained SuperPoint weights ready!")
    print(f"Location: {weights_dir}")
    return True

if __name__ == "__main__":
    download_pretrained_superpoint() 