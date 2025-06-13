import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, SuperPointForKeypointDetection, get_linear_schedule_with_warmup
from PIL import Image
import numpy as np
from tqdm import tqdm

class SyntheticShapesDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def collate_fn(batch):
    return batch

def train_superpoint(
    data_dir,
    output_dir,
    epochs=50,
    batch_size=32,
    lr=1e-4,
    device='cuda',
    save_every=5
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and processor
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    # Move to device and set to training mode
    model = model.to(device)
    model.train()
    
    # Setup data
    dataset = SyntheticShapesDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=epochs*len(loader))
    
    # Training loop
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        for batch in pbar:
            # Process batch
            inputs = processor(batch, return_tensors="pt").to(device)
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Log epoch metrics
        print(f"\nEpoch {epoch+1} average loss: {epoch_loss/len(loader):.4f}")
        
        # Save checkpoint
        if (epoch+1) % save_every == 0:
            model.save_pretrained(os.path.join(output_dir, f"checkpoint_epoch{epoch+1}"))
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to synthetic or real training images')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_superpoint(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr, args.device) 