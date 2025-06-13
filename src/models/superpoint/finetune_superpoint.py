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
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint").to(device)
    dataset = SyntheticShapesDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=epochs*len(loader))
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            inputs = processor(batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else outputs[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss': loss.item()})
        if (epoch+1) % save_every == 0:
            model.save_pretrained(os.path.join(output_dir, f"checkpoint_epoch{epoch+1}"))
    model.save_pretrained(os.path.join(output_dir, "final"))

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