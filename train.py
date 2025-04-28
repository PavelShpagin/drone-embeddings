import yaml
import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from src.dataset.map_dataset import MapDataset, MapDataLoader
from src.models.siamese_net import SiameseNet
from src.losses.triplet_loss import WeightedTripletLoss
from src.utils.transforms import get_train_transforms
from dotenv import load_dotenv, dotenv_values
import json
import os
from PIL import Image
import base64
from io import BytesIO


def calculate_distances_and_AP(query_embed, positive_embeds, negative_embeds):
    """Calculate distances and Average Precision for one query"""
    # Calculate distances
    pos_distances = torch.cdist(query_embed.unsqueeze(0), positive_embeds).squeeze()
    neg_distances = torch.cdist(query_embed.unsqueeze(0), negative_embeds).squeeze()

    # Combine distances and create labels
    all_distances = torch.cat([pos_distances, neg_distances])
    all_labels = torch.cat(
        [torch.ones_like(pos_distances), torch.zeros_like(neg_distances)]
    )

    # Sort by distance
    sorted_indices = torch.argsort(all_distances)
    sorted_labels = all_labels[sorted_indices]

    # Calculate precision at each position
    precisions = torch.cumsum(sorted_labels, dim=0) / torch.arange(
        1, len(sorted_labels) + 1, device=sorted_labels.device
    )

    # Calculate AP
    return float((precisions * sorted_labels).sum() / sorted_labels.sum())


def evaluate_model(model, dataloader, device, num_evaluations=5):
    """Evaluate model using multiple samplings"""
    model.eval()
    all_APs = []

    with torch.no_grad():
        for _ in range(num_evaluations):
            batch_APs = []
            # Get 10 batches for evaluation
            for i, (query, positive, negative) in enumerate(dataloader):
                if i >= 10:  # Only evaluate on 10 batches
                    break

                # Move to device
                query = query.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                # Get embeddings
                query_embed, positive_embed, negative_embed = model(
                    query, positive, negative
                )

                # Calculate AP for this batch
                ap = calculate_distances_and_AP(
                    query_embed[0], positive_embed, negative_embed
                )
                batch_APs.append(ap)

            # Average AP for this evaluation
            all_APs.append(np.mean(batch_APs))

    # Calculate mean and std of APs across evaluations
    mean_AP = np.mean(all_APs)
    std_AP = np.std(all_APs)

    model.train()
    return mean_AP, std_AP


def save_batch_to_json(
    query, positive, negative, batch_idx, epoch, save_dir="data_samples"
):
    """Save batch images and metadata to JSON"""
    os.makedirs(save_dir, exist_ok=True)

    def tensor_to_base64(tensor):
        # Convert tensor to PIL Image and then to base64
        img = Image.fromarray(
            (tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
        )
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    batch_data = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "images": {
            "query": [tensor_to_base64(img) for img in query],
            "positive": [tensor_to_base64(img) for img in positive],
            "negative": [tensor_to_base64(img) for img in negative],
        },
    }

    json_path = os.path.join(save_dir, f"batch_{epoch}_{batch_idx}.json")
    with open(json_path, "w") as f:
        json.dump(batch_data, f)


def train():
    # Load config
    with open("config/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load environment variables
    load_dotenv()
    secrets = dotenv_values(".env")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = SiameseNet(
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        gem_p=config["model"]["gem_p"],
        embedding_dim=config["model"]["embedding_dim"],
    ).to(device)

    # Initialize dataset
    locations = [
        (lat, lon, config["data"]["base_height"])
        for lat, lon in config["data"]["locations"]
    ]

    dataset = MapDataset(
        locations=locations,
        google_api_key=secrets.get("GOOGLE_MAPS_API_KEY"),
        azure_api_key=secrets.get("AZURE_MAPS_API_KEY"),
        transform=get_train_transforms(),
    )

    dataloader = MapDataLoader(
        dataset,
        pos_samples=config["training"]["pos_samples"],
        neg_samples=config["training"]["neg_samples"],
    )

    # Initialize loss and optimizer
    criterion = WeightedTripletLoss(alpha=config["training"]["alpha"])
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Training loop
    best_AP = 0.0
    model.train()

    for epoch in range(config["training"]["num_epochs"]):
        epoch_loss = 0
        batch_count = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, (query, positive, negative) in enumerate(progress_bar):
            # Save batch data to JSON
            save_batch_to_json(query, positive, negative, batch_idx, epoch)

            # Move to device
            query = query.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Forward pass
            query_embed, positive_embed, negative_embed = model(
                query, positive, negative
            )

            # Calculate loss
            loss = criterion(query_embed, positive_embed, negative_embed)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

            if batch_count >= config["training"]["num_mini_batches"]:
                break

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count

        # Evaluate model
        mean_AP, std_AP = evaluate_model(model, dataloader, device)
        print(
            f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, Mean AP: {mean_AP:.4f} ± {std_AP:.4f}"
        )

        # Save best model
        if mean_AP > best_AP:
            best_AP = mean_AP
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "mean_AP": mean_AP,
                    "std_AP": std_AP,
                },
                "models/siamese_net_best.pth",
            )

        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "mean_AP": mean_AP,
                    "std_AP": std_AP,
                },
                f"models/siamese_net_epoch_{epoch+1}.pth",
            )


if __name__ == "__main__":
    train()
