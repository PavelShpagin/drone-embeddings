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
from random import Random
from src.azure_maps import get_azure_maps_image


def calculate_distances_and_AP(query_embed, positive_embeds, negative_embeds):
    """Calculate distances and Average Precision for one query"""
    # Ensure inputs are tensors
    if not isinstance(query_embed, torch.Tensor):
        query_embed = torch.tensor(query_embed)
    if not isinstance(positive_embeds, torch.Tensor):
        positive_embeds = torch.tensor(positive_embeds)
    if not isinstance(negative_embeds, torch.Tensor):
        negative_embeds = torch.tensor(negative_embeds)

    # Move tensors to the same device
    device = query_embed.device
    positive_embeds = positive_embeds.to(device)
    negative_embeds = negative_embeds.to(device)

    # Handle cases with 0 positive or negative samples gracefully
    if positive_embeds.numel() == 0:
        print("Warning: 0 positive samples found for AP calculation.")
        # Depending on desired behavior, return 0 or handle differently
        if negative_embeds.numel() == 0:
            return 1.0  # Or 0.0? If no neg, all are pos?
        pos_distances = torch.tensor([], device=device)  # Empty tensor
    else:
        pos_distances = torch.cdist(query_embed.unsqueeze(0), positive_embeds).squeeze(
            0
        )

    if negative_embeds.numel() == 0:
        print("Warning: 0 negative samples found for AP calculation.")
        if positive_embeds.numel() == 0:
            return 0.0  # No samples at all
        neg_distances = torch.tensor([], device=device)  # Empty tensor
    else:
        neg_distances = torch.cdist(query_embed.unsqueeze(0), negative_embeds).squeeze(
            0
        )

    # Proceed only if there are distances to compare
    if pos_distances.numel() == 0 and neg_distances.numel() == 0:
        return 0.0  # No samples to calculate AP

    all_distances = torch.cat([pos_distances, neg_distances])
    all_labels = torch.cat(
        [torch.ones_like(pos_distances), torch.zeros_like(neg_distances)]
    )

    # Check for empty labels (shouldn't happen if we checked distances)
    if all_labels.numel() == 0:
        return 0.0

    sorted_indices = torch.argsort(all_distances)
    sorted_labels = all_labels[sorted_indices]

    precisions = torch.cumsum(sorted_labels, dim=0) / torch.arange(
        1, len(sorted_labels) + 1, device=device
    )

    # Denominator for AP calculation
    num_positives = sorted_labels.sum()
    if num_positives == 0:
        return 0.0  # No relevant items

    ap_value = (precisions * sorted_labels).sum() / num_positives
    if torch.isnan(ap_value):
        return 0.0
    return float(ap_value)


def evaluate_model(model, dataloader, device, num_evaluations=5):
    """Evaluate model using multiple samplings"""
    model.eval()
    all_APs = []
    num_batches_for_eval = 10

    # Ensure dataloader used for evaluation doesn't save images
    # It's recommended to create a separate dataloader instance for eval
    # Example: eval_dataloader = MapDataLoader(dataloader.dataset, ...) # with save=False

    print("Starting evaluation...")
    with torch.no_grad():
        for eval_run in range(num_evaluations):
            batch_APs = []
            batches_evaluated = 0
            try:
                # Use the provided dataloader (caller should ensure it's configured correctly)
                for i, batch_data in enumerate(dataloader):
                    if batches_evaluated >= num_batches_for_eval:
                        break

                    print()

                    if len(batch_data) != 3:
                        print(
                            f"Warning [Eval]: Skipping batch {i} due to unexpected data format (length {len(batch_data)}). Expected 3 Tensors."
                        )
                        continue

                    query, positive, negative = batch_data

                    # Move to device
                    query = query.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)

                    # Get embeddings
                    # Check if model requires eval mode specific handling if different from train
                    query_embed, positive_embed, negative_embed = model(
                        query, positive, negative
                    )

                    # Calculate AP for each item in the batch
                    current_batch_aps = []
                    batch_size = query_embed.shape[0]
                    # Determine expected number of pos/neg per query item
                    # Assuming it's stored or accessible via dataloader attributes
                    pos_per_query = getattr(
                        dataloader, "pos_samples", 1
                    )  # Default or get from loader
                    neg_per_query = getattr(
                        dataloader, "neg_samples", 1
                    )  # Default or get from loader

                    # Check tensor shapes before slicing
                    if (
                        positive_embed.shape[0] != batch_size * pos_per_query
                        or negative_embed.shape[0] != batch_size * neg_per_query
                    ):
                        print(
                            f"Warning [Eval]: Mismatch in embedding shapes for batch {i}. "
                            f"Pos: {positive_embed.shape[0]}, Expected: {batch_size * pos_per_query}. "
                            f"Neg: {negative_embed.shape[0]}, Expected: {batch_size * neg_per_query}. Skipping AP calc."
                        )
                        continue  # Skip AP calculation for this batch if shapes are wrong

                    for item_idx in range(batch_size):
                        ap = calculate_distances_and_AP(
                            query_embed[item_idx],
                            positive_embed[
                                item_idx
                                * pos_per_query : (item_idx + 1)
                                * pos_per_query
                            ],
                            negative_embed[
                                item_idx
                                * neg_per_query : (item_idx + 1)
                                * neg_per_query
                            ],
                        )
                        current_batch_aps.append(ap)

                    if current_batch_aps:
                        batch_APs.append(np.mean(current_batch_aps))
                    batches_evaluated += 1

                if batch_APs:
                    all_APs.append(np.mean(batch_APs))
                else:
                    print(
                        f"Warning: No batches evaluated successfully during evaluation run {eval_run + 1}."
                    )

            except Exception as e:
                print(
                    f"Error during evaluation run {eval_run + 1}, batch {batches_evaluated}: {e}"
                )
                import traceback

                traceback.print_exc()
                # Decide whether to continue to next eval run or stop

    print("Evaluation finished.")
    if not all_APs:
        print("Warning: Evaluation failed, no AP scores recorded.")
        mean_AP, std_AP = 0.0, 0.0
    else:
        mean_AP = np.mean(all_APs)
        std_AP = np.std(all_APs)

    model.train()  # Set model back to training mode
    return mean_AP, std_AP


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

    # Corrected: project_root should be the directory containing train.py (assuming it's at the root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    # If train.py is NOT at the root, adjust accordingly. E.g., if it's in 'scripts/':
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    print(
        f"Corrected Project Root: {project_root}"
    )  # Should print /home/pavel/dev/drone-embeddings

    # These paths should now be correct
    imagery_output_dir = os.path.join(project_root, "imagery")
    jsonl_output_file = os.path.join(project_root, "dataset.jsonl")
    model_save_dir = os.path.join(project_root, "models")
    # Dataloader now handles dir creation if save=True, but doesn't hurt here
    os.makedirs(imagery_output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

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

    # Pass the CORRECT project_root
    train_dataloader = MapDataLoader(
        dataset,
        pos_samples=config["training"]["pos_samples"],
        neg_samples=config["training"]["neg_samples"],
        save=True,
        imagery_output_dir=imagery_output_dir,
        jsonl_output_file=jsonl_output_file,
        project_root=project_root,  # <<< Ensure this uses the corrected path
    )
    print(
        f"Initializing Training DataLoader: save=True, imagery_dir='{imagery_output_dir}', jsonl_file='{jsonl_output_file}'"
    )

    eval_dataloader = MapDataLoader(
        dataset,
        pos_samples=config["training"]["pos_samples"],
        neg_samples=config["training"]["neg_samples"],
        save=False,
    )
    print("Initialized Evaluation DataLoader: save=False")

    effective_batch_size = getattr(train_dataloader, "batch_size", 1)
    print(f"Using effective Batch Size: {effective_batch_size}")

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

    # Open JSONL file in append mode
    try:
        with open(jsonl_output_file, "a") as f_jsonl:
            for epoch in range(config["training"]["num_epochs"]):
                epoch_loss = 0
                batch_count = 0
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

                for query_t, pos_t, neg_t in progress_bar:
                    query_t = query_t.to(device)
                    pos_t = pos_t.to(device)
                    neg_t = neg_t.to(device)

                    query_embed, positive_embed, negative_embed = model(
                        query_t, pos_t, neg_t
                    )

                    loss = criterion(query_embed, positive_embed, negative_embed)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1
                    progress_bar.set_postfix({"loss": loss.item()})

                    if (
                        config["training"].get("num_mini_batches")
                        and batch_count >= config["training"]["num_mini_batches"]
                    ):
                        print(
                            f"Reached num_mini_batches limit ({config['training']['num_mini_batches']}). Ending epoch early."
                        )
                        break

                if batch_count == 0:
                    print(
                        f"Warning: Epoch {epoch+1} completed without processing any batches."
                    )
                    continue

                avg_epoch_loss = epoch_loss / batch_count

                mean_AP, std_AP = evaluate_model(model, eval_dataloader, device)
                print(
                    f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, Mean AP: {mean_AP:.4f} ± {std_AP:.4f}"
                )

                if mean_AP > best_AP:
                    best_AP = mean_AP
                    save_path = os.path.join(model_save_dir, "siamese_net_best.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": avg_epoch_loss,
                            "mean_AP": mean_AP,
                            "std_AP": std_AP,
                        },
                        save_path,
                    )
                    print(f"New best model saved to {save_path}")

                if (epoch + 1) % config["training"].get("checkpoint_freq", 10) == 0:
                    save_path = os.path.join(
                        model_save_dir, f"siamese_net_epoch_{epoch+1}.pth"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": avg_epoch_loss,
                            "mean_AP": mean_AP,
                            "std_AP": std_AP,
                        },
                        save_path,
                    )
                    print(f"Checkpoint saved to {save_path}")

    except IOError as e:
        print(f"Error opening or writing to {jsonl_output_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    train()
