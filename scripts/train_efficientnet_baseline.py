#!/usr/bin/env python3
"""
DefectVision AI — EfficientNet-B0 Supervised Baseline Training CLI

Fine-tunes a supervised EfficientNet-B0 binary classifier (normal vs. defective)
on industrial anomaly images, saving the production checkpoint with state dict
and feature extraction compatibility.

Usage:
    python scripts/train_efficientnet_baseline.py --data_dir ./datasets/mvtec/ --epochs 15
    python scripts/train_efficientnet_baseline.py --data_dir ./datasets/mvtec/ --category bottle --epochs 10
"""

import os
import sys
import argparse
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "DefectVisionAI"))

EFF_IMG_SIZE = 256
EFF_MEAN = (0.485, 0.456, 0.406)
EFF_STD  = (0.229, 0.224, 0.225)

class AnomalyClassificationDataset(Dataset):
    """Pairs images with binary labels: 0 for nominal, 1 for defective."""
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (filepath, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((EFF_IMG_SIZE, EFF_IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (EFF_IMG_SIZE, EFF_IMG_SIZE))

        img = img.astype(np.float32) / 255.0
        img = (img - np.array(EFF_MEAN, dtype=np.float32)) / np.array(EFF_STD, dtype=np.float32)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return tensor, torch.tensor(label, dtype=torch.long)

def collect_samples(data_dir, category=None):
    """Walks the dataset directory to collect (filepath, label) tuples."""
    samples = []
    cat_dirs = [os.path.join(data_dir, category)] if category else glob.glob(os.path.join(data_dir, "*"))

    for cdir in sorted(cat_dirs):
        if not os.path.isdir(cdir):
            continue
        cname = os.path.basename(cdir)

        # 1. Normal train images (label = 0)
        train_good = glob.glob(os.path.join(cdir, "train", "good", "*.png")) + \
                     glob.glob(os.path.join(cdir, "train", "good", "*.jpg"))
        for p in train_good:
            samples.append((p, 0))

        # 2. Test images (good -> 0, defects -> 1)
        test_dir = os.path.join(cdir, "test")
        if os.path.isdir(test_dir):
            for sub in os.listdir(test_dir):
                subdir_path = os.path.join(test_dir, sub)
                if not os.path.isdir(subdir_path):
                    continue
                imgs = glob.glob(os.path.join(subdir_path, "*.png")) + \
                       glob.glob(os.path.join(subdir_path, "*.jpg"))
                lbl = 0 if sub == "good" else 1
                for p in imgs:
                    samples.append((p, lbl))

    return samples

def train_efficientnet(data_dir, category, output_dir, epochs=15, batch_size=32, lr=1e-4, device="cpu"):
    """Runs supervised fine-tuning and checkpoint saving."""
    print("\n==========================================")
    print("DefectVision AI — EfficientNet-B0 Fine-Tuning")
    print(f"Data Dir: {data_dir}")
    print(f"Target Category: {category or 'All Categories'}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Device: {device}")
    print("==========================================")

    samples = collect_samples(data_dir, category)
    if not samples:
        raise ValueError(f"No samples found under {data_dir}")

    normals = sum(1 for _, l in samples if l == 0)
    defects = sum(1 for _, l in samples if l == 1)
    print(f"Dataset indexed: {len(samples)} total ({normals} normal, {defects} defective)")

    dataset = AnomalyClassificationDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Initialize model
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = (correct / max(1, total)) * 100.0
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:5.1f}% | Time: {elapsed:.2f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss": best_loss,
                "accuracy": epoch_acc,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            torch.save(ckpt, best_ckpt_path)

    print(f"\nTraining completed! Saved best checkpoint to: {best_ckpt_path}")
    print(f"Checkpoint size: {os.path.getsize(best_ckpt_path):,} bytes")
    return best_ckpt_path

def main():
    parser = argparse.ArgumentParser(
        description="DefectVision AI — EfficientNet-B0 Supervised Baseline Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_efficientnet_baseline.py --data_dir ./datasets/mvtec/ --epochs 15
  python scripts/train_efficientnet_baseline.py --data_dir ./datasets/mvtec/ --category bottle --epochs 10
        """
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root path to dataset directory.")
    parser.add_argument("--category", type=str, default=None,
                        help="Train on specific category (default: all available).")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of fine-tuning epochs (default: 15).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4).")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "DefectVisionAI", "models", "efficientnet_baseline_v3"),
                        help="Directory where best_model.pt will be saved.")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device ('cuda' or 'cpu'). Default: auto-detect.")

    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_efficientnet(
        data_dir=args.data_dir,
        category=args.category,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

if __name__ == "__main__":
    main()
