
import torch
import torch.nn as nn
import numpy as np
import cv2

class ConvAutoencoder(nn.Module):
    """
    Reusable conv autoencoder for DefectVision AI.
    Input: 3x128x128. Trained per-category on normal images only.
    """
    def __init__(self, latent_channels=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),   # 64
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),  # 32
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),# 16
            nn.Conv2d(128, latent_channels, 4, stride=2, padding=1), nn.BatchNorm2d(latent_channels), nn.ReLU(True), # 8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), # 16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),   # 32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),    # 64
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),                          # 128
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def compute_error_map(original, reconstructed):
    """original, reconstructed: CxHxW tensors in [0,1]. Returns HxW numpy error map."""
    diff = (original - reconstructed) ** 2
    error_map = diff.mean(dim=0)  # average over channels
    return error_map.cpu().numpy()


def image_level_score(error_map, topk_frac=0.05):
    """Mean of the top-k% highest error pixels — more robust than global mean, less noisy than pure max."""
    flat = error_map.flatten()
    k = max(1, int(len(flat) * topk_frac))
    topk = np.partition(flat, -k)[-k:]
    return float(topk.mean())


def normalize_map(error_map):
    m = error_map - error_map.min()
    if m.max() > 0:
        m = m / m.max()
    return m


def overlay_heatmap(image_rgb_uint8, error_map_norm, alpha=0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * error_map_norm), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (image_rgb_uint8.shape[1], image_rgb_uint8.shape[0]))
    overlay = (alpha * heatmap + (1 - alpha) * image_rgb_uint8).astype(np.uint8)
    return overlay


def map_iou(error_map_norm, gt_mask, threshold=0.5):
    err_bin = (error_map_norm >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    if err_bin.shape != gt_bin.shape:
        err_bin = cv2.resize(err_bin, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection = np.logical_and(err_bin, gt_bin).sum()
    union = np.logical_or(err_bin, gt_bin).sum()
    if union == 0:
        return 0.0
    return intersection / union
