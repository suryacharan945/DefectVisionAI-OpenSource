
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2

class PaDiMExtractor(nn.Module):
    """
    Extracts multi-layer ResNet-18 features and fuses them onto a common grid.
    Follows the PaDiM paper (Defard et al., 2020): concatenate layer1/2/3
    activations, resized to a common resolution, then random-dimension-reduce.
    """
    def __init__(self, grid_size=28, reduced_dim=100, seed=42):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64ch
        self.layer2 = backbone.layer2  # 128ch
        self.layer3 = backbone.layer3  # 256ch
        self.grid_size = grid_size

        total_channels = 64 + 128 + 256  # 448
        rng = np.random.RandomState(seed)
        self.register_buffer(
            "dim_idx",
            torch.tensor(rng.choice(total_channels, reduced_dim, replace=False), dtype=torch.long)
        )

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x0 = self.layer0(x)
        f1 = self.layer1(x0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        g = self.grid_size
        f1r = F.interpolate(f1, size=(g, g), mode="bilinear", align_corners=False)
        f2r = F.interpolate(f2, size=(g, g), mode="bilinear", align_corners=False)
        f3r = F.interpolate(f3, size=(g, g), mode="bilinear", align_corners=False)

        fused = torch.cat([f1r, f2r, f3r], dim=1)  # B, 448, g, g
        fused = fused[:, self.dim_idx, :, :]        # B, reduced_dim, g, g
        return fused


class PaDiMModel:
    """Per-category Gaussian model over patch embeddings. mean/cov per grid position."""
    def __init__(self, grid_size=28, reduced_dim=100, epsilon=0.01):
        self.grid_size = grid_size
        self.reduced_dim = reduced_dim
        self.epsilon = epsilon
        self.mean = None      # (g*g, reduced_dim)
        self.cov_inv = None   # (g*g, reduced_dim, reduced_dim)

    def fit(self, embeddings):
        """embeddings: (N, reduced_dim, g, g) numpy array from normal training images."""
        N, C, g, _ = embeddings.shape
        embeddings = embeddings.reshape(N, C, g * g)  # N, C, g*g

        mean = embeddings.mean(axis=0)  # C, g*g
        cov_inv = np.zeros((g * g, C, C), dtype=np.float32)
        identity = np.eye(C, dtype=np.float32) * self.epsilon

        for i in range(g * g):
            patch = embeddings[:, :, i]  # N, C
            cov = np.cov(patch, rowvar=False) + identity
            cov_inv[i] = np.linalg.inv(cov)

        self.mean = mean.T  # g*g, C
        self.cov_inv = cov_inv

    def score(self, embedding):
        """embedding: (reduced_dim, g, g) numpy array for ONE image. Returns (g,g) distance map."""
        C, g, _ = embedding.shape
        embedding = embedding.reshape(C, g * g).T  # g*g, C

        diff = embedding - self.mean  # g*g, C
        dist = np.einsum("ni,nij,nj->n", diff, self.cov_inv, diff)  # Mahalanobis^2 per patch
        dist = np.sqrt(np.clip(dist, 0, None))
        return dist.reshape(g, g)

    def save(self, path):
        np.savez(path, mean=self.mean, cov_inv=self.cov_inv,
                  grid_size=self.grid_size, reduced_dim=self.reduced_dim)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.cov_inv = data["cov_inv"]
        self.grid_size = int(data["grid_size"])
        self.reduced_dim = int(data["reduced_dim"])


def normalize_map(m):
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return m


def overlay_heatmap(image_rgb_uint8, score_map_norm, alpha=0.45):
    hm = cv2.resize(score_map_norm, (image_rgb_uint8.shape[1], image_rgb_uint8.shape[0]))
    hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    overlay = (alpha * hm + (1 - alpha) * image_rgb_uint8).astype(np.uint8)
    return overlay


def map_iou(score_map_norm, gt_mask, threshold=0.5):
    sm = cv2.resize(score_map_norm, (gt_mask.shape[1], gt_mask.shape[0]))
    sm_bin = (sm >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(sm_bin, gt_bin).sum()
    union = np.logical_or(sm_bin, gt_bin).sum()
    if union == 0:
        return 0.0
    return intersection / union
