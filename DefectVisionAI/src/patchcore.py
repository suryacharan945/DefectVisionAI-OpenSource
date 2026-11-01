
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import faiss
import cv2

class PatchCoreExtractor(nn.Module):
    def __init__(self, grid_size=28):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.grid_size = grid_size
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x0 = self.layer0(x)
        f1 = self.layer1(x0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f2 = self.local_pool(f2)
        f3 = self.local_pool(f3)
        g = self.grid_size
        f2r = F.interpolate(f2, size=(g, g), mode="bilinear", align_corners=False)
        f3r = F.interpolate(f3, size=(g, g), mode="bilinear", align_corners=False)
        fused = torch.cat([f2r, f3r], dim=1)
        return fused


def greedy_coreset_gpu(patch_embeddings, coreset_ratio=0.10, seed=42, device="cuda", log_every=2000):
    """
    GPU-vectorized greedy k-center coreset selection.
    Replaces the old O(N*K) pure-NumPy loop — same algorithm, ~50-100x faster
    because each iteration's distance-to-all-points update runs as one
    vectorized GPU tensor op instead of a Python-level NumPy call.
    """
    patches = torch.from_numpy(patch_embeddings).float().to(device)
    N, D = patches.shape
    K = max(1, int(N * coreset_ratio))

    rng = np.random.RandomState(seed)
    start_idx = int(rng.randint(N))
    selected_idx = [start_idx]

    min_dists = torch.norm(patches - patches[start_idx], dim=1)

    for i in range(1, K):
        next_idx = int(torch.argmax(min_dists).item())
        selected_idx.append(next_idx)
        new_dists = torch.norm(patches - patches[next_idx], dim=1)
        min_dists = torch.minimum(min_dists, new_dists)

        if log_every and i % log_every == 0:
            print(f"    coreset progress: {i}/{K}")

    coreset = patches[selected_idx].cpu().numpy()
    return coreset


class PatchCoreModel:
    def __init__(self):
        self.memory_bank = None
        self.index = None
        self.grid_size = None

    def fit(self, coreset_embeddings, grid_size):
        self.memory_bank = coreset_embeddings.astype(np.float32)
        self.grid_size = grid_size
        self.index = faiss.IndexFlatL2(self.memory_bank.shape[1])
        self.index.add(self.memory_bank)

    def score(self, patch_embedding):
        C, g, _ = patch_embedding.shape
        patches = patch_embedding.reshape(C, g * g).T.astype(np.float32)
        distances, _ = self.index.search(patches, k=1)
        dist_map = np.sqrt(np.clip(distances[:, 0], 0, None)).reshape(g, g)
        image_score = float(dist_map.max())
        return dist_map, image_score

    def save(self, path):
        np.save(path, self.memory_bank)

    def load(self, path, grid_size=28):
        self.memory_bank = np.load(path)
        self.grid_size = grid_size
        self.index = faiss.IndexFlatL2(self.memory_bank.shape[1])
        self.index.add(self.memory_bank)


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
