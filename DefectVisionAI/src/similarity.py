
import torch
import torch.nn as nn
import timm
import numpy as np
import faiss
import cv2

class EmbeddingExtractor(nn.Module):
    """
    Wraps a trained EfficientNet-B0 classifier and exposes its 1280-dim
    global-pooled penultimate features as an image embedding, instead of
    the 2-class logits. Reuses the leak-free v3 classifier weights.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)  # num_classes=0 -> feature extractor mode
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state"]
        # Strip the classifier head keys since num_classes=0 has no head to load into
        backbone_state = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.backbone(x)  # B, 1280


class CategoryFAISSIndex:
    """FAISS L2 index + parallel metadata list, one instance per category."""
    def __init__(self, dim=1280):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # list of dicts, same order as vectors added

    def add(self, embeddings, metadata_rows):
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata_rows)

    def search(self, query_embedding, k=5):
        query_embedding = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = dict(self.metadata[idx])
            meta["l2_distance"] = float(dist)
            results.append(meta)
        return results

    def save(self, index_path, metadata_path):
        faiss.write_index(self.index, index_path)
        import json
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def load(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        import json
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
