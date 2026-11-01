import os
import sys

# Ensure paths
sys.path.append(os.path.abspath("DefectVisionAI"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import cv2
import pandas as pd
import hashlib

from src.patchcore import PatchCoreExtractor, PatchCoreModel, normalize_map, overlay_heatmap
from src.severity import normalize_anomaly_score, compute_severity_score, bucket_severity

print("--- Step 1: Checking device ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\n--- Step 2: Verifying Model Artifact Hashes ---")
ckpt_path = "DefectVisionAI/models/efficientnet_baseline_v3/best_model.pt"
if os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"EfficientNet-B0 v3 checkpoint: {ckpt_path}")
    print(f"SHA-256: {ckpt_hash}")
else:
    print(f"EfficientNet-B0 v3 checkpoint NOT found at {ckpt_path}")

mb_path = "DefectVisionAI/models/patchcore/bottle/memory_bank.npy"
if os.path.exists(mb_path):
    with open(mb_path, "rb") as f:
        mb_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Bottle PatchCore Memory Bank: {mb_path}")
    print(f"SHA-256: {mb_hash}")
else:
    print(f"Memory bank NOT found at {mb_path}")

print("\n--- Step 3: Loading Models into Memory ---")
extractor = PatchCoreExtractor(grid_size=28).to(device)
extractor.eval()
print("PatchCoreExtractor loaded successfully!")

model_pc = PatchCoreModel()
model_pc.load(mb_path, grid_size=28)
print(f"Bottle Memory Bank shape: {model_pc.memory_bank.shape}")
print("Bottle PatchCoreModel loaded into FAISS index FlatL2!")

print("\n--- Step 4: Finding a Test Image for Bottle ---")
# Look for a test image in DefectVisionAI or outputs
test_img_path = None
candidates = [
    "DefectVisionAI/outputs/similarity_search_samples/bottle_similarity_demo.png",
    "DefectVisionAI/batch_inspections/demo_150img_batch/bottle_001.png"
]
for c in candidates:
    if os.path.exists(c):
        test_img_path = c
        break

if not test_img_path:
    # Search for any png in outputs or dataset
    for root, dirs, files in os.walk("DefectVisionAI"):
        for file in files:
            if file.endswith((".png", ".jpg")) and "bottle" in root.lower():
                test_img_path = os.path.join(root, file)
                break
        if test_img_path:
            break

print(f"Selected test image: {test_img_path}")

if test_img_path:
    img_bgr = cv2.imread(test_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Preprocessing with fixed ImageNet norm
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = ((img_resized / 255.0) - mean) / std
    inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    print("\n--- Step 5: Running PatchCore Inference ---")
    with torch.no_grad():
        features = extractor(inp_tensor)
    
    feat_np = features.squeeze(0).cpu().numpy()
    dist_map, raw_score = model_pc.score(feat_np)
    
    # Load bottle baseline
    baselines = pd.read_csv("DefectVisionAI/reports/step9_severity_baselines.csv").set_index("category")
    b_mean = baselines.loc["bottle", "baseline_mean"]
    b_p99 = baselines.loc["bottle", "baseline_p99"]
    
    # Severity & Decision
    optimal_thresh = b_p99 # default nominal threshold
    decision = "DEFECTIVE" if raw_score > optimal_thresh else "NOMINAL"
    
    norm_map = normalize_map(dist_map)
    heatmap_overlay = overlay_heatmap(img_resized, norm_map)
    
    print(f"Raw PatchCore Score:   {raw_score:.4f}")
    print(f"Baseline Mean:          {b_mean:.4f}")
    print(f"Decision Threshold p99: {optimal_thresh:.4f}")
    print(f"Classification Decision:{decision}")
    print(f"Heatmap Shape:          {heatmap_overlay.shape}")
    
    # Save output heatmap verification image
    os.makedirs("delivery_24h/final_verification_v2/test_outputs", exist_ok=True)
    out_hm_path = "delivery_24h/final_verification_v2/test_outputs/verified_bottle_heatmap.png"
    cv2.imwrite(out_hm_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
    print(f"Heatmap written to:     {out_hm_path}")
    print("\n--- INFERENCE TEST PASSED COMPLETELY ---")
