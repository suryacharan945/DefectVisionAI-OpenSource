import os
import sys

# 1. OpenMP Runtime duplicate DLL fix (crucial on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add current folder to path to enable local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import base64
import shutil
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import local models and helpers
from src.patchcore import PatchCoreExtractor, PatchCoreModel, normalize_map, overlay_heatmap
from src.severity import normalize_anomaly_score, compute_severity_score, bucket_severity
from src.similarity import EmbeddingExtractor, CategoryFAISSIndex
from src.llm_explainer import load_explainer_pipeline, build_explanation_prompt, generate_explanation
from src.explanation_cleanup import clean_explanation
from src.rag_assistant import KnowledgeBaseIndex, build_rag_prompt
from src.dashboard_router import route_intent, extract_category, extract_model_name
from src.dashboard_analytics import DashboardAnalytics
from src.pattern_analysis import PatternAnalyzer
from src.numeric_validator import numbers_are_grounded
from src.batch_inspection import build_batch_report

# -----------------------------
# Setup Global Configurations & Devices
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
REPORTS_DIR = os.path.join(APP_DIR, "reports")
LOGS_DIR = os.path.join(APP_DIR, "logs")
BATCH_DIR = os.path.join(APP_DIR, "batch_inspections")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")

os.makedirs(BATCH_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -----------------------------
# Startup Model Caching
# -----------------------------
GRID_SIZE = 28
PC_IMG_SIZE = 224
PC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PC_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Load baseline lookup
baseline_df = pd.read_csv(os.path.join(REPORTS_DIR, "step9_severity_baselines.csv"))
baseline_lookup = baseline_df.set_index("category").to_dict("index")
known_categories = sorted(baseline_lookup.keys())

# Load PatchCore Extractor
print("Loading PatchCore Feature Extractor...")
extractor = PatchCoreExtractor(grid_size=GRID_SIZE).to(device)
extractor.eval()

# Load all 15 PatchCore Category models
print("Loading 15 PatchCore Category Memory Banks...")
PATCHCORE_DIR = os.path.join(MODELS_DIR, "patchcore")
loaded_patchcore_models = {}
def get_patchcore_model(category):
    if category not in loaded_patchcore_models:
        mb_path = os.path.join(PATCHCORE_DIR, category, "memory_bank.npy")
        if not os.path.exists(mb_path):
            return None
        m = PatchCoreModel()
        m.load(mb_path, grid_size=GRID_SIZE)
        loaded_patchcore_models[category] = m
    return loaded_patchcore_models[category]

# Warm up loaded_patchcore_models cache for all known categories
for cat in known_categories:
    try:
        get_patchcore_model(cat)
    except Exception as e:
        print(f"Warning: could not load PatchCore memory bank for {cat}: {e}")

# Load EfficientNet-B0 Classifier (v3 best model)
print("Loading EfficientNet-B0 v3 Classifier...")
CKPT_DIR_V3 = os.path.join(MODELS_DIR, "efficientnet_baseline_v3")
effnet_ckpt_path = os.path.join(CKPT_DIR_V3, "best_model.pt")
effnet_model = None
if os.path.exists(effnet_ckpt_path):
    try:
        best_ckpt = torch.load(effnet_ckpt_path, map_location=device)
        effnet_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        effnet_model.load_state_dict(best_ckpt["model_state"])
        effnet_model = effnet_model.to(device)
        effnet_model.eval()
        print("EfficientNet-B0 v3 Classifier loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load EfficientNet checkpoint from {effnet_ckpt_path}: {e}")
        effnet_model = None
else:
    print(f"[Info] EfficientNet checkpoint not found at {effnet_ckpt_path}. Operating in PatchCore-only or demo mode.")

EFF_IMG_SIZE = 256
eff_transform = A.Compose([
    A.Resize(EFF_IMG_SIZE, EFF_IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Load FAISS Indices
print("Loading FAISS Index galleries for all categories...")
loaded_faiss_indices = {}
for cat in known_categories:
    idx_path = os.path.join(MODELS_DIR, "faiss_index", cat, "index.faiss")
    meta_path = os.path.join(MODELS_DIR, "faiss_index", cat, "metadata.json")
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        idx = CategoryFAISSIndex()
        idx.load(idx_path, meta_path)
        loaded_faiss_indices[cat] = idx
    else:
        print(f"Warning: FAISS index not found for category {cat}")

# Load Embedding Extractor for query similarity search
print("Loading Embedding Extractor...")
embedding_extractor = None
if os.path.exists(effnet_ckpt_path):
    try:
        embedding_extractor = EmbeddingExtractor(effnet_ckpt_path, device=device).to(device)
        embedding_extractor.eval()
        print("Embedding Extractor loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load Embedding Extractor: {e}")
        embedding_extractor = None
else:
    print(f"[Info] Embedding Extractor checkpoint not found at {effnet_ckpt_path}. Operating in weights-pending mode.")

# Load RAG Knowledge Base and semantic index
print("Initializing RAG Knowledge Base...")
from sentence_transformers import SentenceTransformer
rag_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
kb_index = KnowledgeBaseIndex(rag_embed_model)
kb_df = pd.read_csv(os.path.join(MODELS_DIR, "rag_knowledge_base", "defect_knowledge_base.csv"))
kb_index.build(kb_df)

# Initialize Analytics classes
analytics = DashboardAnalytics(logs_dir=LOGS_DIR, reports_dir=REPORTS_DIR)
severity_log_path = os.path.join(LOGS_DIR, "step9_severity_log.csv")
if os.path.exists(severity_log_path):
    severity_log_df = pd.read_csv(severity_log_path)
else:
    severity_log_df = pd.DataFrame(columns=[
        "filepath", "category", "true_label", "defect_type",
        "raw_patchcore_score", "anomaly_norm_0_100", "defect_area_pct",
        "confidence_pct", "severity_score", "severity_level"
    ])
auroc_matrix_path = os.path.join(REPORTS_DIR, "step8_per_category_auroc_matrix.csv")
auroc_matrix = pd.read_csv(auroc_matrix_path, index_col=0) if os.path.exists(auroc_matrix_path) else None
rca = PatternAnalyzer(severity_log=severity_log_df, per_category_auroc_matrix=auroc_matrix)

# Try loading Phi-3 explainer LLM pipeline
llm_pipeline = None
llm_tokenizer = None
llm_loaded = False
try:
    print("Attempting to load microsoft/Phi-3-mini-4k-instruct (local)...")
    llm_pipeline, llm_tokenizer = load_explainer_pipeline(device, local_files_only=True)
    llm_loaded = True
    print("Phi-3 LLM loaded successfully!")
except Exception as e:
    print(f"Warning: Phi-3 LLM could not be loaded ({e}). Server starting with high-quality template fallbacks.")

# -----------------------------
# Helper Functions
# -----------------------------
def load_pc_tensor(filepath, size=PC_IMG_SIZE):
    img = cv2.imread(filepath)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    img = (img - PC_MEAN) / PC_STD
    return torch.from_numpy(img).permute(2, 0, 1).float()

def inspect_single_image(filepath, category):
    """Runs full PatchCore + EfficientNet + severity pipeline on one image."""
    if category not in baseline_lookup:
        return None

    baseline = baseline_lookup[category]
    patchcore_model = get_patchcore_model(category)
    if patchcore_model is None:
        mb_path = os.path.join(PATCHCORE_DIR, category, "memory_bank.npy")
        raise HTTPException(
            status_code=404,
            detail=f"Model artifact for category '{category}' not found at {mb_path}. Please download authorized dataset and generate memory banks following REPRODUCIBILITY.md."
        )

    pc_tensor = load_pc_tensor(filepath)
    if pc_tensor is None:
        return None
    pc_tensor = pc_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = extractor(pc_tensor)[0].cpu().numpy()
    dist_map, raw_score = patchcore_model.score(emb)
    dist_map_norm = normalize_map(dist_map)
    defect_area_pct = 100.0 * (dist_map_norm >= 0.5).sum() / dist_map_norm.size
    anomaly_norm = normalize_anomaly_score(raw_score, baseline["baseline_mean"], baseline["baseline_p99"])

    optimal_thresh = float(round(baseline["baseline_p99"], 4))
    patchcore_defective = bool(raw_score > optimal_thresh)

    image_bgr = cv2.imread(filepath)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if effnet_model is not None:
        eff_tensor = eff_transform(image=image_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = effnet_model(eff_tensor)
            prob_defective = F.softmax(output, dim=1)[0, 1].item()
    else:
        prob_defective = 1.0 if patchcore_defective else 0.0
    confidence_pct = prob_defective * 100
    effnet_defective = bool(prob_defective >= 0.5)
    
    # Joint decision: flag as defective if either unsupervised PatchCore or supervised EfficientNet triggers
    predicted_label = "defective" if (patchcore_defective or effnet_defective) else "normal"

    severity_score = compute_severity_score(anomaly_norm, defect_area_pct, confidence_pct)
    severity_level = bucket_severity(severity_score)

    # Identify nearest historical defect from FAISS gallery
    defect_type = "anomaly"
    nearest_matches = []
    if category in loaded_faiss_indices and embedding_extractor is not None:
        try:
            eff_tensor = eff_transform(image=image_rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                eff_emb = embedding_extractor(eff_tensor)[0].cpu().numpy()
            nearest_matches = loaded_faiss_indices[category].search(eff_emb, k=3)
            if nearest_matches and len(nearest_matches) > 0:
                top_m = nearest_matches[0]
                defect_type = top_m.get("defect_type", "anomaly")
        except Exception:
            defect_type = "anomaly"

    if predicted_label == "normal" and (defect_type == "none" or defect_type == "anomaly"):
        defect_type = "normal"
    elif predicted_label == "defective" and defect_type in ("none", "normal"):
        defect_type = "structural_anomaly"

    return {
        "filepath": filepath,
        "category": category,
        "predicted_label": predicted_label,
        "defect_type": defect_type,
        "confidence_pct": float(round(confidence_pct, 2)),
        "raw_patchcore_score": float(round(raw_score, 6)),
        "anomaly_score": float(round(raw_score, 6)),
        "optimal_threshold": optimal_thresh,
        "anomaly_norm_0_100": float(round(anomaly_norm, 2)),
        "defect_area_pct": float(round(defect_area_pct, 2)),
        "severity_score": float(round(severity_score, 2)),
        "severity_level": severity_level,
        "dist_map_norm": dist_map_norm,
        "nearest_matches": nearest_matches[:3]
    }

def inspect_frame_in_memory(image_bgr, category="bottle"):
    """
    Runs full PatchCore + EfficientNet-B0 inference entirely in memory on a BGR image array.
    Zero disk I/O for ultra-low latency optical inspection.
    """
    if category not in baseline_lookup:
        category = "bottle" if "bottle" in baseline_lookup else list(baseline_lookup.keys())[0]

    t_start = time.perf_counter()
    baseline = baseline_lookup[category]
    patchcore_model = get_patchcore_model(category)
    if patchcore_model is None:
        mb_path = os.path.join(PATCHCORE_DIR, category, "memory_bank.npy")
        raise HTTPException(
            status_code=404,
            detail=f"Model artifact for category '{category}' not found at {mb_path}. Please download authorized dataset and generate memory banks following REPRODUCIBILITY.md."
        )

    # 1. PatchCore feature extraction & distance scoring
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pc = cv2.resize(img_rgb, (PC_IMG_SIZE, PC_IMG_SIZE)).astype(np.float32) / 255.0
    img_pc = (img_pc - PC_MEAN) / PC_STD
    pc_tensor = torch.from_numpy(img_pc).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        emb = extractor(pc_tensor)[0].cpu().numpy()
    dist_map, raw_score = patchcore_model.score(emb)
    dist_map_norm = normalize_map(dist_map)
    defect_area_pct = float(100.0 * (dist_map_norm >= 0.5).sum() / dist_map_norm.size)
    anomaly_norm = float(normalize_anomaly_score(raw_score, baseline["baseline_mean"], baseline["baseline_p99"]))

    optimal_thresh = float(round(baseline["baseline_p99"], 4))
    patchcore_defective = bool(raw_score > optimal_thresh)

    # 2. EfficientNet-B0 defect classification
    if effnet_model is not None:
        eff_tensor = eff_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = effnet_model(eff_tensor)
            prob_defective = float(F.softmax(output, dim=1)[0, 1].item())
    else:
        prob_defective = 1.0 if patchcore_defective else 0.0
    confidence_pct = float(round(prob_defective * 100, 2))
    effnet_defective = bool(prob_defective >= 0.5)

    # 3. Severity scoring & nearest FAISS match
    predicted_label = "defective" if (patchcore_defective or effnet_defective) else "normal"
    severity_score = float(round(compute_severity_score(anomaly_norm, defect_area_pct, confidence_pct), 2))
    severity_level = bucket_severity(severity_score)

    defect_type = "anomaly"
    nearest_matches = []
    if category in loaded_faiss_indices and embedding_extractor is not None:
        try:
            eff_tensor = eff_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                eff_emb = embedding_extractor(eff_tensor)[0].cpu().numpy()
            nearest_matches = loaded_faiss_indices[category].search(eff_emb, k=3)
            if nearest_matches and len(nearest_matches) > 0:
                top_m = nearest_matches[0]
                defect_type = top_m.get("defect_type", "anomaly")
        except Exception:
            defect_type = "anomaly"

    if predicted_label == "normal" and defect_type in ("none", "anomaly"):
        defect_type = "normal"
    elif predicted_label == "defective" and defect_type in ("none", "normal"):
        defect_type = "surface_anomaly"

    # 4. Generate visual overlays & Base64 encoding
    overlay = overlay_heatmap(img_rgb, dist_map_norm)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buf_overlay = cv2.imencode(".jpg", overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    heatmap_base64 = "data:image/jpeg;base64," + base64.b64encode(buf_overlay).decode("utf-8")

    _, buf_orig = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    orig_base64 = "data:image/jpeg;base64," + base64.b64encode(buf_orig).decode("utf-8")

    hm = cv2.resize(dist_map_norm, (w, h))
    hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    _, buf_hm = cv2.imencode(".jpg", hm, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    raw_hm_base64 = "data:image/jpeg;base64," + base64.b64encode(buf_hm).decode("utf-8")

    elapsed_ms = float(round((time.perf_counter() - t_start) * 1000, 2))

    return {
        "predicted_label": predicted_label,
        "defect_type": defect_type,
        "severity_level": severity_level,
        "severity_score": severity_score,
        "anomaly_score": float(round(raw_score, 4)),
        "optimal_threshold": optimal_thresh,
        "anomaly_norm": float(round(anomaly_norm, 2)),
        "confidence_pct": confidence_pct,
        "defect_area_pct": float(round(defect_area_pct, 2)),
        "heatmap_base64": heatmap_base64,
        "original_image": orig_base64,
        "raw_heatmap": raw_hm_base64,
        "processing_time_ms": elapsed_ms,
        "nearest_matches": nearest_matches[:3]
    }

def generate_stream_frames(category="bottle"):
    """
    Yields multipart MJPEG chunks for real-time video stream.
    Connects to OpenCV webcam (VideoCapture(0)) if available.
    If webcam is unavailable or errors out, gracefully falls back to an
    animated high-fidelity industrial conveyor test feed cycling through
    real category specimens from outputs/gradcam/<category>/.
    Draws holographic HUD reticles, neon corner brackets colored by severity,
    telemetry timestamps, and live inspection status.
    """
    COLOR_NORMAL = (129, 185, 16)    # Green (#10B981)
    COLOR_MODERATE = (11, 158, 245)  # Amber (#F59E0B)
    COLOR_CRITICAL = (68, 68, 239)   # Red (#EF4444)
    COLOR_CYAN = (212, 182, 6)       # Cyan (#06B6D4)
    COLOR_WHITE = (240, 240, 240)

    # Attempt opening hardware camera
    cap = None
    use_webcam = False
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, test_f = cap.read()
            if ret and test_f is not None:
                use_webcam = True
    except Exception:
        use_webcam = False

    # Prepare specimen fallback frames if webcam is not used
    specimen_frames = []
    cat_dir = os.path.join(OUTPUTS_DIR, "gradcam", category)
    if os.path.exists(cat_dir):
        files = [f for f in os.listdir(cat_dir) if f.endswith((".png", ".jpg"))][:25]
        for f in files:
            img = cv2.imread(os.path.join(cat_dir, f))
            if img is not None:
                specimen_frames.append(img)
    if not specimen_frames:
        demo_path = os.path.join(OUTPUTS_DIR, "similarity_search_samples", f"{category}_similarity_demo.png")
        if os.path.exists(demo_path):
            d_img = cv2.imread(demo_path)
            if d_img is not None:
                h, w = d_img.shape[:2]
                specimen_frames.append(d_img[:, :min(w, 300)])

    if not specimen_frames:
        dummy = np.full((320, 320, 3), 40, dtype=np.uint8)
        cv2.circle(dummy, (160, 160), 80, (120, 120, 120), -1)
        specimen_frames.append(dummy)

    frame_idx = 0
    specimen_idx = 0
    scanline_y = 60
    scanline_dir = 5
    last_infer_time = 0
    cached_infer = {
        "predicted_label": "inspecting",
        "defect_type": "surface",
        "severity_level": "normal",
        "severity_score": 0.0,
        "anomaly_score": 0.0,
        "processing_time_ms": 0.0
    }

    try:
        while True:
            t_loop_start = time.time()
            frame = None

            if use_webcam and cap is not None and cap.isOpened():
                ret, cam_frame = cap.read()
                if ret and cam_frame is not None:
                    # Resize to standard width if needed
                    ch, cw = cam_frame.shape[:2]
                    if cw > 640:
                        cam_frame = cv2.resize(cam_frame, (640, int(ch * 640 / cw)))
                    frame = cam_frame
                else:
                    use_webcam = False

            if frame is None:
                # High-fidelity industrial conveyor test animation
                base_specimen = specimen_frames[specimen_idx % len(specimen_frames)]
                if frame_idx % 50 == 0 and frame_idx > 0:
                    specimen_idx = (specimen_idx + 1) % len(specimen_frames)
                    base_specimen = specimen_frames[specimen_idx]

                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                # Subtle industrial grid background
                for gx in range(0, 640, 40):
                    cv2.line(canvas, (gx, 0), (gx, 480), (20, 26, 36), 1)
                for gy in range(0, 480, 40):
                    cv2.line(canvas, (0, gy), (640, gy), (20, 26, 36), 1)

                target_size = 280
                resized = cv2.resize(base_specimen, (target_size, target_size))
                offset_x = int(8 * np.sin(frame_idx * 0.08))
                py = (480 - target_size) // 2
                px = max(0, min(640 - target_size, (640 - target_size) // 2 + offset_x))
                canvas[py:py+target_size, px:px+target_size] = resized
                frame = canvas

            fh, fw = frame.shape[:2]

            # Central Inspection ROI
            roi_w = int(fw * 0.58)
            roi_h = int(fh * 0.68)
            roi_x1 = (fw - roi_w) // 2
            roi_y1 = (fh - roi_h) // 2
            roi_x2 = roi_x1 + roi_w
            roi_y2 = roi_y1 + roi_h

            # Periodic in-memory inference update (~every 0.6s)
            if time.time() - last_infer_time > 0.6:
                try:
                    roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    if roi_crop.size > 0:
                        infer_res = inspect_frame_in_memory(roi_crop, category)
                        cached_infer = infer_res
                        last_infer_time = time.time()
                except Exception:
                    pass

            # Severity Color Palette
            sev_level = cached_infer.get("severity_level", "normal")
            if sev_level == "critical":
                bracket_color = COLOR_CRITICAL
            elif sev_level == "moderate":
                bracket_color = COLOR_MODERATE
            else:
                bracket_color = COLOR_NORMAL

            # Neon Corner Brackets on ROI
            bracket_len = 26
            bracket_thick = 2
            # Top-left
            cv2.line(frame, (roi_x1, roi_y1), (roi_x1 + bracket_len, roi_y1), bracket_color, bracket_thick)
            cv2.line(frame, (roi_x1, roi_y1), (roi_x1, roi_y1 + bracket_len), bracket_color, bracket_thick)
            # Top-right
            cv2.line(frame, (roi_x2, roi_y1), (roi_x2 - bracket_len, roi_y1), bracket_color, bracket_thick)
            cv2.line(frame, (roi_x2, roi_y1), (roi_x2, roi_y1 + bracket_len), bracket_color, bracket_thick)
            # Bottom-left
            cv2.line(frame, (roi_x1, roi_y2), (roi_x1 + bracket_len, roi_y2), bracket_color, bracket_thick)
            cv2.line(frame, (roi_x1, roi_y2), (roi_x1, roi_y2 - bracket_len), bracket_color, bracket_thick)
            # Bottom-right
            cv2.line(frame, (roi_x2, roi_y2), (roi_x2 - bracket_len, roi_y2), bracket_color, bracket_thick)
            cv2.line(frame, (roi_x2, roi_y2), (roi_x2, roi_y2 - bracket_len), bracket_color, bracket_thick)

            # Center Crosshair Reticle
            cx, cy = fw // 2, fh // 2
            cv2.line(frame, (cx - 14, cy), (cx + 14, cy), COLOR_CYAN, 1)
            cv2.line(frame, (cx, cy - 14), (cx, cy + 14), COLOR_CYAN, 1)
            cv2.circle(frame, (cx, cy), 20, COLOR_CYAN, 1)

            # Animated Vertical Laser Sweep Line
            scanline_y += scanline_dir
            if scanline_y >= roi_y2 - 8 or scanline_y <= roi_y1 + 8:
                scanline_dir *= -1
            laser_overlay = frame.copy()
            cv2.line(laser_overlay, (roi_x1, scanline_y), (roi_x2, scanline_y), COLOR_CYAN, 2)
            cv2.addWeighted(laser_overlay, 0.4, frame, 0.6, 0, frame)

            # Top Telemetry Header Bar
            cv2.rectangle(frame, (0, 0), (fw, 36), (10, 14, 23), -1)
            cv2.line(frame, (0, 36), (fw, 36), (30, 41, 59), 1)
            feed_source = "HARDWARE WEBCAM" if use_webcam else "INDUSTRIAL FEED"
            cv2.putText(frame, f"DEFECTVISION AI // {feed_source}", (12, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_WHITE, 1, cv2.LINE_AA)
            status_text = f"MODEL: {category.upper()} | LOCK: ACTIVE"
            cv2.putText(frame, status_text, (fw - 280, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CYAN, 1, cv2.LINE_AA)

            # Bottom HUD Metrics Bar
            cv2.rectangle(frame, (0, fh - 40), (fw, fh), (10, 14, 23), -1)
            cv2.line(frame, (0, fh - 40), (fw, fh - 40), (30, 41, 59), 1)

            pred_label = cached_infer.get("predicted_label", "NORMAL").upper()
            sev_score = cached_infer.get("severity_score", 0.0)
            latency = cached_infer.get("processing_time_ms", 0.0)
            hud_info = f"PRED: {pred_label} | SEV: {sev_level.upper()} ({sev_score:.1f}/100) | {latency:.1f}ms"
            cv2.putText(frame, hud_info, (12, fh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, bracket_color, 1, cv2.LINE_AA)

            ts_str = datetime.now().strftime("%H:%M:%S.%f")[:-4]
            cv2.putText(frame, ts_str, (fw - 105, fh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

            # Encode frame to JPEG
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

            frame_idx += 1
            elapsed = time.time() - t_loop_start
            sleep_time = max(0.01, 0.045 - elapsed)
            time.sleep(sleep_time)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()

def fallback_explain(category, defect_type, severity_level, severity_score, defect_area_pct, confidence_pct):
    templates = [
        f"Industrial inspection of the {category} identified a {defect_type} defect. "
        f"The defect covers approximately {defect_area_pct}% of the surface area. "
        f"Based on these measurements, the anomaly has been classified as {severity_level} (severity score: {severity_score}/100) with a classification confidence of {confidence_pct}%.",
        
        f"A {defect_type} anomaly was detected on the inspected {category}. "
        f"This issue affects {defect_area_pct}% of the product surface. "
        f"The inspection engine assigned a severity score of {severity_score}/100 ({severity_level} level) with a classifier confidence of {confidence_pct}%.",
        
        f"Inspection completed for the {category} component, revealing a {defect_type} defect. "
        f"The anomalous region spans {defect_area_pct}% of the surface area. "
        f"The system has classified this as a {severity_level} issue (severity score: {severity_score}/100, confidence: {confidence_pct}%)."
    ]
    idx = int(severity_score) % len(templates)
    return templates[idx]

# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(title="DefectVision AI — Live Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

active_batches = {}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail if isinstance(exc.detail, str) else "HTTP Exception", "detail": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Server Error", "detail": str(exc)}
    )

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
def read_root():
    return {
        "status": "ready",
        "device": str(device),
        "llm_loaded": llm_loaded,
        "categories": known_categories
    }

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), category: str = Form(...)):
    if category not in known_categories:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}. Choose from {known_categories}")

    # Write temporary file
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = inspect_single_image(filepath, category)
        if result is None:
            raise HTTPException(status_code=500, detail="Inference returned empty result.")

        # Overlay heatmap on original image and encode to base64
        dist_map_norm = result.pop("dist_map_norm")
        image_bgr = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay = overlay_heatmap(image_rgb, dist_map_norm)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", overlay_bgr)
        heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

        _, orig_buffer = cv2.imencode(".jpg", image_bgr)
        orig_base64 = base64.b64encode(orig_buffer).decode("utf-8")

        hm = cv2.resize(dist_map_norm, (image_rgb.shape[1], image_rgb.shape[0]))
        hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        _, raw_hm_buffer = cv2.imencode(".jpg", hm)
        raw_hm_base64 = base64.b64encode(raw_hm_buffer).decode("utf-8")

        result["heatmap"] = f"data:image/jpeg;base64,{heatmap_base64}"
        result["original_image"] = f"data:image/jpeg;base64,{orig_base64}"
        result["raw_heatmap"] = f"data:image/jpeg;base64,{raw_hm_base64}"
        return result
    finally:
        # Cleanup temp file
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/api/inspect-frame")
@app.post("/inspect-frame")
async def inspect_frame_endpoint(
    frame: UploadFile = File(None),
    file: UploadFile = File(None),
    category: str = Form("bottle")
):
    """
    Accepts raw frame bytes from browser webcam or edge sensor.
    Decodes in-memory with zero disk write, runs resident PatchCore + EfficientNet-B0,
    and returns comprehensive optical defect telemetry in under ~80ms.
    """
    upload = frame or file
    if upload is None:
        raise HTTPException(status_code=400, detail="Missing frame or file image payload.")

    contents = await upload.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image buffer into OpenCV matrix.")

    if category not in known_categories and category != "all":
        category = "bottle"

    result = inspect_frame_in_memory(image_bgr, category)
    return result

@app.get("/api/stream/{category}")
@app.get("/stream/{category}")
def stream_endpoint(category: str):
    """
    Real-time multipart MJPEG live stream.
    Directly interfaces with hardware webcam or falls back to animated industrial
    feed with moving laser reticles, neon brackets, and real-time telemetry HUD.
    """
    if category not in known_categories and category != "all":
        category = "bottle"
    return StreamingResponse(
        generate_stream_frames(category),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Background worker for batch inspection
def process_batch_worker(batch_id, files_data, category):
    batch_dir = os.path.join(BATCH_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    
    records = []
    total = len(files_data)
    active_batches[batch_id] = {
        "status": "processing",
        "progress": 0,
        "total": total,
        "report": None
    }
    
    try:
        for idx, (filename, content) in enumerate(files_data):
            fp = os.path.join(temp_dir, filename)
            with open(fp, "wb") as buffer:
                buffer.write(content)
                
            res = inspect_single_image(fp, category)
            if res:
                res.pop("dist_map_norm", None)
                # Keep file reference friendly
                res["filename"] = filename
                records.append(res)
                
            active_batches[batch_id]["progress"] = idx + 1
            
        results_df = pd.DataFrame(records)
        results_df.to_csv(os.path.join(batch_dir, "results.csv"), index=False)
        
        report = build_batch_report(results_df, batch_id)
        report["results"] = results_df.to_dict("records")
        with open(os.path.join(batch_dir, "batch_report.json"), "w") as f:
            json.dump(report, f, indent=2)
            
        active_batches[batch_id]["status"] = "completed"
        active_batches[batch_id]["report"] = report
    except Exception as e:
        active_batches[batch_id]["status"] = f"failed: {str(e)}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/batch")
async def batch_endpoint(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...), category: str = Form(...)):
    if category not in known_categories:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")
        
    batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    
    # Read files memory content to pass to thread safely
    files_data = []
    for f in files:
        content = await f.read()
        files_data.append((f.filename, content))
        
    background_tasks.add_task(process_batch_worker, batch_id, files_data, category)
    return {"batch_id": batch_id, "status": "queued"}

@app.get("/batch/{batch_id}/status")
def batch_status_endpoint(batch_id: str):
    if batch_id not in active_batches:
        # check if it exists in files
        report_path = os.path.join(BATCH_DIR, batch_id, "batch_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            return {"batch_id": batch_id, "status": "completed", "progress": report["total_inspected"], "total": report["total_inspected"], "report": report}
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return active_batches[batch_id]

@app.post("/similarity")
async def similarity_endpoint(file: UploadFile = File(...), category: str = Form(...)):
    if category not in known_categories:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")
    if category not in loaded_faiss_indices:
        raise HTTPException(status_code=400, detail=f"No FAISS index available for {category}")
    if embedding_extractor is None:
        raise HTTPException(
            status_code=404,
            detail="Embedding extractor model artifact not found. Please train model or see REPRODUCIBILITY.md."
        )

    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Load and transform image
        img_bgr = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = eff_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            emb = embedding_extractor(tensor)[0].cpu().numpy()
            
        # Query FAISS index
        faiss_index = loaded_faiss_indices[category]
        top_matches = faiss_index.search(emb, k=5)
        
        return {"matches": top_matches}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

class ExplainPayload(BaseModel):
    category: str
    defect_type: str
    severity_level: str
    severity_score: float
    defect_area_pct: float
    confidence_pct: float

@app.post("/explain")
def explain_endpoint(payload: ExplainPayload):
    if llm_loaded:
        prompt_msgs = build_explanation_prompt(
            category=payload.category,
            defect_type=payload.defect_type,
            severity_level=payload.severity_level,
            severity_score=payload.severity_score,
            defect_area_pct=payload.defect_area_pct,
            confidence_pct=payload.confidence_pct
        )
        try:
            raw_explanation = generate_explanation(llm_pipeline, llm_tokenizer, prompt_msgs)
            cleaned = clean_explanation(raw_explanation)
            return {"explanation": cleaned}
        except Exception as e:
            print(f"LLM explanation error: {e}")
            
    # Fallback template explanation
    explanation = fallback_explain(
        category=payload.category,
        defect_type=payload.defect_type,
        severity_level=payload.severity_level,
        severity_score=payload.severity_score,
        defect_area_pct=payload.defect_area_pct,
        confidence_pct=payload.confidence_pct
    )
    return {"explanation": explanation}

class RagPayload(BaseModel):
    question: str

@app.post("/rag/ask")
def rag_ask_endpoint(payload: RagPayload):
    # Semantic retrieval from KB
    retrieved_entries = kb_index.search(payload.question, k=3)
    defect_types = [entry["defect_type"] for entry in retrieved_entries]
    
    if llm_loaded:
        prompt_msgs = build_rag_prompt(payload.question, retrieved_entries)
        try:
            answer = generate_explanation(llm_pipeline, llm_tokenizer, prompt_msgs, max_new_tokens=150)
            return {
                "answer": answer,
                "retrieved_defect_types": defect_types
            }
        except Exception as e:
            print(f"RAG LLM error: {e}")
            
    # Fallback semantic RAG lookup
    matched_entry = None
    for entry in retrieved_entries:
        if entry["defect_type"].lower() in payload.question.lower():
            matched_entry = entry
            break
    if not matched_entry and retrieved_entries:
        matched_entry = retrieved_entries[0]
        
    if matched_entry:
        answer = (
            f"Regarding {matched_entry['defect_type']}: it is commonly caused by {matched_entry['cause']}. "
            f"The operational impact is {matched_entry['impact']}. "
            f"The recommended solution is {matched_entry['solution']}, and it can be prevented by {matched_entry['prevention']}."
        )
    else:
        answer = "I searched the knowledge base, but couldn't find matching defect types. Try asking about specific defects like scratch, crack, or contamination."
        
    return {
        "answer": answer,
        "retrieved_defect_types": defect_types
    }

class AskPayload(BaseModel):
    question: str

@app.post("/dashboard/ask")
def ask_dashboard_endpoint(payload: AskPayload):
    question = payload.question
    intent = route_intent(question)
    
    if not intent:
        return {
            "answer": "I'm sorry, I couldn't identify the dashboard analytic you are asking about. You can ask about defect rates, most common defects, best model comparison, severity distribution, or production drift.",
            "source": "template fallback",
            "intent": None
        }
        
    try:
        if intent == "most_common_defect":
            cat = extract_category(question, known_categories)
            res = analytics.most_common_defect(cat)
            if res:
                ans = f"The most common defect{' for ' + cat if cat else ''} is '{res['defect_type']}', appearing in {res['count']} out of {res['total_defective']} defective cases."
            else:
                ans = "No defective records found."
        elif intent == "defect_rate_by_category":
            res = analytics.defect_rate_by_category()
            rates_str = ", ".join([f"{k}: {v}%" for k, v in res.items()])
            ans = f"Here are the defect rates by product category: {rates_str}."
        elif intent == "severity_summary":
            res = analytics.severity_summary()
            ans = f"The average severity score of defective items is {res['mean_severity_defective']}/100. Breakdown: Critical ({res['pct_critical']}% of cases), Moderate ({res['pct_moderate']}% of cases), and Minor ({res['pct_minor']}% of cases)."
        elif intent == "model_ranking":
            res = analytics.model_ranking()
            rankings = ", ".join([f"{r['Model']} (AUROC: {r['Mean Category AUROC']})" for r in res])
            ans = f"The models ranked by performance are: {rankings}."
        elif intent == "drift_summary":
            res = analytics.drift_summary()
            drift_str = "; ".join([f"{k} has {v['alerted']}/{v['total']} alerts ({v['alert_rate_pct']}%, status: {v['status']})" for k, v in res.items()])
            ans = f"Production stream drift analysis: {drift_str}."
        elif intent == "active_learning_efficiency":
            res = analytics.active_learning_efficiency(10)
            ans = f"With a review budget of {res['pct_of_dataset']}%, we can capture {res['pct_of_total_errors_captured']}% of all errors, flagging {res['n_flagged']} uncertain images."
        elif intent == "worst_performing_category":
            res = analytics.worst_performing_category("PatchCore")
            ans = f"The weakest category for PatchCore is '{res['category']}' with an AUROC score of {res['auroc']}."
        else:
            ans = "I routed the intent but could not compute the answer."
            
        return {
            "answer": ans,
            "source": "template fallback",
            "intent": intent
        }
    except Exception as e:
        return {
            "answer": f"Error calculating dashboard analytic: {str(e)}",
            "source": "error",
            "intent": intent
        }

@app.post("/rca/ask")
def ask_rca_endpoint(payload: AskPayload):
    question = payload.question
    q = question.lower()
    
    try:
        if "correlation" in q or "disagreement" in q or "agree" in q:
            res = rca.confidence_severity_correlation()
            ans = f"The correlation coefficient between model confidence and anomaly severity score is {res['correlation']} (based on {res['n_samples']} samples), indicating a weak/moderate tracking relationship."
        elif "low confidence" in q or "high severity" in q or "review" in q or "flag" in q:
            res = rca.low_confidence_high_severity_cases()
            ans = f"There are {res['count']} cases ({res['pct_of_defective']}% of defective items) that exhibit high severity but low model confidence. These are key targets for manual expert review. The top categories flagged are: {', '.join([f'{k} ({v} cases)' for k, v in res['example_categories'].items()])}."
        elif "area" in q or "size" in q:
            res = rca.area_severity_relationship()
            ans = f"The correlation between the physical defect area percentage and the overall severity score is {res['correlation']} (n={res['n_samples']})."
        elif "weakest" in q or "struggle" in q or "auroc" in q:
            res = rca.weakest_detection_categories()
            ans = f"Based on the evaluation AUROC, the categories where the detection model struggles the most are: {', '.join([f'{r['category']} (avg AUROC: {r['avg_auroc']})' for r in res])}."
        else:
            # default to defect type severity ranking
            cat = extract_category(question, known_categories)
            res = rca.defect_type_severity_ranking(cat)
            if res:
                ranking_str = ", ".join([f"{r['defect_type']} (avg severity: {r['mean']})" for r in res[:5]])
                ans = f"Here is the severity ranking of defect types{' for ' + cat if cat else ''}: {ranking_str}."
            else:
                ans = "Could not compute root cause patterns. Try asking about confidence-severity correlation, high-severity low-confidence cases, defect area relationship, or weakest detection categories."
                
        return {
            "answer": ans,
            "source": "template fallback",
            "analysis_type": "rca"
        }
    except Exception as e:
        return {
            "answer": f"Error performing RCA analysis: {str(e)}",
            "source": "error",
            "analysis_type": "rca"
        }

@app.get("/models/comparison")
def get_models_comparison():
    df = pd.read_csv(os.path.join(REPORTS_DIR, "step8_model_comparison_FAIR.csv"))
    df = df.fillna("")
    return df.to_dict("records")

@app.get("/severity/logs")
def get_severity_logs(category: str = None, severity_level: str = None, page: int = 1, page_size: int = 20):
    df = pd.read_csv(os.path.join(LOGS_DIR, "step9_severity_log.csv"))
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if severity_level:
        df = df[df["severity_level"].str.lower() == severity_level.lower()]
        
    total_records = len(df)
    total_pages = (total_records + page_size - 1) // page_size
    
    df = df.sort_values("severity_score", ascending=False)
    
    start = (page - 1) * page_size
    end = start + page_size
    records = df.iloc[start:end].to_dict("records")
    
    # Pre-cache explanation lookup for benchmark
    exp_path = os.path.join(LOGS_DIR, "step15_llm_explanations.csv")
    exp_map = {}
    if os.path.exists(exp_path):
        try:
            _edf = pd.read_csv(exp_path)
            exp_map = dict(zip(_edf["filepath"], _edf["explanation"]))
        except Exception:
            pass

    for idx, r in enumerate(records):
        r["id"] = f"DV-{(page - 1) * page_size + idx + 1:04d}"
        r["filename"] = os.path.basename(r["filepath"])
        cat = r.get("category", "")
        demo_file = f"{cat}_similarity_demo.png"
        if os.path.exists(os.path.join("outputs", "similarity_search_samples", demo_file)):
            r["display_image_url"] = f"/outputs/similarity_search_samples/{demo_file}"
        else:
            r["display_image_url"] = None

        fp = r.get("filepath", "")
        if fp in exp_map:
            r["explanation"] = exp_map[fp]
        else:
            r["explanation"] = fallback_explain(
                category=r.get("category", ""),
                defect_type=r.get("defect_type", "anomaly"),
                severity_level=r.get("severity_level", "Moderate"),
                severity_score=float(r.get("severity_score", 50.0)),
                defect_area_pct=float(r.get("defect_area_pct", 5.0)),
                confidence_pct=float(r.get("confidence_pct", 95.0))
            )
        
    return {
        "records": records,
        "page": page,
        "page_size": page_size,
        "total_records": total_records,
        "total_pages": total_pages
    }

@app.get("/drift")
def get_drift():
    df = pd.read_csv(os.path.join(LOGS_DIR, "step12_drift_log_v3.csv"))
    df = df.fillna("")
    return df.to_dict("records")

@app.get("/rag/entries")
def get_rag_entries():
    kb_path = os.path.join(MODELS_DIR, "rag_knowledge_base", "defect_knowledge_base.csv")
    if os.path.exists(kb_path):
        df = pd.read_csv(kb_path)
        df = df.fillna("")
        return df.to_dict("records")
    return []

@app.get("/categories/samples")
def get_category_samples():
    samples_dir = os.path.join("outputs", "similarity_search_samples")
    results = []
    for cat in known_categories:
        demo_file = f"{cat}_similarity_demo.png"
        sample_path = os.path.join(samples_dir, demo_file)
        has_sample = os.path.exists(sample_path)
        results.append({
            "category": cat,
            "has_sample": has_sample,
            "sample_url": f"/outputs/similarity_search_samples/{demo_file}" if has_sample else None,
            "baseline": baseline_lookup.get(cat, {})
        })
    return results

@app.get("/active-learning/sensitivity")
def get_active_learning_sensitivity():
    sens_path = os.path.join(REPORTS_DIR, "step11_threshold_sensitivity.csv")
    if os.path.exists(sens_path):
        df = pd.read_csv(sens_path)
        df = df.fillna("")
        return df.to_dict("records")
    return []

@app.get("/reports/summary")
def get_reports_summary():
    summary_path = os.path.join(REPORTS_DIR, "final_research_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}

@app.get("/reports/auroc-matrix")
def get_auroc_matrix():
    auroc_path = os.path.join(REPORTS_DIR, "step8_per_category_auroc_matrix.csv")
    if os.path.exists(auroc_path):
        df = pd.read_csv(auroc_path)
        df = df.fillna("")
        return df.to_dict("records")
    return []

@app.get("/samples/presets")
def get_sample_presets():
    """Curated list of test samples across categories for 1-click inspection."""
    descriptions = {
        "bottle": "Large structural fracture along bottle sidewall",
        "cable": "Cable missing conductor strand / deformed wire",
        "capsule": "Capsule squeeze and surface coating defect",
        "carpet": "Carpet weave puncture / localized fiber defect",
        "grid": "Metal grid bent wire and mesh distortion",
        "hazelnut": "Cracked shell and outer hull hole",
        "leather": "Surface fold, crease, and color stain defect",
        "metal_nut": "Nut thread scratch / heavy metal indentation",
        "pill": "Pill contamination / crack on tablet coating",
        "screw": "Thread damage / deformed screw head",
        "tile": "Ceramic tile deep crack / glaze scratch",
        "toothbrush": "Bristle defect / misaligned tuft pattern",
        "transistor": "Bent lead / semiconductor casing crack",
        "wood": "Wood grain scratch and surface hole",
        "zipper": "Broken zipper tooth / fabric tape tear"
    }
    samples = []
    samples_dir = os.path.join("outputs", "similarity_search_samples")
    for cat in known_categories:
        demo_file = f"{cat}_similarity_demo.png"
        sample_path = os.path.join(samples_dir, demo_file)
        if os.path.exists(sample_path):
            samples.append({
                "category": cat,
                "title": f"{cat.replace('_', ' ').title()}",
                "filename": demo_file,
                "image_url": f"/outputs/similarity_search_samples/{demo_file}",
                "description": descriptions.get(cat, f"Industrial defect specimen for {cat}"),
                "baseline": baseline_lookup.get(cat, {})
            })
    return samples

@app.get("/batch/list")
def list_batches():
    """Returns past and active batch inspections."""
    batches = []
    if os.path.exists(BATCH_DIR):
        for name in os.listdir(BATCH_DIR):
            bpath = os.path.join(BATCH_DIR, name)
            if os.path.isdir(bpath):
                report_file = os.path.join(bpath, "batch_report.json")
                if os.path.exists(report_file):
                    try:
                        with open(report_file, "r") as f:
                            rep = json.load(f)
                            batches.append({
                                "batch_id": name,
                                "total_inspected": rep.get("total_inspected", 0),
                                "defective_count": rep.get("defective_count", 0),
                                "normal_count": rep.get("normal_count", 0),
                                "defect_rate_pct": rep.get("defect_rate_pct", 0.0),
                                "severity_distribution": rep.get("severity_distribution", {}),
                                "timestamp": rep.get("timestamp", "")
                            })
                    except Exception:
                        pass
    # Sort newest first
    batches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return batches

@app.get("/similarity/category/{category}")
def get_category_similarity_gallery(category: str):
    """Returns representative gallery items from the category FAISS index."""
    if category not in known_categories:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")
    if category not in loaded_faiss_indices:
        raise HTTPException(status_code=400, detail=f"No FAISS index available for {category}")

    faiss_index = loaded_faiss_indices[category]
    results = []
    for item in faiss_index.metadata[:8]:
        meta = dict(item)
        meta["filename"] = os.path.basename(meta.get("filepath", ""))
        demo_file = f"{category}_similarity_demo.png"
        meta["sample_url"] = f"/outputs/similarity_search_samples/{demo_file}"
        results.append(meta)
    return {"category": category, "gallery": results}

# -----------------------------
# Start FastAPI server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
