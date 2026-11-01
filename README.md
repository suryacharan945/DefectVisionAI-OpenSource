# DefectVision AI — Industrial Visual Anomaly Detection & Selective Inspection Platform

[![Release Tag](https://img.shields.io/badge/Release-v1.0.0--research--prototype-blue.svg)](https://github.com/suryacharan945/DefectVision-AI/releases/tag/v1.0.0-research-prototype)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![React: 19](https://img.shields.io/badge/React-19.2-cyan)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.7%2B-green)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal)](https://fastapi.tiangolo.com/)

> **Status:** Research Prototype & Open-Source Benchmark.  
> **Target Audience:** Computer vision scientists, quality assurance automation engineers, and MLOps researchers.  
> **Key Differentiator:** Combines state-of-the-art unsupervised memory-bank anomaly detection (**PatchCore**) with supervised classifier verification (**EfficientNet-B0**), dual explainability heatmaps, and a decision-theoretic **selective human-review triage engine** designed to optimize quality-control economics under finite inspection budgets.

---

## 1. Project Overview

Manufacturing visual inspection faces a fundamental economic dilemma: fully automated models cannot guarantee zero defect escapes without triggering high false-alarm rates that paralyze production lines, yet complete manual inspection of every part is physically and economically impossible.

**DefectVision AI** bridges this gap by implementing a production-grade, human-in-the-loop (HITL) quality control architecture:
1. **Unsupervised Localization (PatchCore):** Uses a pre-trained WideResNet-50 backbone with greedy minimax coreset-subsampled memory banks indexed in FAISS to detect fine surface, texture, and structural defects without requiring labeled anomaly training data.
2. **Supervised Verification (EfficientNet-B0):** Evaluates global part integrity to catch gross structural anomalies and fractured components.
3. **Dual-Stage Explainability:** Generates pixel-level nearest-neighbor distance heatmaps (PatchCore) alongside activation maps (Grad-CAM) to pinpoint defect coordinates.
4. **Decision-Theoretic Triage:** Leverages calibrated cost-sensitive risk ranking (Chow, 1970; Elkan, 2001) to allocate a finite human visual inspection budget ($K$) to the highest-risk parts, reducing quality-control loss by over $30\%$ compared to standard heuristics.

---

## 2. Architecture & System Topology

DefectVision AI is structured as a decoupled full-stack architecture optimized for local workstation deployment or on-premise industrial edge nodes:

```
+---------------------------------------------------------------------------------------+
|                                    DEFECTVISION AI                                    |
+---------------------------------------------------------------------------------------+

       [ Client Layer ]                                     [ Backend API Layer ]
 +--------------------------+                          +-------------------------------+
 |  React 19 + Tailwind v4  |                          |      FastAPI REST Service     |
 |  - 3D Part Inspection    |    HTTP / JSON           |  - /predict (Live Inference)  |
 |  - Interactive Heatmaps  | <=====================>  |  - /batch (Async Queue)       |
 |  - Triage Review Console |    Proxy (:5173 -> :8000)|  - /severity/logs (Audit Trail)|
 |  - RAG Copilot Interface |                          |  - /rag/ask (Root-Cause QA)   |
 +--------------------------+                          +-------------------------------+
                                                                       |
                                                +----------------------+----------------------+
                                                |                                             |
                                    [ Optical Anomaly Engine ]                     [ Explainability & Triage ]
                             +--------------------------------------+       +------------------------------------+
                             | PatchCore (WideResNet-50 + FAISS L2) |       | Dual Heatmaps (PatchCore + GradCAM)|
                             | EfficientNet-B0 Hybrid Rule          |       | Cost-Weighted Bayes Risk Ranking   |
                             | Category Decision Thresholds (p99.5) |       | Calibrated Defect Probabilities    |
                             +--------------------------------------+       +------------------------------------+
```

Detailed architectural blueprints, component specifications, and data flow sequences are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 3. Implemented Capabilities

- **Real-Time Single Specimen Inference:** Sub-50ms CPU steady-state forward inference (~42ms forward pass: WideResNet patch extraction + FAISS FlatL2 search, excluding one-time startup index loading) calculating anomaly score, decision threshold margin, defect area percentage, and severity categorization (Nominal, Low, Moderate, High, Critical).
- **Dual Visual Explainability:** High-resolution normalized PatchCore distance heatmaps overlaid on specimen images, complemented by EfficientNet Grad-CAM attention maps.
- **k-Nearest Neighbor Similarity Search:** FAISS FlatL2 retrieval returning the top-3 most similar training and nominal specimens to validate localization decisions.
- **Asynchronous Batch Inspection:** Multi-file background worker processing production lots, computing batch statistics (defect rate, escape risk), and generating structured export summaries.
- **RAG Defect Engineering Copilot:** Domain-specific contextual retrieval knowledge base answering operator questions regarding defect root causes, material handling adjustments, and preventative maintenance.
- **Selective Human-Review Triage Console:** Ranks specimens under budget constraints ($K$) using decision-theoretic loss reduction to maximize captured defects while minimizing labor cost.
- **Audit Logging & Provenance Tracking:** Complete local logging of inference scores, decisions, timestamps, and operator actions.

---

## 4. Installation & Windows Startup Guide

### Prerequisites
- **Operating System:** Windows 10 / 11 (64-bit), Linux, or macOS.
- **Python:** Version 3.10, 3.11, or 3.12 (Tested on Python 3.12.x).
- **Node.js:** Version 18.x, 20.x, or newer with `npm`.
- **Hardware:** 8 GB RAM minimum (16 GB recommended). Dedicated NVIDIA GPU supported via CUDA, but CPU-only inference is fully supported out-of-the-box via FAISS-CPU and PyTorch-CPU.

### Step 1: Clone Repository
```powershell
git clone https://github.com/suryacharan945/DefectVision-AI.git
cd DefectVision-AI
```

### Step 2: Set Up Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies
```powershell
npm install
```

### Step 4: Launch the Full Stack (One-Click)
Run the provided launcher script:
```powershell
.\start_defectvision.bat
```
This automatically starts:
- **FastAPI Backend:** `http://127.0.0.1:8000` (API documentation at `http://127.0.0.1:8000/docs`)
- **Vite React Frontend:** `http://127.0.0.1:5173`

Open your web browser and navigate to `http://127.0.0.1:5173`.

---

## 5. Model Inference & API Endpoints

The backend exposes fully typed REST endpoints:

| Endpoint | Method | Payload / Parameters | Description |
|:---|:---:|:---|:---|
| `/` | `GET` | None | Service health, device type (`cpu`/`cuda`), and loaded category list. |
| `/predict` | `POST` | `multipart/form-data`: `file` (image), `category` (string) | Synchronous inspection returning anomaly score, threshold, decision, severity, nearest neighbors, and base64 heatmaps. |
| `/batch` | `POST` | `multipart/form-data`: `files` (list of images), `category` | Asynchronous batch job submission returning `batch_id`. |
| `/batch/{id}/status` | `GET` | Path: `id` | Polling endpoint for batch progress and completed inspection report. |
| `/severity/logs` | `GET` | Query: `category`, `page`, `page_size` | Historical audit logs with pagination and severity breakdown. |
| `/rag/ask` | `POST` | JSON: `{"question": "string", "category": "string"}` | Grounded defect engineering QA copilot response. |

---

## 6. Research Findings & Scientific Integrity

A core contribution of DefectVision AI is its commitment to **unvarnished, reproducible scientific reporting**. During the research audit phases (Phases 7 through 9), multiple algorithmic hypotheses were formally implemented, tested, and audited:

### 1. Hybrid Decision Rule Verification (Phase 3)
- An empirical audit of the hybrid `PatchCore OR EfficientNet` rule demonstrated that while EfficientNet effectively catches gross fractures (e.g. broken bottle necks), combining models via a naive OR rule increases false positive rates on borderline normal textures.
- The recommended configuration applies PatchCore as the primary detector and uses classifier consensus for critical severity escalation.

### 2. CPND Uncertainty Hypothesis (Phase 7B / 7C — Negative Finding)
- **Hypothesis:** Conformal Patch-Neighborhood Dispersion (measuring spatial dispersion of retrieved memory bank neighbors) was hypothesized to provide an epistemic uncertainty signal superior to raw decision-boundary distance.
- **Empirical Finding:** Evaluated across 330 development specimens. Density-normalized CPND achieved only a $22.2\%$ review error yield, performing significantly *worse* than raw boundary distance ($30.6\%$). The dispersion hypothesis was formally rejected (`STOP CPND`).

### 3. MACBA Equivalence Finding (Phase 8C — Methodological Finding)
- **Hypothesis:** Multi-Category Adaptive Cost-Weighted Budget Allocation (MACBA) was proposed as a novel multi-line knapsack allocation algorithm.
- **Mathematical & Empirical Proof:** Proved that under additive loss and unit review capacity, greedy knapsack on marginal utility is **mathematically isomorphic** to Chow's (1970) classic cost-weighted Bayes error-risk ranking ($\rho_s = 1.000000$, $100.0\%$ selection identity).
- **Discrepancy Resolution:** An apparent Phase 8B difference at $\lambda = 1$ was traced to an implementation inconsistency where the baseline used an unsigned error model. When evaluated consistently, the methods are identical.

Detailed derivations, proofs, and traces are published in [`research/phase8c_macba_equivalence/`](research/phase8c_macba_equivalence/) and [`release/open_source_readiness/TECHNICAL_CASE_STUDY.md`](release/open_source_readiness/TECHNICAL_CASE_STUDY.md).

---

## 7. Dataset Provenance & Independence

To maintain strict scientific independence:
- **Exposed Development Pool:** 330 images across 3 MVTec AD categories (`bottle`, `capsule`, `metal_nut`) used for Phase 7C and Phase 8B prototype tuning and threshold calibration.
- **VisA Exposure Breakdown (12 Categories Total):** Forensic audit reveals VisA `capsules` is development-exposed via earlier cloud experiments. VisA `candle`, `macaroni1`, `macaroni2`, and `pcb1` are designated **Unknown / Disputed** due to exploratory notes and are excluded from pristine benchmark status. The remaining 7 VisA categories (`pcb2`, `pcb3`, `pcb4`, `cashew`, `chewinggum`, `fryum`, `pipe_fryum`) remain potentially untouched.
- **Quarantined Independent Benchmarks:** The BTAD (3 categories, 2,830 images) and Chukyo fastener datasets remain **completely untouched and unaccessed**, ensuring a pristine benchmark for future independent evaluations.
- **Zero Label Leakage:** All reported threshold and probability calibrations use strictly disjoint splits (50% validation normal/defect), with zero access to holdout test labels.

See [`DATASET_PROVENANCE.md`](DATASET_PROVENANCE.md) and [`release/final_integrity_gate/DATASET_EXPOSURE_RECONCILIATION.md`](release/final_integrity_gate/DATASET_EXPOSURE_RECONCILIATION.md) for full manifest hashes.

---

## 8. Verification & Test Suite

The repository includes a comprehensive, multi-tiered test suite comprising **34 automated software tests** (22 unit tests, 5 backend integration endpoints, 1 model inference check, 1 Vite proxy check, 4 input validation guards, and 1 frontend production build):

```powershell
# 1. Run all unit tests (22 tests covering vision reporting, CPND, and density normalization)
.\.venv\Scripts\python.exe -m unittest discover tests -v

# 2. Run backend end-to-end integration test (Verifies live inference, batching, logs, RAG)
.\.venv\Scripts\python.exe scripts/e2e_backend_test.py

# 3. Run model artifact hash and PatchCore inference verification
.\.venv\Scripts\python.exe scripts/test_backend_inference.py

# 4. Run Vite dev proxy communication test
.\.venv\Scripts\python.exe scripts/test_vite_proxy.py

# 5. Run input validation and error safety test
.\.venv\Scripts\python.exe scripts/test_invalid_inputs_safety.py

# 6. Verify frontend production build
npm run build --prefix frontend
```

All 34 software tests pass with zero errors in under 25 seconds. Note that offline research equivalence comparisons (e.g. 870 MACBA/BL6 trace rows) are classified as empirical research evaluations, not software unit tests. Detailed test classifications are documented in [`release/final_integrity_gate/TEST_CLASSIFICATION_AND_RESULTS.md`](release/final_integrity_gate/TEST_CLASSIFICATION_AND_RESULTS.md).

---

## 9. Known Limitations

- **Research Prototype Status:** DefectVision AI is an advanced research and engineering prototype. It is not certified for turn-key, safety-critical factory deployment without line-specific calibration and compliance validation.
- **Domain Shift Sensitivity:** Memory-bank nearest-neighbor scores are sensitive to lighting shifts, camera optical focal changes, and significant part orientation variance.
- **Sample Complexity of Calibration:** Calibrating continuous anomaly scores via Platt scaling requires at least $20–30$ representative validation anomalies per category; on ultra-rare defect types, uncalibrated boundary distance remains more robust.
- **Synthetic Demonstration Mode:** When deployed in static environments (e.g., Vercel without a connected Python GPU backend), the web UI operates in a clearly labeled synthetic demonstration mode using recorded development presets.

Detailed boundary conditions are documented in [`LIMITATIONS.md`](LIMITATIONS.md).

---

## 10. License & Attribution

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

> **Notice Regarding Pretrained Weights & Feature Banks:**  
> The software architecture and code of DefectVision AI are licensed under the MIT License. However, the precomputed PatchCore memory banks (`models/patchcore/`) and PaDiM models (`models/padim/`) distributed with this repository were computed from images belonging to the MVTec Anomaly Detection dataset. These artifacts are strictly provided for non-commercial academic research and demonstration purposes under MVTec Software GmbH's licensing terms. Commercial users must fit memory banks exclusively on their own proprietary inspection imagery.

### Third-Party Attribution:
- **MVTec AD Dataset:** Bergmann et al., *"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection"*, CVPR 2019 / IJCV 2021.
- **VisA Dataset:** Zou et al., *"SPOT-the-Difference: VisA Dataset for Industrial Anomaly Detection"*, ECCV 2022.
- **PatchCore:** Roth et al., *"Towards Total Recall in Industrial Anomaly Detection"*, CVPR 2022.
- **EfficientNet & Timm:** Tan & Le, ICML 2019; Ross Wightman, *PyTorch Image Models*.
- **FAISS:** Facebook AI Research (Johnson et al., IEEE TBD 2019).
