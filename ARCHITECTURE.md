# DefectVision AI — System Architecture Specification

**Document Version:** 1.0.0  
**Status:** Frozen Research Prototype  
**Scope:** Full-Stack Architecture, Optical Pipeline, and Selective Triage Engine  

---

## 1. System Topology Overview

DefectVision AI is engineered as an asynchronous, decoupled full-stack platform designed for high-throughput visual inspection and human-in-the-loop quality management.

```
+---------------------------------------------------------------------------------------------------+
|                                      SYSTEM ARCHITECTURE LAYOUT                                   |
+---------------------------------------------------------------------------------------------------+

    [ CLIENT PRESENTATION LAYER ]
    +---------------------------------------------------------------------------------------+
    |  Single Page Application (React 19 + TypeScript + Tailwind CSS v4)                    |
    |  - Three.js / React Three Fiber: Interactive 3D Digital Twin Viewer                   |
    |  - Lucide Icons & Recharts: Production KPIs, Escape Rates, and Cost Trade-offs        |
    |  - Interactive Heatmap Viewer: Raw vs. Anomaly Overlay Toggle                         |
    |  - Selective Inspection Console: Budget Slider (K) & Priority Queue                   |
    |  - RAG Copilot Panel: Natural Language Defect Remediation Advice                      |
    +---------------------------------------------------------------------------------------+
                                              |
                                              |  HTTP REST / WebSocket (Port 5173 -> Port 8000)
                                              v
    [ REVERSE PROXY & API ROUTING LAYER ]
    +---------------------------------------------------------------------------------------+
    |  Vite Development Server Proxy (Local Dev) / Uvicorn ASGI Server                      |
    |  - URL Rewrite: /api/* -> http://127.0.0.1:8000/*                                     |
    |  - CORS Middleware: Header Injection & Preflight Handling                             |
    +---------------------------------------------------------------------------------------+
                                              |
                                              v
    [ BACKEND SERVICE LAYER (FastAPI / Python 3.12) ]
    +---------------------------------------------------------------------------------------+
    |  app.py — Core Asynchronous API Service                                               |
    |  - Route Handlers: /predict, /batch, /severity/logs, /rag/ask, /samples/presets       |
    |  - Task Queue: Background lot processing with threadpool executor                     |
    |  - State Manager: In-memory FAISS indices & cached model backbones                   |
    +---------------------------------------------------------------------------------------+
                                              |
                         +--------------------+--------------------+
                         |                                         |
                         v                                         v
    [ OPTICAL ANOMALY ENGINE ]                         [ EXPLAINABILITY & TRIAGE ENGINE ]
    +-----------------------------------------+        +----------------------------------------+
    | 1. PatchCore Anomaly Detector           |        | 1. Dual-Stream Heatmap Generator       |
    |    - Backbone: WideResNet-50            |        |    - PatchCore Gaussian-smoothed Map   |
    |    - Feature: Mid-level layers 2 & 3    |        |    - EfficientNet Grad-CAM Activation  |
    |    - Index: FAISS FlatL2 Memory Bank    |        | 2. Calibrated Cost-Sensitive Triage    |
    |    - Decision: Threshold tau_c (p99.5)  |        |    - Platt Logistic Probability Model  |
    | 2. EfficientNet-B0 Classifier           |        |    - Cost Matrix (c_FN >> c_FP)        |
    |    - Global Structural Integrity        |        |    - Greedy Bayes Risk Sorting         |
    |    - Fine-tuned on multi-class defects  |        | 3. RAG Knowledge Retriever             |
    | 3. Hybrid Decision Orchestrator         |        |    - Embedded Defect Manual            |
    |    - Logical consensus & severity gate  |        |    - Cosine similarity retrieval       |
    +-----------------------------------------+        +----------------------------------------+
                         |                                         |
                         +--------------------+--------------------+
                                              |
                                              v
    [ PERSISTENCE & DATA STORAGE LAYER ]
    +---------------------------------------------------------------------------------------+
    |  - Model Checkpoints: best_model.pt (PyTorch weights), memory_bank.npy (Embeddings)   |
    |  - Calibration Parameters: decision_thresholds.json, logistic_calibration_models.pkl  |
    |  - Audit Logs: batch_inspections/*/report.json, severity_logs.json                    |
    |  - Presets & Static Samples: similarity_search_samples/*                              |
    +---------------------------------------------------------------------------------------+
```

---

## 2. Core Subsystems and Data Flow

### 2.1 Optical Anomaly Detection Engine

The detection pipeline processes images through an unsupervised memory-bank localization workflow coupled with a supervised classification check:

```
[ Input Image (224x224x3) ]
             |
             +---------------------------------------+
             |                                       |
             v                                       v
[ PatchCore Extractor (WideResNet-50) ]   [ EfficientNet-B0 Backbone ]
             |                                       |
             | Extract patch tokens                  | Forward pass
             | Layers 2 & 3 (dim: 384)               | Feature vector (dim: 1280)
             v                                       v
[ FAISS IndexFlatL2 Query ]               [ Linear Classification Head ]
             |                                       |
             | Find top-k nearest neighbors          | Softmax probability vector
             | Compute maximum L2 distance           | Class: {normal, scratch, broken, etc.}
             v                                       v
[ Anomaly Score S(X) ]                    [ Classifier Decision Y_eff ]
             |                                       |
             +-------------------+-------------------+
                                 |
                                 v
                     [ Hybrid Decision Gate ]
```

#### Decision Rules:
1. **Primary PatchCore Rule:**
   $$\hat{Y}_{\text{PC}} = \mathbb{I}(S(X) \ge \tau_c)$$
   where $\tau_c$ is calibrated at the $99.5^{\text{th}}$ percentile of nominal validation samples.
2. **Hybrid Decision Rule:**
   $$\hat{Y}_{\text{hybrid}} = \hat{Y}_{\text{PC}} \lor \hat{Y}_{\text{eff}}$$
   If either model flags an anomaly, the specimen is escalated for review or flagged as defective.

---

### 2.2 Dual-Stream Visual Explainability

To provide actionable feedback to quality control operators, DefectVision AI renders two synchronized visual explainability streams:

1. **PatchCore Anomaly Heatmap:**
   - Reshapes patch-level nearest-neighbor distances into a 2D spatial grid corresponding to image coordinates ($28 \times 28$).
   - Applies bilinear interpolation to upsample the grid to the native image resolution ($224 \times 224$).
   - Smooths the heatmap using a Gaussian filter ($\sigma = 4.0$).
   - Normalizes values to $[0, 1]$ and applies an OpenCV `COLORMAP_JET` overlay.
2. **Grad-CAM Attention Map:**
   - Computes gradients of the target defect class with respect to the final convolutional feature maps of EfficientNet-B0.
   - Computes global average pooled gradients as channel importance weights.
   - Generates a coarse localization map highlighting regions driving the supervised classification.

Both maps are base64-encoded and transmitted to the frontend for side-by-side or blend-overlay inspection.

---

### 2.3 Decision-Theoretic Selective Triage Engine

When production volume exceeds manual review bandwidth, the selective review engine triages specimens using calibrated cost-sensitive risk ranking (Chow 1970; Elkan 2001):

1. **Probability Calibration:**
   Maps continuous margin $S(X) - \tau_c$ to calibrated defect probability $\hat{p}_d(X)$ using Platt scaling (univariate logistic regression):
   $$\hat{p}_d(X) = \sigma(\beta_0 + \beta_1 (S(X) - \tau_c))$$
2. **Expected Automated Loss Calculation:**
   $$\mathbb{E}[L_{\text{auto}} \mid X, \hat{Y}] = \begin{cases} \hat{p}_d(X) \cdot c_{\text{FN}}, & \text{if } \hat{Y} = 0 \\ (1 - \hat{p}_d(X)) \cdot c_{\text{FP}}, & \text{if } \hat{Y} = 1 \end{cases}$$
3. **Marginal Utility of Human Review:**
   Assuming expert human review ($p_{\text{sens}}=1, p_{\text{spec}}=1$) at capacity cost $c_{\text{rev}}$:
   $$\Delta L(X) = \mathbb{E}[L_{\text{auto}}] - c_{\text{rev}}$$
4. **Greedy Knapsack Allocation:**
   Sorts specimens in descending order of $\Delta L(X)$ and selects the top $K$ specimens, guaranteeing integer-optimal risk reduction under uniform review capacity.

---

## 3. Asynchronous Batch Processing Lifecycle

Batch inspections run as detached asynchronous jobs to prevent blocking the REST event loop:

```
Client                             FastAPI Backend                      Worker Thread
  |                                       |                                   |
  |--- POST /batch (files, category) ---->|                                   |
  |                                       |--- Create batch_id & directory -->|
  |                                       |--- Spawn background task -------->|
  |<-- Return 202 Accepted (batch_id) ----|                                   |
  |                                       |                                   |--- Process image 1
  |--- GET /batch/{id}/status ----------->|                                   |--- Process image 2
  |<-- Return {status: processing, 2/10}--|                                   |--- ...
  |                                       |                                   |--- Generate report.json
  |--- GET /batch/{id}/status ----------->|                                   |--- Mark complete
  |<-- Return {status: completed, report}-|<-- Read report.json --------------|
```

---

## 4. Hardware and Concurrency Characteristics

- **Inference Latency:** $\approx 35 - 50\text{ ms}$ per specimen on modern multi-core x86-64 CPUs.
- **Memory Scaling:** Each loaded category FAISS index requires $\approx 25 - 45\text{ MB}$ of RAM. All 15 MVTec categories loaded simultaneously consume $<600\text{ MB}$ of system memory.
- **Threading Model:** FastAPI asynchronous event loop handles network I/O; CPU-bound PyTorch and FAISS operations execute in threadpool workers with PyTorch thread affinity (`torch.set_num_threads(4)`).
