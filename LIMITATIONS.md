# DefectVision AI — Technical Limitations and Operational Boundary Conditions

**Document Version:** 1.0.0  
**Scope:** Known Constraints, Failure Modes, and Environmental Dependencies  

---

## 1. Prototype Status Disclaimer

DefectVision AI is an **advanced research prototype and open-source scientific benchmark platform**. It is designed to evaluate algorithms, explainability pipelines, and human-in-the-loop triage strategies.

**Explicit Operational Notice:**
- It is **not** certified under industrial quality assurance regulatory standards (e.g., ISO 9001, CE Machine Directive, FDA 21 CFR Part 820).
- It must **not** be deployed as a sole, autonomous gatekeeper for safety-critical components (aerospace structural airframes, medical implants, automotive braking assemblies) without rigorous plant-specific validation and redundant mechanical failsafes.

---

## 2. Hardware and Compute Constraints

### 2.1 CPU vs. GPU Inference Latency
- On modern multi-core x86-64 CPUs, single-image inference requires $\approx 35 - 50\text{ ms}$. This throughput ($\approx 20\text{ parts/second}$) is well-suited for discrete assembly lines, but cannot sustain ultra-high-speed continuous web or can-filling lines operating at $>100\text{ parts/second}$ ($<10\text{ ms}$ budget).
- Achieving sub-10ms latency requires exporting the WideResNet-50 backbone to TensorRT or ONNX Runtime and executing on a dedicated CUDA GPU (e.g., NVIDIA RTX 4090 or Jetson AGX Orin).

### 2.2 Memory Bank Scalability
- PatchCore stores $10\%$ of nominal patch representations in a FAISS index. Each category requires $\approx 25 - 45\text{ MB}$ of RAM.
- While managing 15 categories requires only $\approx 500\text{ MB}$, scaling to a plant with $>500$ distinct stock-keeping units (SKUs) would require dynamic disk-backed caching or Product Quantization (FAISS `IndexIVFPQ`) to prevent memory exhaustion.

---

## 3. Optical and Domain Shift Vulnerabilities

Nearest-neighbor anomaly detection relies on feature proximity in a pre-trained metric space. The system exhibits known sensitivities to environmental variations:

1. **Rigid Spatial Alignment:** PatchCore incorporates spatial context. If a manufactured component undergoes severe angular rotation ($>15^\circ$) or translation outside the training fixture frame, nominal regions will register high nearest-neighbor distances and trigger false positive anomalies.
2. **Illumination and Specular Glare:** Variations in factory lighting (e.g., daylight shifting through skylights, aging LED light bars) alter surface reflectance. On polished metals (e.g., VisA `metal_nut`), specular highlights frequently trigger false alarm spikes.
3. **Camera Optical Drift:** Focal blur, lens dust, or aperture shifts introduce high-frequency texture distortions that degrade image AUROC.

---

## 4. Statistical Calibration and Review Limitations

### 4.1 Sample Complexity of Platt Scaling
- The selective review engine uses Platt scaling (univariate logistic regression) to map score margins to posterior error probabilities.
- On industrial manufacturing lines where defects are extremely rare ($<0.1\%$), obtaining the $20–30$ representative validation anomalies required for robust logistic fitting is difficult.
- In low-data regimes ($N_{\text{defect}} < 10$), Platt scaling exhibits high parameter variance. Under severe miscalibration, simple non-parametric boundary distance ranking is more reliable.

### 4.2 The "Perfect Reviewer" Assumption
- The primary economic cost reduction metrics reported in Phase 8B assume an expert human reviewer with $100\%$ sensitivity and specificity ($p_{\text{sens}} = 1.0, p_{\text{spec}} = 1.0$).
- In real factory environments, human visual inspectors suffer from cognitive fatigue, eye strain, and subjective bias. If inspector sensitivity drops below $85\%$, the realized cost advantage of selective review degrades significantly.

---

## 5. Deployment Modes: Local vs. Cloud Demonstration

- **Local Workstation Mode (Recommended):** Connects the React frontend directly to the local FastAPI backend, executing genuine PyTorch inference, FAISS memory-bank querying, and dynamic heatmap rendering on user-uploaded images.
- **Cloud Demonstration Mode (Vercel):** When the frontend is hosted on static serverless platforms (e.g., Vercel) without a live Python GPU backend, the application operates in a **clearly labeled synthetic demonstration mode**, serving pre-extracted development presets (`bottle`, `capsule`, `metal_nut`) rather than executing real-time server-side PyTorch passes.
