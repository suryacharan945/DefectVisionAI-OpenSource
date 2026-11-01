# DefectVision AI — Model Card

**Model Name:** DefectVision AI Optical Anomaly Suite  
**Version:** 1.0.0-research-prototype  
**Release Date:** November 2026  
**License:** MIT License  

---

## 1. Model Details

### 1.1 Architecture Specifications
The DefectVision AI system utilizes two complementary neural network models:

1. **PatchCore Anomaly Detector:**
   - **Backbone Network:** WideResNet-50-2 pre-trained on ImageNet-1k.
   - **Feature Extraction Layers:** Mid-level feature maps from `layer2` and `layer3`, combined via adaptive average pooling to create local neighborhood patch representations (dimension $d = 384$).
   - **Memory Subsampling:** Greedy minimax facility location coreset reduction retaining $10\%$ of patch tokens.
   - **Indexing & Retrieval:** FAISS `IndexFlatL2` (exact Euclidean distance search).
   - **Decision Rule:** $k\text{NN}$ distance ($k=3$) compared against category-specific empirical threshold $\tau_c$.

2. **EfficientNet-B0 Classifier:**
   - **Backbone Network:** EfficientNet-B0 (Tan & Le, 2019).
   - **Pre-training:** ImageNet pre-trained weights fine-tuned on multi-category industrial defect samples.
   - **Head:** Linear classification head producing class logits for defect categorization.
   - **Explainability:** Grad-CAM backpropagation targeting final convolutional feature maps (`features[8]`).

3. **Platt Probability Calibrator:**
   - **Model:** Univariate logistic regression fitted on signed score margin $S(X) - \tau_c$.
   - **Objective:** Predicts posterior defect probability $\hat{P}(Y=1 \mid S(X) - \tau_c) \in [0, 1]$.

---

## 2. Intended Use and Target Domain

### 2.1 Primary Intended Applications
- **Quality Control Screening:** High-throughput screening of manufactured components in industrial manufacturing lines.
- **Operator Decision Support:** Presenting localized anomaly heatmaps and defect root-cause suggestions to human visual inspectors.
- **Selective Review Triage:** Prioritizing specimens with ambiguous anomaly margins for manual audit under strict inspection capacity limits.
- **Academic & Research Benchmarking:** Evaluating selective classification algorithms, calibration stability, and human-in-the-loop workflows.

### 2.2 Out-of-Scope and Prohibited Uses
- **Autonomous Safety-Critical Certification:** Deploying the model without human oversight in life-critical domains (e.g., aerospace turbine manufacturing, nuclear component fabrication, implantable medical devices).
- **Uncalibrated Distributional Shifts:** Operating on novel object geometries, altered lighting configurations, or camera resolutions without re-calibrating decision thresholds $\tau_c$.
- **Surveillance and Non-Industrial Imaging:** The models are strictly calibrated for industrial surface quality inspection and are unsuitable for general natural image classification or facial detection.

---

## 3. Training and Calibration Data

### 3.1 Training Data (PatchCore Nominal Fitting)
- **Source:** MVTec Anomaly Detection (MVTec AD) benchmark.
- **Condition:** 100% nominal (defect-free) images. No defective specimens are ever included in PatchCore memory banks.
- **Resolution:** Resized and center-cropped to $224 \times 224$ pixels, normalized using standard ImageNet mean and variance.

### 3.2 Calibration Data
- **Composition:** Disjoint $50\%$ split of available validation specimens (both normal and labeled anomalies).
- **Function:** Used strictly to determine decision threshold $\tau_c$ at the $99.5^{\text{th}}$ percentile of nominal validation samples, and to fit Platt scaling parameters $(\beta_0, \beta_1)$.
- **Quarantine Guarantee:** Evaluation holdout sets (and the quarantined VisA, BTAD, and Chukyo benchmarks) are strictly excluded from calibration.

---

## 4. Quantitative Performance Summary

### 4.1 Development Pool Metrics (MVTec AD Prototyping Subset)
Performance evaluated on the 330 development specimens across MVTec development categories:

| Category | Image AUROC (PatchCore) | Optimal Threshold ($\tau_c$) | F1-Score | Mean Steady-State Latency (CPU) |
|:---|:---:|:---:|:---:|:---:|
| **Bottle** | $99.8\%$ | $1.4138$ | $0.985$ | $42.3\text{ ms}$ |
| **Capsule** | $98.6\%$ | $1.3850$ | $0.962$ | $45.1\text{ ms}$ |
| **Metal Nut** | $99.1\%$ | $1.4420$ | $0.974$ | $39.8\text{ ms}$ |

*(Note: CPU latency reflects batch-1 forward inference pass including WideResNet patch extraction and FAISS FlatL2 vector search; excludes one-time ~1.2s cold-start index loading.)*

### 4.2 Hybrid Decision Rule Performance (Audited across 1,725 Images)
Evaluated using `scripts/evaluate_hybrid_decision_rule.py` across the full multi-category benchmark:

| Decision Rule | Full Dataset Recall ($N=1,725$) | Full Dataset FPR | Benchmark Partition Recall ($N=856$) | Benchmark Partition FPR |
|:---|:---:|:---:|:---:|:---:|
| **1. EfficientNet Alone** | $97.06\%$ ($1221 / 1258$) | $0.00\%$ ($0 / 467$) | $99.04\%$ ($619 / 625$) | $0.00\%$ ($0 / 231$) |
| **2. PatchCore Alone** | $74.09\%$ ($932 / 1258$) | $3.21\%$ ($15 / 467$) | $71.84\%$ ($449 / 625$) | $4.76\%$ ($11 / 231$) |
| **3. Hybrid OR (Safety Mode)** | $\mathbf{98.57\%}$ ($1240 / 1258$) | $3.21\%$ ($15 / 467$) | $\mathbf{99.52\%}$ ($622 / 625$) | $4.76\%$ ($11 / 231$) |
| **4. Hybrid AND (Diagnostic)** | $72.58\%$ ($913 / 1258$) | $0.00\%$ ($0 / 467$) | $71.36\%$ ($446 / 625$) | $0.00\%$ ($0 / 231$) |

---

## 5. Limitations, Licensing and Operational Risks

1. **Non-Commercial Model License Notice:** PatchCore memory banks (`models/patchcore/`) and PaDiM feature models (`models/padim/`) were computed from MVTec AD imagery. In accordance with MVTec Software GmbH license terms, these pre-computed artifacts are restricted strictly to non-commercial research and academic demonstration. Commercial deployment requires retraining on proprietary in-house imagery.
2. **Camera Angle & Fixture Sensitivity:** The nearest-neighbor memory bank assumes consistent specimen positioning. Variations in camera angle $>10^\circ$ or significant rotation will register as false anomalies.
3. **Surface Reflectivity:** Highly specular metallic surfaces (e.g., polished aluminum, chrome) can produce specular glare artifacts that inflate anomaly scores.
4. **Platt Scaling Variance on Small Samples:** Calibrated probabilities on categories with fewer than 20 validation defects exhibit wider confidence intervals ($95\%\text{ CI} \approx \pm 0.12$).
5. **Class Imbalance in Manufacturing:** Because true defects represent $<1\%$ of real factory volume, false alarm rates must be closely monitored to prevent operator cognitive fatigue.

---

## 6. Human-in-the-Loop Safeguards

DefectVision AI is intentionally designed as an **augmented intelligence tool** rather than an autonomous decision-maker:
- All borderline predictions are flagged for manual operator review.
- The UI displays explicit confidence margins and nearest training exemplars to allow human validation before physical part rejection.
- Operator overrides are logged to ensure traceability and audit compliance.
