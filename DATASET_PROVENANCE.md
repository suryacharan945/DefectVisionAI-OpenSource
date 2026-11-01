# DefectVision AI — Dataset Provenance and Exposure Manifest

**Document Version:** 1.0.0  
**Status:** COMPLETE & VERIFIED  
**Scope:** Dataset Accounting, Exposure History, and Benchmark Quarantine  

---

## 1. Executive Statement of Provenance

To guarantee complete scientific integrity and prevent data contamination, this document defines the exact boundaries between datasets utilized for exploratory development and datasets held in pristine quarantine for future independent benchmarking.

**Core Principles:**
1. **No Data Leakage:** Development prototypes and calibration models were fitted strictly on authorized development categories.
2. **Quarantine Preserved:** The 7 unexposed VisA categories, the BTAD dataset, and the Chukyo dataset were **never accessed or evaluated** during the research audits.
3. **Disputed Subsets Disclosed:** Four VisA categories (`candle`, `macaroni1`, `macaroni2`, `pcb1`) cited in exploratory notes are formally classified as `UNKNOWN / DISPUTED` and barred from pristine independent evaluation.
4. **Reproducible Provenance:** All datasets derive from standard, publicly available academic benchmarks.

---

## 2. Dataset Exposure Inventory

### 2.1 Exposed Development Pool (Authorized for Prototyping)
These categories were accessed during model development, threshold tuning, and exploratory experiments in Google Colab and local testing:

| Dataset | Category | Split Composition | Image Count | Exposure History |
|:---|:---|:---|:---:|:---|
| **MVTec AD** | `bottle` | Train Normal (209), Test Normal (20), Test Defect (63) | 292 | Exposed during initial training; used for UI presets, latency benchmarking, and smoke testing. |
| **MVTec AD** | `capsule` | Train Normal (219), Test Normal (23), Test Defect (109) | 351 | Exposed during initial PatchCore fitting; used for Phase 7C and Phase 8B calibration sweeps. |
| **MVTec AD** | `metal_nut` | Train Normal (220), Test Normal (22), Test Defect (93) | 335 | Exposed during initial PatchCore fitting; used for Phase 7C and Phase 8B calibration sweeps. |
| **VisA** | `capsules` | Train Normal (542), Test Normal (60), Test Defect (100) | 702 | Exposed during historical Google Drive experiments (`visa_development_v2`, `official_test_v1`). |

### 2.2 Disputed / Exploratory Pool (Excluded from Independent Benchmarks)
These VisA categories were cataloged during extraction and referenced in exploratory notes (e.g., PCB inspection cycle-time modeling). In accordance with the audit mandate, missing proof of non-exposure requires treating them as **UNKNOWN / DISPUTED**:

| Dataset | Category | Total Images | Status | Forensic Finding |
|:---|:---|:---:|:---:|:---|
| **VisA** | `candle` | 1,100 | **UNKNOWN / DISPUTED** | Cataloged in archive; cited in exploratory notes. Excluded from pristine benchmarks. |
| **VisA** | `macaroni1` | 1,100 | **UNKNOWN / DISPUTED** | Cataloged in archive; cited in exploratory notes. Excluded from pristine benchmarks. |
| **VisA** | `macaroni2` | 1,100 | **UNKNOWN / DISPUTED** | Cataloged in archive; cited in exploratory notes. Excluded from pristine benchmarks. |
| **VisA** | `pcb1` | 1,104 | **UNKNOWN / DISPUTED** | Cataloged in archive; cited in Phase 8C cycle-time analysis. Excluded from pristine benchmarks. |

### 2.3 Quarantined Independent Evaluation Pool (Strictly Unaccessed)
These datasets and categories have **never been accessed, read, or evaluated** in any phase of this project:

| Benchmark | Quarantined Categories | Estimated Images | Status | Quarantine Condition |
|:---|:---|:---:|:---:|:---|
| **VisA** (Zou et al., 2022) | 7 categories: `pcb2`, `pcb3`, `pcb4`, `cashew`, `chewinggum`, `fryum`, `pipe_fryum` | $5,715$ | **POTENTIALLY UNTOUCHED** | Zero model fitting, zero feature extraction, zero evaluation rows in repo. |
| **BTAD** (Mishra et al., 2021) | 3 industrial categories: `01`, `02`, `03` (wood, metal, plastic components) | $2,830$ | **PRISTINE** | Completely unaccessed external archive. |
| **Chukyo** Benchmark | Industrial screw and fastener surface defects | $\approx 1,000$ | **PRISTINE** | Completely unaccessed external benchmark. |

*(Note: Earlier drafts listed six invalid VisA categories: `bones`, `pushpins`, `socket`, `tag`, `wallnut`, `waterbottle`. These do not exist in the official 12-category VisA benchmark and have been purged; see `release/final_integrity_gate/DATASET_EXPOSURE_RECONCILIATION.md` for forensic details.)*

---

## 3. Disjoint Partitioning Protocol for Development Experiments

For all reported Phase 7C, 8B, and 8C experiments, the 330 development images were partitioned using a group-aware stratified split:

```
Total Development Pool: 330 specimens (bottle, capsule, metal_nut)
                       |
       +---------------+---------------+
       |                               |
       v                               v
[ Calibration Split (50%) ]    [ Internal Evaluation Split (50%) ]
  N = 156 images                 N = 174 images
  27 Errors (27 FN, 0 FP)        35 Errors (32 FN, 3 FP)
       |                               |
       v                               v
- Fit decision thresholds        - Measure policy cost reduction
- Fit Platt logistic calibrators - Evaluate false negative capture
- Zero holdout labels seen!      - Compute realized loss curves
```

### Protocol Validation:
- Stratification key: `category_defecttype_iserror` ensuring balanced defect representation across splits.
- Random Seed: Fixed at `SEED = 42` for exact reproducibility.
- Cryptographic hash of development score extraction:
  `DefectVisionAI/research/cpnd/dev_extracted_scores_phase7c.csv` (SHA-256 verified).

---

## 4. Licenses and Third-Party Rights

1. **MVTec Anomaly Detection (MVTec AD):**
   - Released by MVTec Software GmbH for non-commercial research and academic evaluation.
   - Reference: Bergmann et al., CVPR 2019 / IJCV 2021.
2. **VisA (Visual Anomaly):**
   - Released by Zou et al., ECCV 2022 under CC BY 4.0 license.
3. **BTAD (BeanTech Anomaly Detection):**
   - Released by Mishra et al., IEEE Access 2021 under academic research permissions.
