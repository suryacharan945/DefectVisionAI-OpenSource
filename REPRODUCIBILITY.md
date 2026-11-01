# DefectVision AI — Reproducibility Guide and Artifact Hashes

**Document Version:** 1.0.0  
**Scope:** Step-by-Step Environment Setup, Execution Commands, and Cryptographic Checksums  

---

## 1. Verified Target Environment

DefectVision AI has been verified on the following reference platform:
- **Operating System:** Windows 11 Pro 64-bit (Build 22631)
- **CPU:** Intel Core i7 / i9 multi-core x86-64
- **Python Environment:** Python 3.12.9 (64-bit) in isolated virtual environment (`.venv`)
- **Node Environment:** Node.js v20.18.0 / v22.x with `npm` 10.x
- **Deep Learning Framework:** PyTorch 2.15.0+cpu / TorchVision 0.29.2+cpu
- **Indexing Engine:** FAISS-CPU 1.15.1
- **API Framework:** FastAPI 0.162.0 with Uvicorn 0.54.0

---

## 2. Clean Installation and Setup

Execute from the root repository directory:

### Step 1: Create and Activate Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Step 2: Install Python Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies
```powershell
npm install
```

---

## 3. Running Automated Test Suites

DefectVision AI includes five automated verification scripts to ensure full-stack integrity:

```powershell
# 1. Run Core Unit Tests (22 unit tests across vision reporting and dispersion)
.\.venv\Scripts\python.exe -m unittest discover tests

# 2. Run Backend End-to-End Test (Tests GET /, POST /predict, POST /batch, /severity/logs, /rag/ask)
.\.venv\Scripts\python.exe scripts/e2e_backend_test.py

# 3. Verify Model Artifact Checksums and Memory Bank Inference
.\.venv\Scripts\python.exe scripts/test_backend_inference.py

# 4. Verify Frontend Vite Reverse Proxy Communication (:5173 -> :8000)
.\.venv\Scripts\python.exe scripts/test_vite_proxy.py

# 5. Verify Input Validation and Error Handling Safety
.\.venv\Scripts\python.exe scripts/test_invalid_inputs_safety.py
```

Expected Output: All 22 unit tests report `OK`; all four integration scripts report complete operational success.

---

## 4. Reproducing Research Experiments

To reproduce the findings reported in Phases 7 through 9:

### Reproduce Phase 8C Mathematical Equivalence & Trace
```powershell
.\.venv\Scripts\python.exe scripts/analyze_phase8c_equivalence.py
```
- Reconstructs the 870 decision instances in `BL6_MACBA_DECISION_TRACE.csv`.
- Confirms Spearman rank correlation $\rho_s = 1.000000$ and $100.0\%$ selection identity between MACBA and properly parameterized BL6.

### Reproduce Phase 7C Density-Normalized CPND Negative Result
```powershell
.\.venv\Scripts\python.exe tests/test_density_norm_cpnd.py
```
- Validates the failure of density-normalized dispersion over raw boundary distance.

---

## 5. Cryptographic Checksums of Primary Artifacts

All core model weights and experiment manifests are registered with immutable SHA-256 hashes:

| Artifact File | Size (Bytes) | SHA-256 Checksum |
|:---|:---:|:---|
| `DefectVisionAI/models/efficientnet_baseline_v3/best_model.pt` | $16,339,261$ | `f470b76d7c00df5be88236d552b3406256a60ee2c27fca361b981fce88902ba8` |
| `DefectVisionAI/models/patchcore/bottle/memory_bank.npy` | $22,639,232$ | `71a68f4e54109c74fec922c02252218d5abfab31256c2bce170e05014db52200` |
| `DefectVisionAI/research/cpnd/dev_extracted_scores_phase7c.csv` | $124,198$ | `678f8cb080ea4cbbf031e67fa80540d585ea1587d60f4da700e1235123d493a7` |
| `research/phase8c_macba_equivalence/BL6_MACBA_DECISION_TRACE.csv` | $158,730$ | `41d66b936a2ce75983d5d468c432a3286099b72bba38f54ed8cfbd8f36f7261e` |
| `research/phase8c_macba_equivalence/ARTIFACT_MANIFEST.json` | $1,598` | `67412702755eeb5e99be5c3dfbcf7df83c92b217be5546b8fc27263cbe54203a` |
| `research/phase9a_publication_feasibility/ARTIFACT_MANIFEST.json` | $1,757$ | `13a5e840a12cfec26e10f36f6d0f66b6c0c2a71bf64ae1fcbfa18a428fe24673` |

---

## 6. Full-Stack One-Click Execution

To launch both backend and frontend simultaneously:
```powershell
.\start_defectvision.bat
```
The script validates the environment, spawns the FastAPI Uvicorn process on port 8000, launches the Vite dev server on port 5173, and opens your browser.
