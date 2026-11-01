# DefectVision AI — Interactive Demonstration Guide

**Document Version:** 1.0.0  
**Target:** Technical Demonstrators, Evaluators, and Quality Engineers  
**Platform:** Local Workstation (Windows 10/11)  

---

## 1. Quick-Start Launch Procedure

### Step 1: Launch the Full-Stack Services
From the repository root, double-click or execute in PowerShell:
```powershell
.\start_defectvision.bat
```
This spawns two persistent terminal windows:
1. `DefectVision AI - FastAPI Backend` on `http://127.0.0.1:8000`
2. `DefectVision AI - Vite Frontend` on `http://127.0.0.1:5173`

### Step 2: Open the Inspection Console
Open Google Chrome or Microsoft Edge and navigate to:
```
http://127.0.0.1:5173
```
You will be greeted by the **DefectVision AI Quality Inspection Console**. The status indicator in the top right will display a green `CONNECTED` pill indicating real-time communication with the backend inference engine.

---

## 2. Interactive Feature Walkthrough

### Scenario A: Single-Specimen Live Inspection

1. **Select a Product Line:**
   - In the category selector dropdown, choose **`bottle`**.
2. **Load a Preset Sample:**
   - Click the preset sample pill: **`Contamination (Defective)`** or **`Broken Large (Critical)`**.
   - Alternatively, drag and drop any image from `DefectVisionAI/outputs/similarity_search_samples/` into the upload dropzone.
3. **Trigger Real-Time Inference:**
   - Click the blue **`Run Optical Inspection`** button.
   - Within $<50\text{ ms}$, the inference results populate:
     - **Decision Badge:** Bright red `DEFECTIVE` pill.
     - **Severity Level:** `Critical (72.5 / 100)`.
     - **Anomaly Score:** `4.5088` (exceeding the calibrated threshold of `1.4138`).
     - **Defect Area Coverage:** `73.98%` of specimen surface.

---

### Scenario B: Inspecting Dual-Stream Explainability Heatmaps

1. **PatchCore Anomaly Heatmap:**
   - Locate the center inspection viewport.
   - Use the heatmap opacity slider to adjust the Jet colormap overlay.
   - Observe how the high-temperature red/yellow contour lines precisely encircle the structural fracture or contamination artifact while leaving defect-free nominal glass uncolored.
2. **Nearest-Neighbor Validation:**
   - Scroll down to the **`Top-3 Nearest Training Exemplars`** panel.
   - The system displays the 3 closest matches retrieved from the FAISS FlatL2 memory bank:
     - Match 1: Historical contamination test sample ($L_2\text{ distance: } 30.03$).
     - Match 2 & 3: Closest nominal training bottles ($L_2\text{ distance: } 30.98$).
   - This provides the operator with immediate context on why the algorithm flagged the region.

---

### Scenario C: Asynchronous Batch Lot Inspection

1. Navigate to the **`Batch Inspection`** tab in the top navigation bar.
2. Select **`bottle`** as the target category.
3. Click **`Upload Lot Samples`** and select 2–5 images from the development folder.
4. Click **`Submit Batch Lot`**:
   - The UI displays a live progress bar updating in real-time as background threads process the lot.
   - Upon completion, the **`Batch Quality Summary Report`** renders:
     - Total Inspected Count.
     - Defect Rate Percentage ($100.0\%$).
     - Breakdown by severity grade.
     - Exportable CSV/JSON audit report button.

---

### Scenario D: Querying the RAG Defect Engineering Copilot

1. Click on the **`RAG Copilot`** panel in the bottom right corner.
2. Select category **`bottle`**.
3. In the query prompt, type:
   > *"What causes broken small defects and how do we prevent them in production?"*
4. Click **`Ask Copilot`**:
   - The grounded retrieval engine searches the embedded manufacturing defect manual and returns:
     - **Root Cause:** Localized stress concentration from pre-existing micro-cracks or mold thermal shock.
     - **Operational Impact:** Cosmetic flaw that propagates to catastrophic failure under pressurized bottling.
     - **Recommended Remediation:** Reject affected units; inspect incoming glass batch; reduce stress concentrators in mold design.

---

### Scenario E: Selective Review Triage Console

1. Navigate to the **`Selective Inspection`** dashboard.
2. Observe the **Review Budget Slider ($K$)**:
   - Slide $K$ from $5$ to $25$ items ($14\%$ inspection capacity).
3. The table dynamically ranks specimens by **Expected Loss Reduction ($\Delta L$)**:
   - Notice that critical false negative candidates near the threshold are assigned the highest priority.
   - The dashboard updates the **Expected Cost Reduction KPI** ($+46.27\%$ cost reduction at $\lambda=20$) and shows the exact defects intercepted.

---

## 3. Troubleshooting & Common Questions

| Issue | Root Cause | Solution |
|:---|:---|:---|
| Frontend shows `DISCONNECTED` | Backend on port 8000 is not running. | Verify the FastAPI terminal window is active. Open `http://127.0.0.1:8000/` in browser to verify health status. |
| Port 8000 already in use | Another application or previous uvicorn instance is running. | Open PowerShell and run: `Get-Process python \| Stop-Process -Force`, then re-launch `start_defectvision.bat`. |
| Port 5173 already in use | Vite spawned on port 5174 or 5175. | Check the Vite terminal output for the active URL, or terminate existing Node processes: `Get-Process node \| Stop-Process -Force`. |
| Inference returns `500 Internal Error` | Image format is corrupted or missing RGB channels. | Ensure input image is a valid PNG or JPEG with 3 color channels. |
