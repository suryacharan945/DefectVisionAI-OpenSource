import urllib.request
import urllib.parse
import json
import os
import io
import mimetypes

BASE_URL = "http://127.0.0.1:8000"

print("=================================================================")
print("=== 1. Health Endpoint Test (GET /) ===")
print("=================================================================")
res = urllib.request.urlopen(f"{BASE_URL}/").read()
data = json.loads(res)
print("Response:", json.dumps(data, indent=2))
assert data["status"] == "ready", "Server not ready!"
assert len(data["categories"]) == 15, "Not all 15 categories loaded!"

print("\n=================================================================")
print("=== 2. Single-Image Live Inspection Test (POST /predict) ===")
print("=================================================================")
# Let's inspect bottle
img_path = "DefectVisionAI/outputs/similarity_search_samples/bottle_similarity_demo.png"
with open(img_path, "rb") as f:
    img_bytes = f.read()

# Build multipart form data
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="bottle_demo.png"\r\n'
    f"Content-Type: image/png\r\n\r\n"
).encode('latin-1') + img_bytes + (
    f"\r\n--{boundary}\r\n"
    f'Content-Disposition: form-data; name="category"\r\n\r\n'
    f"bottle\r\n"
    f"--{boundary}--\r\n"
).encode('latin-1')

req = urllib.request.Request(
    f"{BASE_URL}/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
res = urllib.request.urlopen(req).read()
pred_data = json.loads(res)
print("Single-Image Prediction Output:")
for k, v in pred_data.items():
    if k not in ['heatmap', 'raw_heatmap', 'original_image']:
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: [base64 image string, length {len(str(v))}]")

assert "anomaly_score" in pred_data, "Missing anomaly_score!"
assert "optimal_threshold" in pred_data, "Missing optimal_threshold!"
assert "predicted_label" in pred_data, "Missing predicted_label!"

print(f"\nVerification Check:")
print(f"  Score: {pred_data['raw_patchcore_score']} vs Threshold: {pred_data['optimal_threshold']}")
print(f"  Label: {pred_data['predicted_label']} (Severity: {pred_data['severity_level']})")

print("\n=================================================================")
print("=== 3. Batch Inspection Test (POST /batch & GET /batch/{id}/status) ===")
print("=================================================================")
# Multi-file batch
files_to_send = [
    ("DefectVisionAI/outputs/similarity_search_samples/bottle_similarity_demo.png", "bottle_01.png"),
    ("DefectVisionAI/outputs/similarity_search_samples/screw_similarity_demo.png", "screw_01.png")
]

batch_body = b""
for path, name in files_to_send:
    with open(path, "rb") as f:
        fb = f.read()
    batch_body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode('latin-1') + fb + b"\r\n"

batch_body += (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="category"\r\n\r\n'
    f"bottle\r\n"
    f"--{boundary}--\r\n"
).encode('latin-1')

req_b = urllib.request.Request(
    f"{BASE_URL}/batch",
    data=batch_body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
)
res_b = urllib.request.urlopen(req_b).read()
batch_init = json.loads(res_b)
batch_id = batch_init["batch_id"]
print(f"Batch submitted successfully! ID: {batch_id}")

# Poll status
import time
for i in range(10):
    time.sleep(1)
    st = json.loads(urllib.request.urlopen(f"{BASE_URL}/batch/{batch_id}/status").read())
    print(f"  Polling attempt {i+1}: status = {st['status']}, progress = {st.get('progress', 0)}/{st.get('total', 0)}")
    if st["status"] == "completed":
        print("Batch Completed Report Summary:")
        rep = st["report"]
        print(f"  Total Inspected: {rep['total_inspected']}")
        print(f"  Defect Rate:     {rep['defect_rate_pct']}%")
        print(f"  Defective Count: {rep['defective_count']}")
        print(f"  Normal Count:    {rep['normal_count']}")
        print(f"  Results Count:   {len(rep['results'])}")
        break

print("\n=================================================================")
print("=== 4. Inspection History & Severity Logs (GET /severity/logs) ===")
print("=================================================================")
hist_res = urllib.request.urlopen(f"{BASE_URL}/severity/logs?page=1&page_size=5&category=bottle").read()
hist_data = json.loads(hist_res)
print(f"History query (bottle): total_records = {hist_data['total_records']}, total_pages = {hist_data['total_pages']}")
print("First record:", hist_data['records'][0])

print("\n=================================================================")
print("=== 5. RAG Copilot Knowledge Query (POST /rag/ask) ===")
print("=================================================================")
rag_body = json.dumps({"question": "What causes broken large defects in glass bottles?"}).encode('utf-8')
req_rag = urllib.request.Request(f"{BASE_URL}/rag/ask", data=rag_body, headers={"Content-Type": "application/json"})
rag_res = json.loads(urllib.request.urlopen(req_rag).read())
print("RAG Copilot Answer:", rag_res.get("answer"))

print("\n=================================================================")
print("=== ALL BACKEND ENDPOINTS VERIFIED OPERATIONAL! ===")
print("=================================================================")
