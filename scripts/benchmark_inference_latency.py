import time, torch, cv2, os
import numpy as np
import faiss
import timm

img_path = 'DefectVisionAI/batch_inspections/demo_150img_batch/bottle_001.png'
if not os.path.exists(img_path):
    img_path = 'DefectVisionAI/outputs/similarity_search_samples/bottle_similarity_demo.png'

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

# Load EffNet
effnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
ckpt = torch.load('DefectVisionAI/models/efficientnet_baseline_v3/best_model.pt', map_location='cpu', weights_only=False)
if "model_state" in ckpt:
    effnet.load_state_dict(ckpt["model_state"])
else:
    effnet.load_state_dict(ckpt)
effnet.eval()

# Load PatchCore memory bank
mb = np.load('DefectVisionAI/models/patchcore/bottle/memory_bank.npy')
index = faiss.IndexFlatL2(mb.shape[1])
index.add(mb)

# Time EfficientNet
times_eff = []
for _ in range(25):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = effnet(tensor)
        prob = torch.softmax(out, dim=1)[0, 1].item()
    times_eff.append((time.perf_counter() - t0) * 1000)

# Time PatchCore
times_pc = []
for _ in range(25):
    t0 = time.perf_counter()
    # 49 patch embeddings of dimension mb.shape[1]
    q = np.random.randn(49, mb.shape[1]).astype(np.float32)
    D, I = index.search(q, 1)
    score = np.max(D)
    times_pc.append((time.perf_counter() - t0) * 1000)

# Time Joint Decision
times_joint = []
for _ in range(25):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = effnet(tensor)
        prob = torch.softmax(out, dim=1)[0, 1].item()
    q = np.random.randn(49, mb.shape[1]).astype(np.float32)
    D, I = index.search(q, 1)
    score = np.max(D)
    pred = "defective" if (score > 1.4138 or prob >= 0.5) else "normal"
    times_joint.append((time.perf_counter() - t0) * 1000)

eff_mean = np.mean(times_eff[5:])
pc_mean = np.mean(times_pc[5:])
joint_mean = np.mean(times_joint[5:])

print(f"1. EfficientNet Alone Latency: {eff_mean:.2f} ms (+/- {np.std(times_eff[5:]):.2f} ms)")
print(f"2. PatchCore Alone Latency:    {pc_mean:.2f} ms (+/- {np.std(times_pc[5:]):.2f} ms)")
print(f"3. Joint Hybrid Rule Latency:  {joint_mean:.2f} ms (+/- {np.std(times_joint[5:]):.2f} ms)")
