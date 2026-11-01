# DefectVision AI — Security Policy and Threat Model

**Document Version:** 1.0.0  
**Scope:** Security Posture, Threat Modeling, Input Validation, and Hardening Guidelines  

---

## 1. Security Overview and Prototype Posture

DefectVision AI is an open-source research prototype designed for local workstation execution and internal factory research networks. It does not include enterprise multi-tenant access control or encrypted user databases out-of-the-box.

---

## 2. Threat Modeling and Attack Surface

In a typical industrial edge or laboratory deployment, the system exposes three primary attack surfaces:

1. **REST API Interface (FastAPI on Port 8000):** Exposes file upload endpoints (`/predict`, `/batch`) vulnerable to malicious payloads if exposed directly to untrusted public networks.
2. **Reverse Proxy (Vite Dev Server on Port 5173):** Configured for local pair programming and development; proxy headers should not be exposed directly to the public internet.
3. **Local File System I/O:** Reading and writing temporary images, batch inspection reports, and FAISS memory bank arrays.

---

## 3. Codebase Security Audit Findings

A rigorous automated and manual security audit of the repository confirmed:
- **Zero Committed Secrets:** No API keys, database passwords, private SSH keys, or cloud credentials are present in Git history or tracked files.
- **Environment Isolation:** All `.env`, `.env.local`, and `.env.*.local` files are strictly excluded by `.gitignore`.
- **Model Checkpoint Integrity:** Model weights and memory banks are loaded from validated local paths with registered cryptographic SHA-256 hashes.
- **Input Validation Safety:** Tested via `scripts/test_invalid_inputs_safety.py`:
  - Unknown categories return HTTP 400 with a safe error payload.
  - Non-image files (e.g., text, executables) and zero-byte files fail gracefully with handled exceptions.
  - Invalid batch IDs return HTTP 404 without leaking stack traces or internal server directories.

---

## 4. Production Hardening Checklist

Before deploying DefectVision AI in an enterprise manufacturing network or public cloud, administrators must implement the following safeguards:

### 4.1 CORS Policy Restriction
In `DefectVisionAI/app.py`, replace the open development wildcard:
```python
# CURRENT DEVELOPMENT SETTING:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
with explicit trusted origins:
```python
# PRODUCTION HARDENED SETTING:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-factory-portal.internal"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### 4.2 Authentication and Access Control
- Implement API key or OAuth2 JWT bearer token verification for `/predict` and `/batch` endpoints.
- Restrict `/severity/logs` and audit history access to authorized quality engineering roles.

### 4.3 Payload Sanitization and Rate Limiting
- Enforce a strict maximum upload size (e.g., $10\text{ MB}$ per image) in the ASGI middleware.
- Validate image headers using `PIL.Image.verify()` before passing tensors to PyTorch to defend against image decompression bombs (ZIP/pixel bombs).
- Deploy an upstream reverse proxy (e.g., NGINX, Caddy, or Traefik) configured with rate limiting (e.g., maximum 30 requests/second per IP).

### 4.4 Containerization and Principle of Least Privilege
- Run the FastAPI service inside a rootless Docker container (`USER nonroot`).
- Mount model directories (`DefectVisionAI/models`) as read-only volumes (`:ro`) to prevent runtime weight tampering.

---

## 5. Vulnerability Reporting

If you identify a security vulnerability, please do not file a public GitHub issue. Instead, report it privately to the maintainer:
- **Contact:** `suryacharan945@gmail.com`
- **Response Target:** 48 hours for triage and acknowledgement.
