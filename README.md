# 🔍 PDF Violation Agent — vLLM + Qwen3 Edition
**AI-powered PDF compliance checker | AMD ROCm / NVIDIA CUDA / CPU**

---

## What It Does
1. Upload **Reference PDFs** → chunked & stored in **ChromaDB** vector database
2. Upload **Document to Check** → text extracted with pdfplumber
3. **Qwen3 via vLLM** compares them → finds violations, discrepancies, missing clauses
4. Download a professional **PDF violation report**

---

## Architecture

```
Reference PDF ──► pdfplumber ──► chunks ──► ChromaDB (vector store)
                                                    │
Upload PDF ──► pdfplumber ──► text ──► semantic search ──► top chunks
                                                    │
                                    vLLM /v1/chat/completions (Qwen3)
                                                    │
                                    ReportLab ──► PDF violation report
```

---

## Quick Start

### Linux / Mac — AMD or NVIDIA

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Then in **two terminals**:

```bash
# Terminal 1 — vLLM server
./start_vllm.sh

# Terminal 2 — Web app
./start_app.sh
```

Open **http://localhost:8080**

---

### Windows (AMD CPU / NVIDIA)

```batch
setup_and_run.bat
```

Then run **start_vllm.bat** and **start_app.bat** (each in its own window).

> **AMD GPU on Windows** requires WSL2 with ROCm — see AMD GPU section below.

---

## Manual Setup (Step by Step)

### Step 1 — Install ROCm (AMD GPU only)

```bash
# Ubuntu 22.04 / 24.04
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo dpkg -i amdgpu-install_*.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -a -G render,video $LOGNAME
# Reboot required
sudo reboot
```

Verify:
```bash
rocminfo | grep "gfx"
rocm-smi
```

### Step 2 — Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate.bat   # Windows
pip install --upgrade pip
```

### Step 3 — Install vLLM

```bash
# AMD ROCm (recommended for your AMD notebook)
pip install vllm[rocm] \
    --extra-index-url https://download.pytorch.org/whl/rocm6.1

# NVIDIA CUDA
pip install vllm

# CPU only (slow but works)
pip install vllm
```

### Step 4 — Install App Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code
```

**First run downloads the model from HuggingFace automatically.**
Wait for: `INFO:     Application startup complete.`

Verify it's working:
```bash
curl http://localhost:8000/v1/models
```

### Step 6 — Start Web App

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open **http://localhost:8080**

---

## Model Size Guide

| Model | Download Size | VRAM Needed | Speed | Quality |
|-------|:---:|:---:|:---:|:---:|
| Qwen/Qwen3-1.7B | ~1 GB | 4 GB | ⚡⚡⚡ | Good |
| Qwen/Qwen3-4B | ~2.5 GB | 6 GB | ⚡⚡ | Better |
| **Qwen/Qwen3-8B** | ~5 GB | 10 GB | ⚡ | **Recommended** |
| Qwen/Qwen3-14B | ~9 GB | 18 GB | 🐢 | Best |

Change model in `app.py`:
```python
MODEL_NAME = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B")
```
Or set env var: `export VLLM_MODEL=Qwen/Qwen3-1.7B`

---

## AMD GPU Troubleshooting

```bash
# Check ROCm sees your GPU
rocminfo | grep "gfx"

# Check user groups
groups $USER   # should include 'render' and 'video'

# vLLM AMD: device is called 'cuda' in vLLM (ROCm compatibility layer)
python -c "import torch; print(torch.cuda.is_available())"  # should print True

# If vLLM can't find GPU, set:
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --port 8000
```

---

## General Troubleshooting

| Problem | Fix |
|---------|-----|
| "Cannot connect to vLLM" | Run `./start_vllm.sh` in a separate terminal |
| Model not found (404) | Check `--served-model-name` matches `MODEL_NAME` in app.py |
| Out of VRAM | Use smaller model: `Qwen/Qwen3-1.7B` |
| PDF text empty | PDF may be scanned image; text extraction won't work on image PDFs |
| Port 8000 in use | `--port 8001` for vLLM, set `VLLM_BASE_URL=http://localhost:8001` |
| Slow first response | Normal — model loads into VRAM on first request |

---

## Environment Variables

```bash
export VLLM_BASE_URL=http://localhost:8000   # vLLM server address
export VLLM_MODEL=Qwen/Qwen3-8B              # Model to use
```

---

## Project Structure

```
pdf-violation-agent/
├── app.py               ← FastAPI + ChromaDB + vLLM agent
├── requirements.txt     ← Python dependencies (app only)
├── setup_and_run.sh     ← Linux/Mac one-click setup
├── setup_and_run.bat    ← Windows one-click setup
├── start_vllm.sh        ← Start vLLM server (created by setup)
├── start_app.sh         ← Start web app (created by setup)
├── README.md
├── templates/
│   └── index.html       ← Dark web UI with drag & drop
├── uploads/             ← Temporary PDF storage
├── reports/             ← Generated PDF reports
└── vector_db/           ← ChromaDB persistent storage
```
