#!/bin/bash
# ================================================================
#  PDF Violation Agent — AMD ROCm / vLLM Setup Script
#  Tested on: Ubuntu 22.04 / 24.04 with AMD Radeon (ROCm)
# ================================================================
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

banner() {
echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   PDF Violation Agent  ·  vLLM + Qwen3  ·  AMD ROCm    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
}

step() { echo -e "\n${BLUE}${BOLD}[$1]${NC} $2"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠  $1${NC}"; }
die()  { echo -e "${RED}✗ $1${NC}"; exit 1; }

banner

# ── Detect GPU ────────────────────────────────────────────────────
step "1/7" "Detecting hardware..."
HAS_AMD=false; HAS_NVIDIA=false

if command -v rocminfo &>/dev/null && rocminfo 2>/dev/null | grep -q "gfx"; then
    HAS_AMD=true
    ok "AMD GPU with ROCm detected"
elif ls /dev/kfd &>/dev/null 2>&1; then
    HAS_AMD=true
    warn "/dev/kfd found — assuming AMD GPU (rocminfo not installed)"
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_NVIDIA=true
    ok "NVIDIA GPU detected"
else
    warn "No GPU detected — will run on CPU (slow but works)"
fi

# ── Python ────────────────────────────────────────────────────────
step "2/7" "Checking Python 3.10+..."
if ! command -v python3 &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y python3 python3-pip python3-venv
fi
PY=$(python3 --version 2>&1)
ok "$PY"

# ── Virtual environment ───────────────────────────────────────────
step "3/7" "Creating virtual environment..."
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
ok "venv active"

# ── Install vLLM ─────────────────────────────────────────────────
step "4/7" "Installing vLLM..."
if $HAS_AMD; then
    echo -e "${CYAN}Installing vLLM for AMD ROCm (pip install vllm[rocm])...${NC}"
    pip install vllm[rocm] -q \
        --extra-index-url https://download.pytorch.org/whl/rocm6.1
    ok "vLLM (ROCm) installed"
elif $HAS_NVIDIA; then
    echo -e "${CYAN}Installing vLLM for NVIDIA CUDA...${NC}"
    pip install vllm -q
    ok "vLLM (CUDA) installed"
else
    echo -e "${CYAN}Installing vLLM (CPU fallback)...${NC}"
    pip install vllm -q
    ok "vLLM (CPU) installed"
fi

# ── Install app dependencies ──────────────────────────────────────
step "5/7" "Installing app dependencies..."
pip install -r requirements.txt -q
ok "App dependencies installed"

# ── Download model ────────────────────────────────────────────────
step "6/7" "Downloading Qwen3 model from HuggingFace..."
echo ""
echo -e "${CYAN}Choose Qwen3 model size:${NC}"
echo "  1) Qwen/Qwen3-1.7B  (~1 GB)  — fastest, good for demo"
echo "  2) Qwen/Qwen3-4B    (~2.5 GB) — balanced"
echo "  3) Qwen/Qwen3-8B    (~5 GB)   — recommended quality  [default]"
echo "  4) Qwen/Qwen3-14B   (~9 GB)   — best quality"
echo ""
read -p "Enter choice [1-4, default=3]: " CHOICE
case "$CHOICE" in
    1) MODEL="Qwen/Qwen3-1.7B" ;;
    2) MODEL="Qwen/Qwen3-4B" ;;
    4) MODEL="Qwen/Qwen3-14B" ;;
    *) MODEL="Qwen/Qwen3-8B" ;;
esac

# Save model choice
sed -i "s|MODEL_NAME.*=.*os.getenv.*|MODEL_NAME = os.getenv(\"VLLM_MODEL\", \"$MODEL\")|" app.py
echo "VLLM_MODEL=$MODEL" > .env
ok "Model set to: $MODEL"
echo -e "${YELLOW}Model will be downloaded automatically on first vLLM start (~${CHOICE:-3} GB).${NC}"
echo -e "${YELLOW}HuggingFace caches to ~/.cache/huggingface/hub${NC}"

# ── Create dirs ───────────────────────────────────────────────────
step "7/7" "Creating project directories..."
mkdir -p uploads reports vector_db templates
ok "Directories ready"

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                ✅  SETUP COMPLETE!                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Open browser:  http://localhost:8080                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "${BOLD}You need TWO terminals:${NC}"
echo ""
echo -e "${CYAN}Terminal 1 — Start vLLM server:${NC}"
echo "  source venv/bin/activate"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model $MODEL \\"
echo "    --served-model-name Qwen/Qwen3-8B \\"
echo "    --port 8000 \\"
echo "    --max-model-len 8192 \\"
if $HAS_AMD; then
echo "    --device cuda   # ROCm uses the 'cuda' device name in vLLM"
fi
echo ""
echo -e "${CYAN}Terminal 2 — Start web app:${NC}"
echo "  source venv/bin/activate"
echo "  uvicorn app:app --host 0.0.0.0 --port 8080 --reload"
echo ""
echo -e "${YELLOW}Or use the helper scripts below:${NC}"
echo "  ./start_vllm.sh   — starts vLLM server"
echo "  ./start_app.sh    — starts web app"
echo ""

# Write helper scripts
cat > start_vllm.sh << VEOF
#!/bin/bash
source venv/bin/activate
MODEL=\${VLLM_MODEL:-$MODEL}
echo "Starting vLLM with model: \$MODEL"
echo "First run downloads the model (~$(python3 -c "sizes={'1.7B':'1','4B':'2.5','8B':'5','14B':'9'}; print(next((v for k,v in sizes.items() if k in '$MODEL'), '5'))") GB) — please wait..."
python -m vllm.entrypoints.openai.api_server \\
    --model "\$MODEL" \\
    --served-model-name "\$MODEL" \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --max-model-len 8192 \\
    --dtype auto \\
    --trust-remote-code
VEOF
chmod +x start_vllm.sh

cat > start_app.sh << AEOF
#!/bin/bash
source venv/bin/activate
echo "Starting PDF Violation Agent web app..."
echo "Open: http://localhost:8080"
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
AEOF
chmod +x start_app.sh

ok "Helper scripts created: start_vllm.sh and start_app.sh"
