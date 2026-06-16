# 🚀 AI-Driven Audit & Compliance Validator - Quick Start Guide

## 📋 What This Does

This is an **AI-powered compliance auditing system** that:
1. **Indexes reference documents** (regulations, policies, standards) into a vector database using RAG (Retrieval-Augmented Generation)
2. **Analyzes uploaded documents** against these references
3. **Identifies violations** with confidence scores and evidence
4. **Generates professional PDF reports** with full explainability

Perfect for financial/insurance document compliance checking!

---

## 🏗️ Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER UPLOADS DOCUMENTS                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ REFERENCE PDFs  │             │ DOCUMENT TO     │
    │ (Standards)     │             │ CHECK           │
    └────────┬────────┘             └────────┬────────┘
             │                               │
             ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ pdfplumber      │             │ pdfplumber      │
    │ Text Extraction │             │ Text Extraction │
    └────────┬────────┘             └────────┬────────┘
             │                               │
             ▼                               │
    ┌─────────────────┐                      │
    │ Chunking        │                      │
    │ (800 chars)     │                      │
    └────────┬────────┘                      │
             │                               │
             ▼                               │
    ┌─────────────────┐                      │
    │ ChromaDB        │◄─────────────────────┤
    │ Vector Store    │   Semantic Search    │
    │ + Embeddings    │   (Top-K chunks)     │
    └────────┬────────┘                      │
             │                               │
             └───────────────┬───────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Qwen3 LLM via vLLM  │
                  │ - Compare doc vs    │
                  │   reference chunks  │
                  │ - Identify violations
                  │ - Assign confidence │
                  │ - Provide evidence  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ JSON Analysis Result│
                  │ - Violations list   │
                  │ - Confidence scores │
                  │ - Evidence quotes   │
                  │ - Risk assessment   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ ReportLab PDF       │
                  │ Professional Report │
                  │ with tables & charts│
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ DOWNLOAD REPORT     │
                  └─────────────────────┘
```

---

## 🔧 Step-by-Step Execution (Beginner Friendly)

### Prerequisites
- Python 3.10+ installed
- AMD GPU with ROCm (for your AMD notebook) OR NVIDIA GPU OR CPU fallback
- 8GB+ RAM recommended
- 10GB+ free disk space

---

### STEP 1: Clone/Navigate to Project Directory

```bash
cd /workspace
```

---

### STEP 2: Run the Setup Script

The setup script will:
- Create a Python virtual environment
- Install all dependencies (vLLM, FastAPI, ChromaDB, etc.)
- Download the Qwen3 AI model
- Create helper scripts

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

**What happens:**
1. Detects your AMD GPU automatically
2. Installs `vllm[rocm]` for AMD acceleration
3. Asks you to choose a model size (select option 3 for Qwen3-8B - recommended)
4. Creates `venv/`, `start_vllm.sh`, and `start_app.sh`

---

### STEP 3: Start TWO Terminal Windows

You need **TWO separate terminals** running simultaneously:

#### Terminal 1: Start the vLLM AI Model Server

```bash
source venv/bin/activate
./start_vllm.sh
```

**What this does:**
- Loads the Qwen3-8B AI model into memory/VRAM
- Starts an OpenAI-compatible API server on port 8000
- First run downloads ~5GB model from HuggingFace (takes 2-5 minutes)
- Wait until you see: `INFO: Application startup complete.`

**Expected output:**
```
Starting vLLM with model: Qwen/Qwen3-8B
First run downloads the model (~5 GB) — please wait...
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Keep this terminal open!

#### Terminal 2: Start the Web Application

```bash
source venv/bin/activate
./start_app.sh
```

**What this does:**
- Starts the FastAPI web server on port 8080
- Connects to the vLLM server for AI processing
- Provides the web UI at http://localhost:8080

**Expected output:**
```
Starting PDF Violation Agent web app...
Open: http://localhost:8080
INFO:     Uvicorn running on http://0.0.0.0:8080
```

---

### STEP 4: Open the Web Interface

1. Open your browser
2. Navigate to: **http://localhost:8080**
3. You'll see a dark-themed web UI with:
   - Left panel: Vector DB stats, upload zones
   - Right panel: Analysis results

---

### STEP 5: Upload Reference Documents (Compliance Standards)

**Reference documents** are your "rules" - e.g.:
- Insurance policy templates
- Financial regulations (SOX, GDPR, HIPAA)
- Company compliance standards
- Industry guidelines

**How to:**
1. In the left panel, find **"Index Reference PDF"** section
2. Click or drag-and-drop a PDF file (e.g., `insurance_policy_standard.pdf`)
3. Click **"⬆️ Index to Vector DB"** button
4. Wait for confirmation: `"Indexed 'filename.pdf' → XX chunks"`

**What happens internally:**
```python
PDF → pdfplumber extracts text → split into 800-char chunks 
→ SentenceTransformer creates embeddings → stored in ChromaDB
```

You can upload multiple reference documents. They accumulate in the vector DB.

---

### STEP 6: Upload Document to Analyze

This is the document you want to check for compliance violations.

**How to:**
1. Find **"Document to Analyze"** section
2. Upload a PDF (e.g., `submitted_claim_form.pdf`)
3. Optionally add a context hint (e.g., "Check for missing signatures")
4. Click **"🔍 Analyze Document"** button

---

### STEP 7: Watch the Analysis Progress

The system will:
1. ✅ Extract text from uploaded document
2. 🔍 Perform semantic search in vector DB for relevant rules
3. 🧠 Send document + retrieved chunks to Qwen3 LLM
4. 📊 Parse JSON response with violations & confidence scores
5. 📄 Generate PDF report

**Progress indicator shows each step.**

---

### STEP 8: Review Results & Download Report

After analysis completes, you'll see:

**On Screen:**
- Executive summary with compliance score
- Violations table with confidence scores
- Detailed findings with evidence

**Download Button:**
- Click **"📥 Download PDF Report"** to get a professional audit report

---

## 📊 Understanding the Output

### JSON Analysis Structure

```json
{
  "executive_summary": {
    "total_violations_found": 5,
    "severity_breakdown": {"critical": 1, "high": 2, "medium": 1, "low": 1},
    "overall_compliance_score": 65.5,
    "risk_level": "HIGH"
  },
  "violations": [
    {
      "violation_id": "V-001",
      "title": "Missing Policy Number",
      "severity": "High",
      "confidence_score": 0.92,
      "location_in_uploaded_doc": {
        "exact_quote": "Policy holder: John Doe"
      },
      "reference_requirement": {
        "source_document": "insurance_standard.pdf",
        "exact_quote": "All claims must include valid policy number"
      },
      "evidence_analysis": "Document lacks required policy identifier...",
      "recommendation": "Add policy number field to form"
    }
  ],
  "compliant_sections": [...],
  "risk_assessment": {...},
  "audit_methodology": {
    "rag_retrieval_method": "Semantic search...",
    "model_used": "Qwen/Qwen3-8B",
    "chunks_analyzed": 6,
    "explainability_note": "Each violation includes direct evidence..."
  }
}
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Confidence Scores** | 0.0-1.0 indicating AI certainty about each violation |
| **Evidence Quotes** | Direct text from both uploaded doc AND reference standard |
| **Severity Levels** | Critical/High/Medium/Low based on regulatory impact |
| **Explainability** | Clear reasoning chain for each finding |
| **RAG Traceability** | Shows which reference chunks were used |

---

## 🔑 Code Flow Explanation

### Main Components

#### 1. `app.py` - Core Logic

```python
# A. PDF Text Extraction
def extract_pdf_text(pdf_path):
    # Uses pdfplumber to read PDF pages
    # Returns: {"pages": {...}, "full_text": "..."}

# B. Chunking for RAG
def chunk_text(text, source):
    # Splits text into 800-char overlapping chunks
    # Each chunk gets metadata: source doc name, chunk index

# C. Vector Database Operations
def index_pdf_to_vectordb(pdf_path, doc_name):
    # Extract → Chunk → Embed → Store in ChromaDB
    
def search_vectordb(query, n=6):
    # Semantic similarity search
    # Returns top-N most relevant chunks

# D. AI Analysis (The Magic!)
def analyze_violations(uploaded_text, reference_chunks, uploaded_name):
    # Builds prompt with:
    #   - Uploaded document text
    #   - Retrieved reference chunks (RAG)
    #   - Instructions for JSON output
    # Calls vLLM API → Parses JSON response
    # Returns structured dict with violations & confidence

# E. PDF Report Generation
def generate_pdf_report(analysis_data, ...):
    # Converts JSON analysis to professional PDF
    # Uses ReportLab for formatting
    # Includes tables, colors, sections
```

#### 2. API Endpoints

```python
POST /api/index-reference
    → Uploads reference PDF to vector DB

GET /api/indexed-docs
    → Lists all indexed reference documents

POST /api/analyze
    → Main analysis endpoint:
      1. Extract text from uploaded PDF
      2. Search vector DB for relevant rules
      3. Call LLM for violation analysis
      4. Generate PDF report
      5. Return JSON + download URL

GET /api/download/{filename}
    → Downloads generated PDF report

GET /api/health
    → Checks if vLLM server is running
```

---

## 🛠️ Troubleshooting

### Problem: "Cannot connect to vLLM server"
**Solution:** Make sure Terminal 1 is still running `./start_vllm.sh`

### Problem: "Model not found (404)"
**Solution:** Check that `--served-model-name` matches `MODEL_NAME` in app.py

### Problem: "Out of VRAM"
**Solution:** Use smaller model: `export VLLM_MODEL=Qwen/Qwen3-1.7B` before starting

### Problem: Slow first response
**Normal!** Model loads into VRAM on first request (30-60 seconds)

### Problem: AMD GPU not detected
```bash
# Verify ROCm installation
rocminfo | grep "gfx"

# Check user groups
groups $USER  # should include 'render' and 'video'

# Set environment variables if needed
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
```

---

## 🎯 Hackathon Demo Tips

1. **Prepare sample documents:**
   - Reference: Sample insurance policy template
   - To check: Claim form with intentional errors

2. **Demo flow:**
   - Show empty vector DB
   - Index reference document (show chunks count)
   - Upload non-compliant document
   - Watch real-time analysis progress
   - Highlight confidence scores in results
   - Download and show PDF report

3. **Key selling points:**
   - ✅ RAG-based retrieval (not just LLM hallucination)
   - ✅ Confidence scores for each finding
   - ✅ Evidence-backed violations (quotes from both docs)
   - ✅ Explainable AI (clear reasoning chain)
   - ✅ Professional auditable reports
   - ✅ Runs locally on AMD hardware (no cloud dependency)

---

## 📁 Project Structure

```
/workspace/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── setup_and_run.sh      # One-click setup script
├── start_vllm.sh         # Created by setup - starts AI server
├── start_app.sh          # Created by setup - starts web app
├── templates/
│   └── index.html        # Web UI (dark theme)
├── uploads/              # Temporary PDF storage
├── reports/              # Generated PDF reports
└── vector_db/            # ChromaDB persistent storage
```

---

## 🚀 Advanced Customization

### Change AI Model
```bash
export VLLM_MODEL=Qwen/Qwen3-4B  # Before starting vLLM
```

### Adjust Chunk Size
Edit `app.py`:
```python
CHUNK_SIZE = 800      # Characters per chunk
CHUNK_OVERLAP = 100   # Overlap for context
TOP_K_RESULTS = 6     # Number of chunks to retrieve
```

### Modify Prompt
Edit the `analyze_violations()` function prompt to change:
- Output format
- Severity criteria
- Types of violations to detect

---

## 📞 Support

For issues:
1. Check both terminals for error messages
2. Verify vLLM is responding: `curl http://localhost:8000/v1/models`
3. Ensure PDF files are text-based (not scanned images)

Good luck with your AMD hackathon! 🎉
