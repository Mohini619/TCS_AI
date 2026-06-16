"""
PDF Violation Comparison Agent
AMD ROCm / NVIDIA CUDA / CPU compatible
Uses Qwen3 via vLLM (OpenAI-compatible API)
"""

import os
import uuid
import shutil
import re
from pathlib import Path
from datetime import datetime

import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import requests

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
UPLOAD_DIR   = Path("uploads")
REPORT_DIR   = Path("reports")
VECTOR_DB_DIR = Path("vector_db")

# vLLM server (OpenAI-compatible)
VLLM_BASE_URL   = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_CHAT_URL   = f"{VLLM_BASE_URL}/v1/chat/completions"
VLLM_MODELS_URL = f"{VLLM_BASE_URL}/v1/models"

# Model served by vLLM  — must match --served-model-name in your launch cmd
MODEL_NAME   = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B")

CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
TOP_K_RESULTS   = 6
COLLECTION_NAME = "reference_docs"

for d in [UPLOAD_DIR, REPORT_DIR, VECTOR_DB_DIR]:
    d.mkdir(exist_ok=True)

# ── CHROMADB ───────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
sentence_ef   = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"        # CPU-friendly, ~80 MB
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=sentence_ef,
    metadata={"hnsw:space": "cosine"},
)

# ── FASTAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="PDF Violation Agent — vLLM Edition", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── PDF HELPERS ────────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> dict:
    pages, full_text = {}, ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                pages[i] = text
                full_text += f"\n--- Page {i}/{total} ---\n{text}"
    except Exception as e:
        raise HTTPException(400, f"PDF extraction failed: {e}")
    return {"pages": pages, "full_text": full_text, "total_pages": len(pages)}


def chunk_text(text: str, source: str) -> list[dict]:
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append({
                "id": f"{source}_chunk_{idx}",
                "text": chunk,
                "source": source,
                "chunk_index": idx,
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ── VECTOR DB ──────────────────────────────────────────────────────────────────
def index_pdf_to_vectordb(pdf_path: str, doc_name: str) -> int:
    data   = extract_pdf_text(pdf_path)
    chunks = chunk_text(data["full_text"], doc_name)
    if not chunks:
        return 0
    # Remove stale entries for this document
    try:
        existing = collection.get(where={"source": doc_name})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass
    collection.add(
        ids       =[c["id"]          for c in chunks],
        documents =[c["text"]        for c in chunks],
        metadatas =[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
    )
    return len(chunks)


def search_vectordb(query: str, n: int = TOP_K_RESULTS) -> list[dict]:
    total = collection.count()
    if total == 0:
        return []
    try:
        res = collection.query(query_texts=[query], n_results=min(n, total))
        return [
            {
                "text":     res["documents"][0][i],
                "source":   res["metadatas"][0][i].get("source", "unknown"),
                "distance": (res.get("distances") or [[0]*n])[0][i],
            }
            for i in range(len(res["documents"][0]))
        ]
    except Exception:
        return []


def get_all_indexed_docs() -> list[str]:
    try:
        meta = collection.get()["metadatas"]
        return list({m["source"] for m in meta if m})
    except Exception:
        return []

# ── vLLM / QWEN3 CALL ─────────────────────────────────────────────────────────
def call_vllm(prompt: str, system: str = "") -> str:
    """
    Call vLLM's OpenAI-compatible /v1/chat/completions endpoint.
    Works with any model served by vLLM including Qwen3.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.9,
        "stream": False,
        # Qwen3 supports thinking mode — disable for structured output
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = requests.post(
            VLLM_CHAT_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            503,
            detail=(
                "Cannot connect to vLLM server at "
                f"{VLLM_BASE_URL}. "
                "Start it with: python -m vllm.entrypoints.openai.api_server "
                f"--model {MODEL_NAME} --port 8000"
            ),
        )
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("message", "")
        except Exception:
            pass
        raise HTTPException(502, detail=f"vLLM error: {e} — {detail}")
    except Exception as e:
        raise HTTPException(500, detail=f"Model call failed: {e}")

# ── VIOLATION ANALYSIS ─────────────────────────────────────────────────────────
def analyze_violations(
    uploaded_text: str,
    reference_chunks: list[dict],
    uploaded_name: str,
) -> str:
    ref_context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in reference_chunks
    )

    system = (
        "You are a strict compliance auditor. "
        "Compare the uploaded document against the reference standard documents "
        "and identify ALL violations, discrepancies, missing items, and non-compliances. "
        "Be specific, cite exact text, and categorise by severity. "
        "Respond entirely in clean Markdown."
    )

    prompt = f"""# DOCUMENT AUDIT TASK

## UPLOADED DOCUMENT: "{uploaded_name}"
{uploaded_text[:5000]}

---

## REFERENCE STANDARD (retrieved from vector database):
{ref_context[:4500]}

---

## YOUR ANALYSIS TASK

Compare the uploaded document against the reference standard. Produce the report below.

### 1. EXECUTIVE SUMMARY
- One-paragraph overview of what was compared
- Total violation count
- Severity breakdown — Critical | High | Medium | Low

### 2. VIOLATION TABLE
| # | Severity | Type | Location in Uploaded Doc | Reference Standard Says | Issue |
|---|----------|------|--------------------------|-------------------------|-------|
(one row per violation)

### 3. DETAILED VIOLATION FINDINGS
For every violation (V-001, V-002, …):

**V-XXX — [Short Title]**
- **Severity:** Critical / High / Medium / Low
- **Type:** Data Mismatch / Missing Clause / Contradictory Statement / Policy Violation / Format Error / Other
- **Found in Uploaded Doc:** "[exact quote or NOT FOUND]"
- **Reference Standard Requires:** "[exact quote]"
- **Analysis:** Explain clearly what is wrong
- **Recommendation:** How to fix it

### 4. COMPLIANT SECTIONS
List what IS correct and matches the reference standard.

### 5. RISK ASSESSMENT
Overall compliance score (0–100 %), risk level, and top-3 priority actions.

### 6. CONCLUSION
Final audit verdict in 2–3 sentences."""

    return call_vllm(prompt, system)

# ── PDF REPORT BUILDER ─────────────────────────────────────────────────────────
def generate_pdf_report(
    analysis_text: str,
    uploaded_name: str,
    reference_docs: list[str],
    report_id: str,
) -> str:
    report_path = REPORT_DIR / f"violation_report_{report_id}.pdf"
    doc = SimpleDocTemplate(
        str(report_path), pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    T = lambda name, **kw: ParagraphStyle(name, parent=styles["Normal"], **kw)

    sty_title    = T("CT", fontSize=22, textColor=colors.HexColor("#1a1a2e"),
                     spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold")
    sty_sub      = T("CS", fontSize=10, textColor=colors.HexColor("#555"),
                     spaceAfter=16, alignment=TA_CENTER)
    sty_h1       = T("H1", fontSize=14, textColor=colors.HexColor("#1a1a2e"),
                     spaceBefore=14, spaceAfter=5, fontName="Helvetica-Bold")
    sty_h2       = T("H2", fontSize=12, textColor=colors.HexColor("#16213e"),
                     spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
    sty_body     = T("BD", fontSize=9.5, leading=14,
                     textColor=colors.HexColor("#333"), alignment=TA_JUSTIFY)
    sty_footer   = T("FT", fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    sty_bullet   = T("BL", fontSize=9.5, leading=14,
                     textColor=colors.HexColor("#333"), leftIndent=14)

    SEV = {
        "critical": "#c0392b", "high": "#e67e22",
        "medium": "#f39c12",   "low": "#27ae60",
    }

    story = []

    # Header
    story.append(Spacer(1, .4*cm))
    story.append(Paragraph("PDF VIOLATION AUDIT REPORT", sty_title))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y  %H:%M:%S')}", sty_sub))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, .3*cm))

    # Meta table
    meta = [
        ["Report ID", report_id],
        ["Uploaded Document", uploaded_name],
        ["Reference Documents", "\n".join(reference_docs) or "None"],
        ["AI Model", MODEL_NAME],
        ["Vector DB", f"ChromaDB · {collection.count()} chunks indexed"],
    ]
    mt = Table(meta, colWidths=[4*cm, 12*cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0), (0,-1), colors.white),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), .5, colors.HexColor("#ccc")),
        ("ROWBACKGROUNDS",(1,0), (-1,-1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    story.append(mt)
    story.append(Spacer(1, .5*cm))
    story.append(HRFlowable(width="100%", thickness=.5,
                             color=colors.HexColor("#ccc")))
    story.append(Spacer(1, .3*cm))

    # Parse markdown → ReportLab
    def inline(t: str) -> str:
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", t)
        t = re.sub(r"`(.+?)`",
                   r'<font name="Courier" size="8">\1</font>', t)
        for sev, col in SEV.items():
            t = re.sub(
                rf"\b({sev.capitalize()}|{sev.upper()})\b",
                f'<font color="{col}"><b>\\1</b></font>',
                t, flags=re.IGNORECASE,
            )
        return t

    lines = analysis_text.split("\n")
    i = 0
    while i < len(lines):
        raw  = lines[i]
        line = raw.strip()

        # Markdown table
        if "|" in line and i+1 < len(lines) and "---" in lines[i+1]:
            rows = [line]
            i += 2
            while i < len(lines) and "|" in lines[i]:
                rows.append(lines[i])
                i += 1
            parsed = [
                [c.strip() for c in r.strip("|").split("|")]
                for r in rows
            ]
            if parsed:
                nc  = max(len(r) for r in parsed)
                pad = lambda r: r + [""] * (nc - len(r))
                parsed = [pad(r) for r in parsed]
                cw  = [16*cm / nc] * nc
                tbl = Table(parsed, colWidths=cw, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#1a1a2e")),
                    ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                    ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                    ("FONTSIZE",      (0,0), (-1,-1), 8),
                    ("GRID",          (0,0), (-1,-1), .5, colors.HexColor("#ccc")),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1),
                     [colors.white, colors.HexColor("#f5f5f5")]),
                    ("VALIGN",        (0,0), (-1,-1), "TOP"),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("LEFTPADDING",   (0,0), (-1,-1), 4),
                    ("WORDWRAP",      (0,0), (-1,-1), True),
                ]))
                story.append(tbl)
                story.append(Spacer(1, .3*cm))
            continue

        if not line:
            story.append(Spacer(1, .12*cm))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), sty_h2))
        elif line.startswith("## ") or line.startswith("# "):
            story.append(Paragraph(inline(line.lstrip("# ")), sty_h1))
        elif line.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=.5,
                                     color=colors.HexColor("#ddd")))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph("• " + inline(line[2:]), sty_bullet))
        else:
            try:
                story.append(Paragraph(inline(line), sty_body))
            except Exception:
                story.append(Paragraph(
                    line.encode("ascii", "replace").decode(), sty_body))
        i += 1

    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a1a2e")))
    story.append(Paragraph(
        f"PDF Violation Agent  ·  Report {report_id}  ·  "
        f"{datetime.now().strftime('%Y-%m-%d')}  ·  Model: {MODEL_NAME}",
        sty_footer,
    ))

    doc.build(story)
    return str(report_path)

# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(open("templates/index.html").read())


@app.post("/api/index-reference")
async def index_reference(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")
    path = UPLOAD_DIR / f"ref_{uuid.uuid4().hex}_{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    n = index_pdf_to_vectordb(str(path), file.filename)
    return {
        "success": True,
        "message": f"Indexed '{file.filename}' → {n} chunks",
        "doc_name": file.filename,
        "chunks": n,
        "total_in_db": collection.count(),
    }


@app.get("/api/indexed-docs")
async def indexed_docs():
    docs = get_all_indexed_docs()
    return {"documents": docs, "total_chunks": collection.count(), "count": len(docs)}


@app.delete("/api/clear-db")
async def clear_db():
    global collection
    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return {"success": True, "message": "Vector DB cleared"}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    context_hint: str = Form(default=""),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")
    if collection.count() == 0:
        raise HTTPException(
            400,
            "No reference documents indexed. "
            "Please upload at least one reference PDF first.",
        )

    path = UPLOAD_DIR / f"upload_{uuid.uuid4().hex}_{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pdf_data   = extract_pdf_text(str(path))
    query      = (context_hint + " " + pdf_data["full_text"][:500]).strip()
    ref_chunks = search_vectordb(query)
    ref_docs   = list({c["source"] for c in ref_chunks})

    analysis  = analyze_violations(pdf_data["full_text"], ref_chunks, file.filename)
    report_id = uuid.uuid4().hex[:8].upper()
    rpt_path  = generate_pdf_report(analysis, file.filename, ref_docs, report_id)

    return {
        "success": True,
        "report_id": report_id,
        "report_file": Path(rpt_path).name,
        "uploaded_doc": file.filename,
        "reference_docs": ref_docs,
        "pages_analyzed": pdf_data["total_pages"],
        "chunks_retrieved": len(ref_chunks),
        "analysis_markdown": analysis,
        "download_url": f"/api/download/{Path(rpt_path).name}",
    }


@app.get("/api/download/{filename}")
async def download(filename: str):
    p = REPORT_DIR / filename
    if not p.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(str(p), media_type="application/pdf", filename=filename)


@app.get("/api/health")
async def health():
    """Check vLLM server status."""
    try:
        resp   = requests.get(VLLM_MODELS_URL, timeout=5)
        models = [m["id"] for m in resp.json().get("data", [])]
        ok     = any(MODEL_NAME in m for m in models)
        return {
            "vllm": "online",
            "models_available": models,
            "target_model": MODEL_NAME,
            "model_ready": ok,
            "vector_db_chunks": collection.count(),
            "indexed_docs": get_all_indexed_docs(),
        }
    except Exception as e:
        return {
            "vllm": "offline",
            "error": str(e),
            "vllm_url": VLLM_BASE_URL,
            "fix": (
                f"Start vLLM: python -m vllm.entrypoints.openai.api_server "
                f"--model {MODEL_NAME} --port 8000"
            ),
        }
