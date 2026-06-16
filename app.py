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
) -> dict:
    """
    Analyze uploaded document against reference standards using RAG + LLM.
    Returns structured findings with confidence scores and evidence.
    """
    # Build context with source metadata for better traceability
    ref_context_parts = []
    for c in reference_chunks:
        source_info = c.get('source', 'unknown')
        chunk_idx = c.get('chunk_index', 0)
        ref_context_parts.append(
            f"[Source Document: {source_info}] [Chunk ID: {c['id']}]\n{c['text']}"
        )
    ref_context = "\n\n=== REFERENCE CHUNK SEPARATOR ===\n\n".join(ref_context_parts)

    system = (
        "You are a strict AI compliance auditor specializing in financial and insurance document validation. "
        "Your task is to compare uploaded documents against reference compliance standards and produce an auditable report. "
        "For EACH violation found, you MUST provide: "
        "(1) A confidence score (0.0-1.0) indicating how certain you are about the violation, "
        "(2) Direct evidence quotes from both the uploaded document AND reference standard, "
        "(3) Clear explanation of WHY it's a violation. "
        "Be precise, cite exact text, and justify every finding. "
        "Respond ONLY in valid JSON format matching the schema provided."
    )

    prompt = f"""# FINANCIAL/INSURANCE DOCUMENT COMPLIANCE AUDIT

## TASK OVERVIEW
You are performing an AI-driven audit to validate compliance of a submitted document against official reference standards.
This is for a hackathon project requiring: RAG-based retrieval, rule validation, explainability, and confidence scoring.

## UPLOADED DOCUMENT FOR REVIEW: "{uploaded_name}"
```
{uploaded_text[:6000]}
```

## REFERENCE COMPLIANCE STANDARDS (Retrieved via RAG from Vector Database)
{ref_context[:5000]}

## OUTPUT REQUIREMENTS

Produce a JSON object with the following exact structure:

```json
{{
  "executive_summary": {{
    "document_name": "{uploaded_name}",
    "audit_timestamp": "YYYY-MM-DD HH:MM:SS",
    "total_violations_found": <integer>,
    "severity_breakdown": {{
      "critical": <count>,
      "high": <count>,
      "medium": <count>,
      "low": <count>
    }},
    "overall_compliance_score": <float 0-100>,
    "risk_level": "CRITICAL|HIGH|MEDIUM|LOW"
  }},
  "violations": [
    {{
      "violation_id": "V-001",
      "title": "Short descriptive title",
      "severity": "Critical|High|Medium|Low",
      "violation_type": "Data Mismatch|Missing Clause|Contradictory Statement|Policy Violation|Format Error|Regulatory Non-Compliance|Other",
      "confidence_score": <float 0.0-1.0>,
      "location_in_uploaded_doc": {{
        "page_number": <integer or null>,
        "section": "section name if available",
        "exact_quote": "direct quote from uploaded document"
      }},
      "reference_requirement": {{
        "source_document": "name of reference doc",
        "chunk_id": "chunk identifier",
        "exact_quote": "direct quote from reference standard"
      }},
      "evidence_analysis": "Explain step-by-step reasoning: what the rule requires, what the document says, why this constitutes a violation",
      "recommendation": "Specific actionable steps to remediate this violation",
      "regulatory_impact": "Brief note on potential regulatory/financial impact"
    }}
  ],
  "compliant_sections": [
    {{
      "section_name": "name of compliant section",
      "description": "What this section covers",
      "confidence_score": <float 0.0-1.0>,
      "evidence": "Quote showing compliance"
    }}
  ],
  "risk_assessment": {{
    "overall_risk_score": <float 0-100, higher=worse>,
    "top_3_priority_actions": [
      "First priority action item",
      "Second priority action item", 
      "Third priority action item"
    ],
    "estimated_remediation_effort": "Low|Medium|High",
    "potential_penalties": "Description of potential fines/consequences"
  }},
  "audit_methodology": {{
    "rag_retrieval_method": "Semantic search over vectorized reference documents",
    "model_used": "Qwen3 via vLLM",
    "chunks_analyzed": {len(reference_chunks)},
    "explainability_note": "Each violation includes direct evidence quotes and reasoning chain"
  }},
  "conclusion": "2-3 sentence final audit verdict summarizing overall compliance posture"
}}
```

## CRITICAL INSTRUCTIONS

1. **Confidence Scoring**: Assign confidence based on:
   - 0.9-1.0: Clear violation with explicit contradictory evidence
   - 0.7-0.9: Strong indication with good evidence
   - 0.5-0.7: Moderate evidence, some ambiguity
   - Below 0.5: Weak evidence, flag for human review

2. **Evidence Requirements**: Every violation MUST include:
   - Exact quote from uploaded document (or state "NOT FOUND" if missing required content)
   - Exact quote from reference standard showing the requirement
   - Clear logical connection between the two

3. **Severity Classification**:
   - Critical: Legal/regulatory breach, potential criminal liability
   - High: Significant policy violation, major financial impact
   - Medium: Notable deviation from standards, moderate impact
   - Low: Minor formatting/procedural issues

4. **Output Format**: Return ONLY valid JSON. No markdown fences. No explanatory text outside JSON."""

    response_text = call_vllm(prompt, system)
    
    # Parse JSON response
    import json
    try:
        # Try to extract JSON from response (handle potential markdown wrapping)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response_text)
        return result
    except json.JSONDecodeError as e:
        # Fallback: return structured error with raw response
        return {
            "error": f"Failed to parse LLM response as JSON: {str(e)}",
            "raw_response": response_text,
            "executive_summary": {
                "document_name": uploaded_name,
                "audit_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_violations_found": 0,
                "severity_breakdown": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                "overall_compliance_score": 50.0,
                "risk_level": "UNKNOWN"
            },
            "violations": [],
            "compliant_sections": [],
            "risk_assessment": {
                "overall_risk_score": 50.0,
                "top_3_priority_actions": ["Review parsing error", "Check document quality", "Retry analysis"],
                "estimated_remediation_effort": "Unknown",
                "potential_penalties": "Unknown due to parsing error"
            },
            "audit_methodology": {
                "rag_retrieval_method": "Semantic search over vectorized reference documents",
                "model_used": MODEL_NAME,
                "chunks_analyzed": len(reference_chunks),
                "explainability_note": "JSON parsing failed - see raw_response"
            },
            "conclusion": "Analysis encountered a technical error. Please retry or check document quality."
        }

# ── PDF REPORT BUILDER ─────────────────────────────────────────────────────────
def generate_pdf_report(
    analysis_data: dict,
    uploaded_name: str,
    reference_docs: list[str],
    report_id: str,
) -> str:
    """
    Generate a professional PDF audit report from structured JSON analysis data.
    Includes executive summary, violations with confidence scores, evidence tables, and risk assessment.
    """
    import json
    # Handle both dict and string inputs for backward compatibility
    if isinstance(analysis_data, str):
        try:
            analysis = json.loads(analysis_data)
        except:
            analysis = {"raw_markdown": analysis_data}
    else:
        analysis = analysis_data
    
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
    sty_confidence = T("CF", fontSize=9, textColor=colors.HexColor("#6c63ff"), fontName="Helvetica-Bold")

    SEV = {
        "critical": "#c0392b", "high": "#e67e22",
        "medium": "#f39c12",   "low": "#27ae60",
    }

    story = []

    # Header
    story.append(Spacer(1, .4*cm))
    story.append(Paragraph("🔍 AI-DRIVEN COMPLIANCE AUDIT REPORT", sty_title))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y  %H:%M:%S')}", sty_sub))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, .3*cm))

    # Extract key metrics from analysis
    exec_summary = analysis.get("executive_summary", {})
    total_violations = exec_summary.get("total_violations_found", 0)
    compliance_score = exec_summary.get("overall_compliance_score", 0)
    risk_level = exec_summary.get("risk_level", "UNKNOWN")
    
    # Meta table with key metrics
    severity_breakdown = exec_summary.get("severity_breakdown", {})
    meta = [
        ["Report ID", report_id],
        ["Document Audited", uploaded_name[:40] + "..." if len(uploaded_name) > 40 else uploaded_name],
        ["Compliance Score", f"{compliance_score:.1f}%"],
        ["Risk Level", risk_level],
        ["Total Violations", str(total_violations)],
        ["Severity Breakdown", f"C:{severity_breakdown.get('critical',0)} H:{severity_breakdown.get('high',0)} M:{severity_breakdown.get('medium',0)} L:{severity_breakdown.get('low',0)}"],
        ["Reference Docs", ", ".join(reference_docs[:2]) + ("..." if len(reference_docs) > 2 else "")],
        ["AI Model", MODEL_NAME.split("/")[-1]],
    ]
    mt = Table(meta, colWidths=[4.5*cm, 11.5*cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0), (0,-1), colors.white),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("GRID",          (0,0), (-1,-1), .5, colors.HexColor("#ccc")),
        ("ROWBACKGROUNDS",(1,0), (-1,-1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ]))
    story.append(mt)
    story.append(Spacer(1, .5*cm))
    story.append(HRFlowable(width="100%", thickness=.5,
                             color=colors.HexColor("#ccc")))
    story.append(Spacer(1, .3*cm))

    # Build structured sections from JSON analysis data
    # Section 1: Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", sty_h1))
    
    audit_ts = exec_summary.get("audit_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    story.append(Paragraph(f"<b>Audit Timestamp:</b> {audit_ts}", sty_body))
    
    risk_color = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#f39c12", "LOW": "#27ae60"}.get(risk_level, "#555")
    story.append(Paragraph(f"<b>Risk Level:</b> <font color='{risk_color}'><b>{risk_level}</b></font>", sty_body))
    story.append(Paragraph(f"<b>Overall Compliance Score:</b> <font color='#6c63ff'><b>{compliance_score:.1f}%</b></font>", sty_body))
    story.append(Spacer(1, .2*cm))
    
    # Severity breakdown as a small table
    sev_data = [["Critical", str(severity_breakdown.get("critical", 0))],
                ["High", str(severity_breakdown.get("high", 0))],
                ["Medium", str(severity_breakdown.get("medium", 0))],
                ["Low", str(severity_breakdown.get("low", 0))]]
    sev_tbl = Table(sev_data, colWidths=[3*cm, 2*cm])
    sev_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f0f0f0")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (1,0), (1,-1), "CENTER"),
    ]))
    story.append(sev_tbl)
    story.append(Spacer(1, .4*cm))
    
    # Section 2: Violations Table with Confidence Scores
    violations = analysis.get("violations", [])
    if violations:
        story.append(Paragraph("DETAILED VIOLATION FINDINGS", sty_h1))
        
        # Create violations table with confidence scores
        viol_table_data = [["ID", "Severity", "Confidence", "Type", "Issue Summary"]]
        for v in violations:
            conf = v.get("confidence_score", 0)
            conf_display = f"{conf:.2f}"
            conf_color = "#27ae60" if conf >= 0.8 else ("#f39c12" if conf >= 0.6 else "#e67e22")
            viol_table_data.append([
                v.get("violation_id", "N/A"),
                v.get("severity", "Unknown"),
                f"<font color='{conf_color}'><b>{conf_display}</b></font>",
                v.get("violation_type", "Other")[:25],
                v.get("title", "No title")[:30]
            ])
        
        viol_tbl = Table(viol_table_data, colWidths=[1.5*cm, 2*cm, 2*cm, 3*cm, 7.5*cm], repeatRows=1)
        viol_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#ccc")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("WORDWRAP", (0,0), (-1,-1), True),
        ]))
        story.append(viol_tbl)
        story.append(Spacer(1, .5*cm))
        
        # Detailed findings for each violation
        story.append(Paragraph("EVIDENCE & ANALYSIS", sty_h1))
        for v in violations:
            story.append(Paragraph(f"<b>{v.get('violation_id', 'N/A')} — {v.get('title', 'Untitled')}</b>", sty_h2))
            
            # Confidence score badge
            conf = v.get("confidence_score", 0)
            conf_text = f"Confidence: {conf:.1%}"
            story.append(Paragraph(f"<font color='#6c63ff'><b>{conf_text}</b></font>", sty_confidence))
            
            # Severity
            sev = v.get("severity", "Unknown").lower()
            sev_color = SEV.get(sev, "#555")
            story.append(Paragraph(f"<b>Severity:</b> <font color='{sev_color}'><b>{v.get('severity', 'Unknown')}</b></font>", sty_body))
            
            # Type
            story.append(Paragraph(f"<b>Type:</b> {v.get('violation_type', 'Other')}", sty_body))
            
            # Evidence from uploaded doc
            loc = v.get("location_in_uploaded_doc", {})
            quote = loc.get("exact_quote", "NOT FOUND")
            if len(quote) > 200:
                quote = quote[:200] + "..."
            story.append(Paragraph(f"<b>Found in Document:</b> <i>{quote}</i>", sty_bullet))
            
            # Evidence from reference
            ref = v.get("reference_requirement", {})
            ref_quote = ref.get("exact_quote", "NOT AVAILABLE")
            if len(ref_quote) > 200:
                ref_quote = ref_quote[:200] + "..."
            story.append(Paragraph(f"<b>Reference Requires:</b> <i>{ref_quote}</i>", sty_bullet))
            
            # Analysis/explanation
            analysis_text = v.get("evidence_analysis", "No analysis provided")
            story.append(Paragraph(f"<b>Analysis:</b>", sty_body))
            for para in analysis_text.split("\n\n"):
                if para.strip():
                    try:
                        story.append(Paragraph(inline(para.strip()), sty_body))
                    except:
                        story.append(Paragraph(para.strip(), sty_body))
            
            # Recommendation
            rec = v.get("recommendation", "No recommendation")
            story.append(Paragraph(f"<b>Recommendation:</b> {rec}", sty_bullet))
            
            # Regulatory impact
            impact = v.get("regulatory_impact", "")
            if impact:
                story.append(Paragraph(f"<b>Regulatory Impact:</b> {impact}", sty_bullet))
            
            story.append(Spacer(1, .3*cm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddd")))
            story.append(Spacer(1, .2*cm))
    
    # Section 3: Compliant Sections
    compliant = analysis.get("compliant_sections", [])
    if compliant:
        story.append(Paragraph("COMPLIANT SECTIONS", sty_h1))
        for c in compliant:
            conf = c.get("confidence_score", 0)
            story.append(Paragraph(f"• <b>{c.get('section_name', 'Unnamed')}</b> (Confidence: {conf:.1%})", sty_bullet))
            desc = c.get("description", "")
            if desc:
                story.append(Paragraph(f"  {desc}", sty_body))
        story.append(Spacer(1, .3*cm))
    
    # Section 4: Risk Assessment
    risk_assess = analysis.get("risk_assessment", {})
    story.append(Paragraph("RISK ASSESSMENT", sty_h1))
    overall_risk = risk_assess.get("overall_risk_score", 0)
    story.append(Paragraph(f"<b>Overall Risk Score:</b> {overall_risk:.1f}/100 (higher = worse)", sty_body))
    
    priority_actions = risk_assess.get("top_3_priority_actions", [])
    if priority_actions:
        story.append(Paragraph("<b>Top Priority Actions:</b>", sty_body))
        for i, action in enumerate(priority_actions, 1):
            story.append(Paragraph(f"{i}. {action}", sty_bullet))
    
    effort = risk_assess.get("estimated_remediation_effort", "Unknown")
    story.append(Paragraph(f"<b>Estimated Remediation Effort:</b> {effort}", sty_body))
    
    penalties = risk_assess.get("potential_penalties", "")
    if penalties:
        story.append(Paragraph(f"<b>Potential Penalties:</b> {penalties}", sty_body))
    story.append(Spacer(1, .3*cm))
    
    # Section 5: Audit Methodology (Explainability)
    methodology = analysis.get("audit_methodology", {})
    story.append(Paragraph("AUDIT METHODOLOGY & EXPLAINABILITY", sty_h1))
    story.append(Paragraph(f"<b>RAG Retrieval Method:</b> {methodology.get('rag_retrieval_method', 'Semantic search over vectorized documents')}", sty_body))
    story.append(Paragraph(f"<b>AI Model Used:</b> {methodology.get('model_used', MODEL_NAME)}", sty_body))
    chunks_analyzed = methodology.get("chunks_analyzed", 0)
    story.append(Paragraph(f"<b>Reference Chunks Analyzed:</b> {chunks_analyzed}", sty_body))
    explain_note = methodology.get("explainability_note", "")
    if explain_note:
        story.append(Paragraph(f"<b>Explainability:</b> {explain_note}", sty_bullet))
    story.append(Spacer(1, .3*cm))
    
    # Section 6: Conclusion
    conclusion = analysis.get("conclusion", "No conclusion provided.")
    story.append(Paragraph("AUDIT CONCLUSION", sty_h1))
    try:
        story.append(Paragraph(inline(conclusion), sty_body))
    except:
        story.append(Paragraph(conclusion, sty_body))

    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a1a2e")))
    story.append(Paragraph(
        f"AI-Driven Compliance Auditor  ·  Report {report_id}  ·  "
        f"{datetime.now().strftime('%Y-%m-%d')}  ·  Model: {MODEL_NAME.split('/')[-1]}",
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
