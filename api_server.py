"""
=============================================================================
AI Job Search Agent — FastAPI Backend
=============================================================================
Endpoints:
  GET  /health         — Ollama / Gemini connectivity check
  GET  /jobs           — Return all job postings as JSON
  POST /run-agent      — SSE stream: filter → rank → tailor
  POST /parse-resume   — Extract text from uploaded PDF
  POST /export-resume  — Generate 1-page tailored PDF

Run:
  uvicorn api_server:app --reload --port 8000
=============================================================================
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import AsyncGenerator

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from main import (
    DATASET_PATH,
    MODEL_NAME,
    GEMINI_API_KEY,
    filtering_tool,
    load_dataset,
    ranking_tool,
    resume_tailoring_tool,
    call_ollama_simple,
    call_gemini_simple,
    get_llm_caller,
)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Job Search Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    name: str = "Candidate"
    skills: list[str] = Field(default_factory=list)
    years_experience: int = 2
    preferred_location: str = "Remote"
    remote_only: bool = False
    exclude_companies: list[str] = Field(default_factory=list)
    current_summary: str = ""
    bullet_1: str = ""
    bullet_2: str = ""
    resume_pdf_base64: str = ""
    llm_provider: str = "deepseek"  # "deepseek" or "gemini"


class ExportResumeRequest(BaseModel):
    original_pdf_base64: str
    tailored_summary: str
    tailored_bullet_1: str
    tailored_bullet_2: str
    job_title: str = "Job"
    company: str = "Company"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _check_ollama() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _check_gemini() -> bool:
    return bool(GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ollama_connected": _check_ollama(),
        "gemini_configured": _check_gemini(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /jobs
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/jobs")
def get_jobs():
    return load_dataset(DATASET_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# POST /run-agent  (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────

async def _agent_stream(req: AgentRequest) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()
    provider = req.llm_provider  # "deepseek" or "gemini"
    _, llm_simple_fn = get_llm_caller(provider)
    tailor_timeout = 60 if provider == "gemini" else 300

    profile = {
        "name": req.name,
        "skills": req.skills,
        "years_experience": req.years_experience,
        "preferred_location": req.preferred_location,
        "remote_only": req.remote_only,
        "exclude_companies": req.exclude_companies,
        "current_summary": req.current_summary,
        "bullet_1": req.bullet_1,
        "bullet_2": req.bullet_2,
    }

    try:
        yield _sse({"type": "status", "message": f"Using {provider.upper()} · Loading jobs..."})
        all_jobs = await loop.run_in_executor(None, load_dataset, DATASET_PATH)
        yield _sse({"type": "status", "message": f"Loaded {len(all_jobs)} jobs."})

        # ── Filter ──────────────────────────────────────────────────────────
        yield _sse({"type": "filter_start", "message": f"Filtering {len(all_jobs)} jobs..."})
        filter_result = await loop.run_in_executor(
            None,
            lambda: filtering_tool(
                jobs=all_jobs,
                preferred_location=req.preferred_location,
                max_years_experience=req.years_experience,
                required_skills=req.skills,
                exclude_companies=req.exclude_companies or None,
                remote_only=req.remote_only,
            ),
        )
        filtered_jobs = filter_result["filtered_jobs"]
        yield _sse({
            "type": "filter_result",
            "count": filter_result["count"],
            "trace": filter_result["reasoning_trace"],
            "jobs": [
                {
                    "job_title": j["job_title"],
                    "company": j["company"],
                    "location": j["location"],
                    "years_experience": j["years_experience"],
                    "required_skills": j["required_skills"],
                    "url": j["url"],
                }
                for j in filtered_jobs
            ],
        })

        if not filtered_jobs:
            yield _sse({"type": "error", "message": "No jobs matched your filters. Try broadening your location or skills."})
            yield _sse({"type": "complete", "message": "Done (no matches)"})
            return

        # ── Rank ────────────────────────────────────────────────────────────
        yield _sse({"type": "rank_start", "message": f"Ranking {len(filtered_jobs)} jobs..."})
        rank_result = await loop.run_in_executor(
            None,
            lambda: ranking_tool(
                jobs=filtered_jobs,
                candidate_skills=req.skills,
                candidate_years=req.years_experience,
                preferred_location=req.preferred_location,
            ),
        )
        yield _sse({
            "type": "rank_result",
            "top_3": rank_result["top_3"],
            "all_ranked": rank_result["ranked_jobs"],
        })

        best_job = rank_result["best_job"]
        if not best_job:
            yield _sse({"type": "error", "message": "Ranking produced no results."})
            yield _sse({"type": "complete", "message": "Done"})
            return

        # ── Resume Tailor ───────────────────────────────────────────────────
        yield _sse({
            "type": "tailor_start",
            "message": f"Tailoring resume for {best_job['job_title']} at {best_job['company']}...",
        })

        # Check provider availability
        if provider == "deepseek" and not _check_ollama():
            yield _sse({"type": "error", "message": "Ollama is not running. Start with: ollama serve"})
            yield _sse({"type": "complete", "message": "Done (tailor skipped)"})
            return
        if provider == "gemini" and not _check_gemini():
            yield _sse({"type": "error", "message": "GEMINI_API_KEY not set in .env file."})
            yield _sse({"type": "complete", "message": "Done (tailor skipped)"})
            return

        tailor_result = await loop.run_in_executor(
            None,
            lambda: resume_tailoring_tool(
                job=best_job,
                candidate_profile=profile,
                llm_call_fn=llm_simple_fn,
                timeout=tailor_timeout,
            ),
        )

        if tailor_result["success"]:
            t = tailor_result["tailored"]
            yield _sse({
                "type": "tailor_result",
                "job_title": best_job["job_title"],
                "company": best_job["company"],
                "professional_summary": t.get("professional_summary", ""),
                "bullet_1": t.get("bullet_1_rewritten", ""),
                "bullet_2": t.get("bullet_2_rewritten", ""),
            })
        else:
            yield _sse({
                "type": "tailor_result",
                "job_title": best_job["job_title"],
                "company": best_job["company"],
                "professional_summary": "",
                "bullet_1": "",
                "bullet_2": "",
                "error": "Could not parse tailored JSON from LLM response.",
            })

        yield _sse({"type": "complete", "message": "Done"})

    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        yield _sse({"type": "complete", "message": "Done (error)"})


@app.post("/run-agent")
async def run_agent(req: AgentRequest):
    return StreamingResponse(
        _agent_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /parse-resume  (improved extraction)
# ─────────────────────────────────────────────────────────────────────────────

def _join_wrapped_lines(lines: list[str]) -> list[str]:
    """
    Join lines that are continuation of the same sentence.
    A line is a continuation if the previous line doesn't end with
    a sentence-terminating character and is not a header.
    """
    result = []
    buf = ""
    for line in lines:
        if not line:
            if buf:
                result.append(buf.strip())
                buf = ""
            continue
        if buf:
            # Continuation heuristic: previous didn't end with punctuation
            if not buf.rstrip().endswith((".", ":", "!", "?")):
                buf += " " + line
                continue
            else:
                result.append(buf.strip())
                buf = ""
        buf = line
    if buf:
        result.append(buf.strip())
    return result


def _extract_resume_sections(text: str) -> dict:
    """
    Smarter extraction of professional summary and two bullet points.

    Strategy:
    1. Join wrapped lines to reconstruct full sentences/paragraphs
    2. Look for a SUMMARY/PROFILE/OBJECTIVE section header → extract paragraph
    3. Look for achievement bullets under EXPERIENCE with metrics
    4. Fall back to first long paragraph if no header found
    """
    raw_lines = [l.strip() for l in text.splitlines()]
    joined   = _join_wrapped_lines(raw_lines)

    summary  = ""
    bullets  = []

    # Section header patterns
    summary_headers  = re.compile(r"^(professional\s+)?summary|profile|objective|about\s+me", re.I)
    exp_headers      = re.compile(r"^(work\s+)?experience|employment|positions?|career", re.I)
    bullet_markers   = re.compile(r"^[•\-–\*·▪➢➤►▸]")
    metric_pattern   = re.compile(r"(\d+\s*%|\$\s*\d+|\d+[xX]|\d+\+?\s*(users?|clients?|records?|hours?|days?|years?))", re.I)

    in_summary = False
    in_exp     = False

    for line in joined:
        if not line:
            in_summary = False
            continue

        # Detect section headers
        if summary_headers.match(line) and len(line.split()) <= 6:
            in_summary = True
            in_exp     = False
            continue
        if exp_headers.match(line) and len(line.split()) <= 5:
            in_summary = False
            in_exp     = True
            continue
        # Any ALL-CAPS short line is probably a section header
        if line.isupper() and len(line.split()) <= 5:
            in_summary = False
            in_exp     = False
            continue

        # Collect summary
        if in_summary and not summary and len(line.split()) >= 8:
            summary = line
            in_summary = False  # Only grab the first paragraph
            continue

        # Collect bullets
        if in_exp or bullet_markers.match(line):
            clean = bullet_markers.sub("", line).strip()
            if len(clean.split()) >= 6:
                bullets.append(clean)

    # Fallback: first paragraph with ≥12 words that's not a header/contact line
    if not summary:
        contact_pattern = re.compile(r"@|linkedin\.com|github\.com|\(\d{3}\)|\d{3}-\d{3}", re.I)
        for line in joined:
            if len(line.split()) >= 12 and not contact_pattern.search(line) and not line.isupper():
                summary = line
                break

    # Prefer bullets with metrics if available
    metric_bullets = [b for b in bullets if metric_pattern.search(b)]
    final_bullets  = metric_bullets if len(metric_bullets) >= 2 else bullets

    return {
        "summary":  summary,
        "bullet_1": final_bullets[0] if len(final_bullets) > 0 else "",
        "bullet_2": final_bullets[1] if len(final_bullets) > 1 else "",
        "full_text": text,
    }


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = len(pdf.pages)
            text  = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages  = len(reader.pages)
            text   = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(status_code=500, detail="Install pdfplumber: pip install pdfplumber")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    sections = _extract_resume_sections(text)
    return {**sections, "pages": pages}


# ─────────────────────────────────────────────────────────────────────────────
# POST /export-resume  (professional 1-page PDF)
# ─────────────────────────────────────────────────────────────────────────────

def _build_tailored_pdf(
    original_pdf_bytes: bytes,
    tailored_summary: str,
    tailored_bullet_1: str,
    tailored_bullet_2: str,
    job_title: str,
    company: str,
) -> bytes:
    """
    Build a clean, professional 1-page tailored resume PDF using reportlab.
    Extracts the full original text, replaces summary + bullets with tailored
    versions, and renders with a polished layout. Shrinks font if > 1 page.
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(original_pdf_bytes)) as pdf:
            original_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except ImportError:
        original_text = ""

    orig = _extract_resume_sections(original_text)
    full_text = original_text
    if orig["summary"]  and tailored_summary:   full_text = full_text.replace(orig["summary"],  tailored_summary,  1)
    if orig["bullet_1"] and tailored_bullet_1:  full_text = full_text.replace(orig["bullet_1"], tailored_bullet_1, 1)
    if orig["bullet_2"] and tailored_bullet_2:  full_text = full_text.replace(orig["bullet_2"], tailored_bullet_2, 1)

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.platypus.flowables import HRFlowable
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        raise HTTPException(status_code=500, detail="Install reportlab: pip install reportlab")

    def _render(fs: float) -> bytes:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=letter,
            rightMargin=0.65*inch, leftMargin=0.65*inch,
            topMargin=0.55*inch,   bottomMargin=0.55*inch,
        )
        cyan   = colors.HexColor("#00d4ff")
        dark   = colors.HexColor("#0a0a1a")
        muted  = colors.HexColor("#4a5568")

        name_style = ParagraphStyle("name", fontName="Helvetica-Bold",   fontSize=fs+8, leading=(fs+8)*1.2, textColor=dark,  spaceAfter=2)
        sub_style  = ParagraphStyle("sub",  fontName="Helvetica",        fontSize=fs-1, leading=(fs-1)*1.4, textColor=muted, spaceAfter=8)
        hdr_style  = ParagraphStyle("hdr",  fontName="Helvetica-Bold",   fontSize=fs+1, leading=(fs+1)*1.3, textColor=dark,  spaceBefore=8, spaceAfter=3)
        body_style = ParagraphStyle("body", fontName="Helvetica",        fontSize=fs,   leading=fs*1.45,    textColor=dark,  spaceAfter=3)
        bull_style = ParagraphStyle("bull", fontName="Helvetica",        fontSize=fs,   leading=fs*1.45,    textColor=dark,  leftIndent=10, spaceAfter=2)

        story = []

        # — Header block
        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        # Heuristic: first line is likely the candidate name
        if lines:
            story.append(Paragraph(lines[0], name_style))
        # Second line: contact info
        if len(lines) > 1 and any(c in lines[1] for c in ["@", "|", "•", "·", "linkedin", "github"]):
            story.append(Paragraph(lines[1], sub_style))

        story.append(HRFlowable(width="100%", thickness=1.5, color=cyan, spaceAfter=6))

        # — Tailored for note
        story.append(Paragraph(
            f'<font color="#00d4ff"><b>Tailored for:</b></font> {job_title} at {company}',
            ParagraphStyle("note", fontName="Helvetica", fontSize=fs-1, textColor=muted, spaceAfter=6),
        ))

        # — Professional Summary
        story.append(Paragraph("PROFESSIONAL SUMMARY", hdr_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=muted, spaceAfter=4))
        story.append(Paragraph(tailored_summary or orig["summary"] or "", body_style))

        # — Key Experience (tailored bullets)
        story.append(Paragraph("KEY EXPERIENCE", hdr_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=muted, spaceAfter=4))
        for b in [tailored_bullet_1, tailored_bullet_2]:
            if b:
                story.append(Paragraph(f"• {b}", bull_style))

        # — Remaining original content (skip name/contact/summary/bullets we already used)
        skip_set = {orig["summary"], orig["bullet_1"], orig["bullet_2"], lines[0] if lines else "", lines[1] if len(lines) > 1 else ""}
        skip_set = {s for s in skip_set if s}

        current_section = ""
        for raw_line in lines[2:]:
            if not raw_line or raw_line in skip_set:
                continue
            # Detect section headers
            if raw_line.isupper() and len(raw_line.split()) <= 6:
                current_section = raw_line
                story.append(Paragraph(raw_line, hdr_style))
                story.append(HRFlowable(width="100%", thickness=0.5, color=muted, spaceAfter=3))
            elif raw_line.startswith(("•", "-", "–", "·", "▪", "*")):
                story.append(Paragraph(f"• {raw_line.lstrip('•-–·▪* ')}", bull_style))
            else:
                story.append(Paragraph(raw_line, body_style))

        doc.build(story)
        return buf.getvalue()

    # Shrink font until fits 1 page
    for fs in [10.5, 10.0, 9.5, 9.0, 8.5, 8.0, 7.5]:
        pdf_bytes = _render(fs)
        try:
            import pypdf
            if len(pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages) <= 1:
                return pdf_bytes
        except ImportError:
            return pdf_bytes

    return _render(7.5)


@app.post("/export-resume")
def export_resume(req: ExportResumeRequest):
    try:
        original_pdf_bytes = base64.b64decode(req.original_pdf_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data.")

    pdf_bytes = _build_tailored_pdf(
        original_pdf_bytes=original_pdf_bytes,
        tailored_summary=req.tailored_summary,
        tailored_bullet_1=req.tailored_bullet_1,
        tailored_bullet_2=req.tailored_bullet_2,
        job_title=req.job_title,
        company=req.company,
    )

    safe = lambda s: re.sub(r"[^\w\-]", "_", s)
    filename = f"resume_tailored_{safe(req.job_title)}_{safe(req.company)}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
