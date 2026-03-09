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
    filtering_tool,
    load_dataset,
    ranking_tool,
    resume_tailoring_tool,
    call_ollama_simple,
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


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ollama_connected": _check_ollama(),
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
    provider = "llama"
    _, llm_simple_fn = get_llm_caller(provider)
    tailor_timeout = 300

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
        yield _sse({"type": "status", "message": "Using Qwen 3.5 4B · Loading jobs..."})
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

        # Check Ollama availability
        if not _check_ollama():
            yield _sse({"type": "error", "message": "Ollama is not running. Start with: ollama serve"})
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
# POST /parse-resume
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_from_pdf(content: bytes) -> tuple[str, int]:
    """
    Extract text from PDF bytes using pdfplumber with word-level reconstruction
    to fix missing-space issues common in columnar/ATS-formatted resumes.
    Falls back to pypdf if pdfplumber is unavailable.
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages_count = len(pdf.pages)
            page_texts = []
            for page in pdf.pages:
                # Use extract_words for proper spacing, then reconstruct lines
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=True,
                )
                if not words:
                    page_texts.append(page.extract_text() or "")
                    continue

                # Group words into lines by y-position proximity
                lines: list[list[dict]] = []
                for w in words:
                    placed = False
                    for line in lines:
                        if abs(w["top"] - line[0]["top"]) < 4:
                            line.append(w)
                            placed = True
                            break
                    if not placed:
                        lines.append([w])

                # Sort lines top-to-bottom, words left-to-right, join with spaces
                lines.sort(key=lambda l: l[0]["top"])
                reconstructed = []
                for line_words in lines:
                    line_words.sort(key=lambda w: w["x0"])
                    reconstructed.append(" ".join(w["text"] for w in line_words))

                page_texts.append("\n".join(reconstructed))
            return "\n".join(page_texts), pages_count
    except ImportError:
        pass

    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        return text, len(reader.pages)
    except ImportError:
        raise HTTPException(status_code=500, detail="Install pdfplumber: pip install pdfplumber")


def _extract_resume_sections(text: str) -> dict:
    """
    Extract professional summary and two achievement bullets from resume text.

    Strategy:
    1. Find SUMMARY / PROFILE / OBJECTIVE section header → grab lines until
       the next section header is detected (blank line OR ALL-CAPS header)
    2. Cap summary at 5 lines / 80 words to prevent over-capture
    3. Find bullet points with metrics anywhere in the resume
    4. Prefer bullets containing quantified achievements (%, $, numbers)
    """
    lines = [l.strip() for l in text.splitlines()]

    summary_headers = re.compile(
        r"^(professional\s+)?(summary|profile|objective|about\s+me)$", re.I
    )
    # Any ALL-CAPS word(s) 3+ chars = section header
    section_header = re.compile(r"^[A-Z][A-Z\s&/\-]{2,}$")
    bullet_markers = re.compile(r"^[•\-–\*·▪➢➤►▸◆]\s*")
    metric_pattern = re.compile(
        r"(\d+\s*%|\$\s*[\d,]+|\d+[xX]|\d+\+?\s*(users?|clients?|records?|hours?|days?|years?|ms|gb|tb|k\b))",
        re.I,
    )
    contact_pattern = re.compile(
        r"@|linkedin\.com|github\.com|\(\d{3}\)|\d{3}[-\.]\d{3}|portfolio", re.I
    )
    # Lines that look like job titles / dates — stop summary collection
    date_pattern = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|\d{4}", re.I)

    summary = ""
    bullets: list[str] = []
    in_summary = False
    summary_buf: list[str] = []

    def _is_section_boundary(line: str) -> bool:
        """Return True if this line is a new section header."""
        if not line:
            return True
        stripped = line.strip()
        # ALL-CAPS header (e.g. "EXPERIENCE", "TECHNICAL SKILLS")
        if section_header.match(stripped) and len(stripped.split()) <= 5:
            return True
        # Title-case known section words
        if re.match(r"^(Experience|Education|Skills|Projects|Certifications|Awards|Languages)\b", stripped):
            return True
        return False

    for line in lines:
        if not line:
            # Blank line: flush summary if we have enough content
            if in_summary and summary_buf:
                candidate = " ".join(summary_buf)
                if len(candidate.split()) >= 8:
                    summary = candidate
                    in_summary = False
                    summary_buf = []
            continue

        is_sum_hdr = bool(summary_headers.match(line)) and len(line.split()) <= 4

        if is_sum_hdr:
            in_summary = True
            summary_buf = []
            continue

        if in_summary:
            # Stop collecting if we hit a new section or a date/company line
            if _is_section_boundary(line):
                if summary_buf:
                    candidate = " ".join(summary_buf)
                    if len(candidate.split()) >= 8:
                        summary = candidate
                summary_buf = []
                in_summary = False
                # Don't skip — process as normal line below
            elif contact_pattern.search(line):
                pass  # skip contact lines inside summary zone
            elif date_pattern.search(line) and len(line.split()) <= 6:
                # Looks like "University of Houston  Mar 2025 – Dec 2025" — stop
                if summary_buf:
                    candidate = " ".join(summary_buf)
                    if len(candidate.split()) >= 8:
                        summary = candidate
                summary_buf = []
                in_summary = False
            else:
                summary_buf.append(line)
                # Hard cap: 5 lines or 80 words — flush early
                if len(summary_buf) >= 5 or len(" ".join(summary_buf).split()) >= 80:
                    candidate = " ".join(summary_buf)
                    summary = candidate
                    summary_buf = []
                    in_summary = False

        # Collect bullets regardless of section
        if bullet_markers.match(line):
            clean = bullet_markers.sub("", line).strip()
            if len(clean.split()) >= 6 and not contact_pattern.search(clean):
                bullets.append(clean)

    # Flush summary if file ended while still collecting
    if in_summary and summary_buf and not summary:
        candidate = " ".join(summary_buf)
        if len(candidate.split()) >= 8:
            summary = candidate

    # Fallback: first long non-contact, non-header line
    if not summary:
        for line in lines:
            if (
                len(line.split()) >= 12
                and not contact_pattern.search(line)
                and not section_header.match(line)
                and not date_pattern.search(line)
            ):
                # Truncate to ~80 words to avoid grabbing full paragraphs
                words = line.split()
                summary = " ".join(words[:80])
                break

    # Prefer metric-containing bullets; deduplicate
    seen = set()
    unique_bullets = []
    for b in bullets:
        key = b[:60]
        if key not in seen:
            seen.add(key)
            unique_bullets.append(b)

    metric_bullets = [b for b in unique_bullets if metric_pattern.search(b)]
    final_bullets = metric_bullets if len(metric_bullets) >= 2 else unique_bullets

    return {
        "summary": summary,
        "bullet_1": final_bullets[0] if len(final_bullets) > 0 else "",
        "bullet_2": final_bullets[1] if len(final_bullets) > 1 else "",
        "full_text": text,
    }


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    text, pages = _extract_text_from_pdf(content)

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
    Overlay tailored text on the original PDF.

    For each replacement (old → new):
    1. Use pdfplumber to find the exact bounding box of every word in old_text
    2. Draw a white rectangle covering the entire old block (with padding)
    3. Render new_text at the same position using the same font-size as the
       original, word-wrapped to the same column width — so layout is preserved
    4. Merge the transparent overlay onto each original page via pypdf
    """
    try:
        import pypdf
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        raise HTTPException(status_code=500, detail="Install reportlab and pypdf: pip install reportlab pypdf")

    try:
        import pdfplumber
    except ImportError:
        raise HTTPException(status_code=500, detail="Install pdfplumber: pip install pdfplumber")

    # ── Collect original text sections for matching ──────────────────────────
    with pdfplumber.open(io.BytesIO(original_pdf_bytes)) as pdf:
        original_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    orig = _extract_resume_sections(original_text)

    replacements = []
    if tailored_summary and orig.get("summary"):
        replacements.append((orig["summary"], tailored_summary))
    if tailored_bullet_1 and orig.get("bullet_1"):
        replacements.append((orig["bullet_1"], tailored_bullet_1))
    if tailored_bullet_2 and orig.get("bullet_2"):
        replacements.append((orig["bullet_2"], tailored_bullet_2))

    reader = pypdf.PdfReader(io.BytesIO(original_pdf_bytes))

    def _find_block_bbox(pg, search_text: str):
        """
        Find the bounding box of search_text on a pdfplumber page.

        Strategy: extract all words, join into a flat string, find the span
        of words that best matches the first ~8 words of search_text, then
        collect all word bboxes in that span.

        Returns (x0, y0_pdf, x1, y1_pdf, font_size_est) in pdfplumber coords
        (origin top-left), or None.
        """
        words = pg.extract_words(x_tolerance=3, y_tolerance=3, use_text_flow=True)
        if not words:
            return None

        # Normalise search tokens (first 10 words for matching anchor)
        anchor_tokens = [t.lower().strip(".,;:\"'()") for t in search_text.split()[:10]]
        word_tokens   = [w["text"].lower().strip(".,;:\"'()") for w in words]

        # Sliding-window: find best match start index
        best_start = -1
        best_score = 0
        for i in range(len(word_tokens) - len(anchor_tokens) + 1):
            score = sum(
                1 for a, b in zip(anchor_tokens, word_tokens[i:i + len(anchor_tokens)]) if a == b
            )
            if score > best_score:
                best_score = score
                best_start = i

        if best_score < max(2, len(anchor_tokens) // 3) or best_start < 0:
            return None

        # Now estimate how many words the full old text spans
        total_tokens = search_text.split()
        end_idx = min(best_start + len(total_tokens) + 5, len(words))

        span = words[best_start:end_idx]
        if not span:
            return None

        x0   = min(w["x0"]     for w in span)
        x1   = max(w["x1"]     for w in span)
        top  = min(w["top"]    for w in span)
        bot  = max(w["bottom"] for w in span)

        # Estimate font size from word heights
        heights = [w["bottom"] - w["top"] for w in span if w["bottom"] > w["top"]]
        font_est = round(sum(heights) / len(heights), 1) if heights else 10.0

        # Use full page width for x1 so wrapped text has room
        x1_full = pg.width - (pg.width - x1) * 0.3  # keep some right margin

        return x0, top, x1_full, bot, font_est

    def _make_overlay(pg_width: float, pg_height: float, block_info: list) -> bytes:
        """
        Build a single-page overlay PDF containing whiteout rects + new text.
        block_info: list of (x0, top, x1, bot, font_size, new_text) in
                    pdfplumber coords (origin top-left).
        """
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=(pg_width, pg_height))

        for x0, top, x1, bot, font_size, new_text in block_info:
            # Convert pdfplumber top-left coords → reportlab bottom-left
            rl_top = pg_height - top    # top of block in RL coords
            rl_bot = pg_height - bot    # bottom of block in RL coords

            line_h    = font_size * 1.35
            col_width = x1 - x0

            # Estimate wrapped line count for new text
            avg_char_w = font_size * 0.52
            chars_per_line = max(int(col_width / avg_char_w), 20)
            wrapped = textwrap.wrap(new_text, width=chars_per_line)
            new_block_h = len(wrapped) * line_h

            # White rectangle: tall enough for both old and new text
            rect_h = max(rl_top - rl_bot + 4, new_block_h + line_h)
            c.setFillColorRGB(1, 1, 1)
            c.rect(x0 - 2, rl_top - rect_h, col_width + 4, rect_h + 2,
                   fill=1, stroke=0)

            # Draw new text top-down
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica", font_size)
            y = rl_top - font_size
            for ln in wrapped:
                c.drawString(x0, y, ln)
                y -= line_h

        c.save()
        buf.seek(0)
        return buf.read()

    # ── Build per-page overlays ───────────────────────────────────────────────
    writer = pypdf.PdfWriter()

    with pdfplumber.open(io.BytesIO(original_pdf_bytes)) as plumb_pdf:
        for page_idx, orig_page in enumerate(reader.pages):
            pg_w = float(orig_page.mediabox.width)
            pg_h = float(orig_page.mediabox.height)

            if page_idx < len(plumb_pdf.pages):
                plumb_pg = plumb_pdf.pages[page_idx]
                block_info = []
                for old_text, new_text in replacements:
                    bbox = _find_block_bbox(plumb_pg, old_text)
                    if bbox:
                        x0, top, x1, bot, fs = bbox
                        block_info.append((x0, top, x1, bot, fs, new_text))

                if block_info:
                    overlay_bytes = _make_overlay(pg_w, pg_h, block_info)
                    overlay_page = pypdf.PdfReader(io.BytesIO(overlay_bytes)).pages[0]
                    orig_page.merge_page(overlay_page)

            writer.add_page(orig_page)

    out_buf = io.BytesIO()
    writer.write(out_buf)
    return out_buf.getvalue()


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


# ─────────────────────────────────────────────────────────────────────────────
# POST /export-resume-docx  (in-place DOCX edit — preserves all formatting)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_docx_edits(docx_bytes: bytes, edits: list[dict]) -> bytes:
    """
    Apply a list of {original_text, new_text} replacements to a .docx file,
    doing run-level surgery to preserve bold/italic/font/size/color exactly.

    Strategy (mirrors the reference implementation):
    1. For each paragraph's runs, try to find original_text in a single run
       → replace only that run's text (all formatting XML untouched)
    2. If not found in a single run, fall back to paragraph-level replace
       (loses intra-run formatting for that paragraph only, keeps paragraph style)
    3. Repeat for all table cells too (handles layout tables in resumes)
    """
    try:
        from docx import Document
    except ImportError:
        raise HTTPException(status_code=500, detail="Install python-docx: pip install python-docx")

    doc = Document(io.BytesIO(docx_bytes))

    def _apply_to_paragraph(para, original: str, replacement: str):
        # Try run-level replacement first (preserves formatting perfectly)
        for run in para.runs:
            if original in run.text:
                run.text = run.text.replace(original, replacement, 1)
                return True
        # Fallback: paragraph-level (loses intra-run bold/italic for this para)
        full = para.text
        if original in full:
            # Replace text across runs by clearing all but first, setting full text
            new_full = full.replace(original, replacement, 1)
            # Put new text in first run, clear the rest
            if para.runs:
                para.runs[0].text = new_full
                for run in para.runs[1:]:
                    run.text = ""
            return True
        return False

    for edit in edits:
        orig = edit.get("original_text", "").strip()
        new  = edit.get("new_text", "").strip()
        if not orig or not new or orig == new:
            continue

        # Search all body paragraphs
        for para in doc.paragraphs:
            _apply_to_paragraph(para, orig, new)

        # Search all table cells
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        _apply_to_paragraph(para, orig, new)

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


class ExportDocxRequest(BaseModel):
    original_docx_base64: str
    tailored_summary: str
    tailored_bullet_1: str
    tailored_bullet_2: str
    original_summary: str = ""
    original_bullet_1: str = ""
    original_bullet_2: str = ""
    job_title: str = "Job"
    company: str = "Company"


@app.post("/export-resume-docx")
def export_resume_docx(req: ExportDocxRequest):
    try:
        docx_bytes = base64.b64decode(req.original_docx_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 DOCX data.")

    edits = []
    if req.original_summary and req.tailored_summary:
        edits.append({"original_text": req.original_summary, "new_text": req.tailored_summary})
    if req.original_bullet_1 and req.tailored_bullet_1:
        edits.append({"original_text": req.original_bullet_1, "new_text": req.tailored_bullet_1})
    if req.original_bullet_2 and req.tailored_bullet_2:
        edits.append({"original_text": req.original_bullet_2, "new_text": req.tailored_bullet_2})

    if not edits:
        raise HTTPException(status_code=400, detail="No edits to apply — provide original and tailored text.")

    result_bytes = _apply_docx_edits(docx_bytes, edits)

    safe = lambda s: re.sub(r"[^\w\-]", "_", s)
    filename = f"resume_tailored_{safe(req.job_title)}_{safe(req.company)}.docx"

    return Response(
        content=result_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
