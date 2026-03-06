# 🤖 Claude Code — AI Job Search Agent: Full Build Plan
## Instructions for Autonomous Execution

---

## ✅ PROGRESS TRACKER

Use this as a handoff checklist. Check off items as they are completed.

### Phase 1 — Bug Fixes in `main.py`
- [x] **1.1** Strip `<think>` blocks in `_extract_tool_call()`, try multiple patterns, inject reminder after 2 consecutive failures
- [x] **1.2** Change Ollama timeout to 300 s for `tailor_resume`; add `OLLAMA_TIMEOUT` constant; add `--timeout` CLI flag
- [x] **1.3** Update `jobs_dataset.csv` to comma-separated skills; fix `filtering_tool` to use `.split(",")`
- [x] **1.4** Add state machine `{"filtered", "ranked", "tailored"}` — block `finish` until all three are True
- [x] **1.5** Cascade JSON parser in `resume_tailoring_tool`: strip fences → `json.loads` → regex → line-by-line
- [x] **1.6** Add `--dry-run` CLI flag (filter + rank, no LLM calls)
- [x] **Bonus** Added `from __future__ import annotations` for Python 3.9 compatibility

### Phase 2 — Unit Tests (`tests/test_agent.py`)
- [x] Test 1 — `filtering_tool`: remote-only filter
- [x] Test 2 — `filtering_tool`: experience cap (≤ candidate_years + 1)
- [x] Test 3 — `filtering_tool`: company exclusion
- [x] Test 4 — `filtering_tool`: skill overlap
- [x] Test 5 — `ranking_tool`: all scores 0–100
- [x] Test 6 — `ranking_tool`: first job has highest score
- [x] Test 7 — `ranking_tool`: skill score proportional
- [x] Test 8 — `ranking_tool`: `top_3` has ≤ 3 items
- [x] Test 9 — Dataset integrity: 25 rows, 7 columns, valid years, no empty fields
- [x] Test 10 — State machine: `finish` blocked when `tailored=False`
- [x] **Bonus Test 11** — State machine: `finish` accepted after all 3 steps done
- [x] **All 11 tests pass** (`pytest tests/test_agent.py -v`)

### Phase 3 — FastAPI Backend (`api_server.py`)
- [x] `GET /health` — Ollama connectivity check
- [x] `GET /jobs` — return all 25 jobs as JSON
- [x] `POST /run-agent` — SSE streaming: status → filter → rank → tailor → complete
- [x] `POST /parse-resume` — PDF upload → extract summary + bullets (pdfplumber / pypdf fallback)
- [x] `POST /export-resume` — rebuild 1-page PDF with tailored sections (reportlab)
- [x] CORS open (`allow_origins=["*"]`)

### Phase 4 — Frontend (`frontend/index.html`)
- [x] Hero header: animated hexagon logo, Ollama status dot (calls `/health` on load)
- [x] Candidate form: tag-chip skills input, location, years, remote toggle, exclusions
- [x] Resume upload: drag-drop PDF zone → auto-calls `/parse-resume` → auto-fills form
- [x] Run Agent button: disabled + spinner while running
- [x] Live execution panel: terminal-style SSE trace + 3 step cards
- [x] Results dashboard: Top 3 job cards with animated score bars
- [x] Resume side-by-side diff (original vs tailored)
- [x] Export PDF button → calls `/export-resume` → triggers download
- [x] Single self-contained file (no local imports)

### Phase 5 — Integration Test
- [ ] `curl http://localhost:8000/health` returns `ollama_connected: true`
- [ ] `POST /run-agent` SSE stream completes all events
- [ ] Frontend loads in browser and health dot turns green

### Phase 6 — Final Checks
- [x] `pytest tests/test_agent.py -v` → 11 passed
- [x] `python main.py --dry-run` → completes without errors
- [x] `frontend/index.html` is self-contained (no local file imports)
- [x] `README.md` updated with full-stack instructions
- [ ] `frontend/index.html` opens in browser with green health dot (requires uvicorn running)

---

## WHAT'S BEEN BUILT (COMPLETED FILES)

| File | Status | Notes |
|------|--------|-------|
| `main.py` | Done | 6 bugs fixed + `--dry-run` + `--timeout` |
| `jobs_dataset.csv` | Done | Skills now comma-separated |
| `requirements.txt` | Done | All deps including FastAPI, reportlab, pdfplumber |
| `api_server.py` | Done | Full FastAPI backend with SSE |
| `frontend/index.html` | Done | Single-file UI, Terminal Intelligence theme |
| `tests/test_agent.py` | Done | 11 tests, all passing |
| `README.md` | Done | Full-stack instructions |

---

---

## CONTEXT & REFERENCE FILES

You have been given 3 reference files. Use them as follows:
- **`main.py`** — The core agent logic. This is your foundation. Do NOT rewrite from scratch; extend and fix it.
- **`jobs_dataset.csv`** — The 25-job dataset. Keep it as-is; only modify if a bug requires it.
- **`README.md`** — Setup documentation. Update it at the end with any changes you make.

---

## PROJECT STRUCTURE TO CREATE

```
job_search_agent/
├── main.py                  ← (provided) Core agent — fix bugs, do NOT rewrite
├── jobs_dataset.csv         ← (provided) Dataset — keep as-is
├── README.md                ← (provided) Update at end
├── requirements.txt         ← Create: all Python dependencies
├── api_server.py            ← Create: FastAPI backend serving the agent
├── frontend/
│   └── index.html           ← Create: Single-file interactive frontend
└── tests/
    └── test_agent.py        ← Create: Unit tests for all three tools
```

---

## PHASE 1 — BUG FIXES IN `main.py`

Read `main.py` carefully and fix ALL of the following known issues:

### 1.1 — Tool Call JSON Extraction
The `_extract_tool_call()` method uses a fragile regex. DeepSeek-R1:8b often wraps its response in `<think>...</think>` blocks and the JSON may appear AFTER them. Fix:
- Strip `<think>...</think>` blocks FIRST before attempting regex extraction
- Try multiple JSON extraction patterns (```tool_call, bare JSON with "tool" key, JSON anywhere in response)
- Add a fallback: if no tool call detected after 2 consecutive iterations, inject a reminder message to the LLM to output a tool call

### 1.2 — Ollama Timeout
120 seconds is too low for resume tailoring on a cold model. Change to 300 seconds for the `tailor_resume` tool call specifically. Add a `--timeout` CLI flag.

### 1.3 — Skill Matching in `ranking_tool`
The current skill match splits `required_skills` on spaces, but the CSV uses comma-separated skills. Fix the split to use `","` as the delimiter consistently across both `filtering_tool` and `ranking_tool`.

### 1.4 — Agent Loop Termination
If the LLM calls `finish` before `tailor_resume` (it sometimes skips steps), detect this and force the missing steps before allowing termination. Add a state machine: `["filtered", "ranked", "tailored"]` must all be True before `finish` is accepted.

### 1.5 — Resume Tailoring JSON Parse
The `resume_tailoring_tool` regex sometimes fails if the LLM adds markdown around the JSON. Make the parser more robust:
- Strip ```json ... ``` fences
- Try `json.loads()` on the full response first
- Then try the regex
- Finally, manually extract fields using line-by-line parsing as last resort

### 1.6 — Add `--dry-run` flag
Add a `--dry-run` CLI argument that runs filtering + ranking WITHOUT calling Ollama (uses rule-based logic only, skips resume tailoring). Useful for testing the dataset and tool logic without needing the model.

```bash
python main.py --dry-run
```

---

## PHASE 2 — UNIT TESTS (`tests/test_agent.py`)

Write pytest unit tests. Run them and fix until ALL pass.

### Tests to write:

```python
# Test 1: filtering_tool — location filter
# Input: jobs with mix of Remote and on-site, preferred_location="Remote"
# Expected: only remote jobs returned

# Test 2: filtering_tool — experience cap
# Input: jobs requiring 0-5 years, candidate has 2 years
# Expected: no job requiring >3 years returned

# Test 3: filtering_tool — company exclusion
# Input: jobs from Google, Meta, OpenAI, preferred exclusion=["Google"]
# Expected: no Google jobs in output

# Test 4: filtering_tool — skill overlap
# Input: candidate with skills=["Python"], jobs with/without Python
# Expected: only Python jobs returned

# Test 5: ranking_tool — correct score range
# All scores must be 0–100

# Test 6: ranking_tool — top job has highest score
# The first item in ranked_jobs must have the highest score

# Test 7: ranking_tool — skill score proportional
# A job matching 5/5 candidate skills should outscore one matching 1/5

# Test 8: ranking_tool — returns top_3 key with exactly 3 items (or fewer if <3 jobs)

# Test 9: dataset CSV integrity
# Load jobs_dataset.csv and verify: 25 rows, all 7 required columns present,
# years_experience is numeric, no empty job_title or company

# Test 10: agent state machine
# Verify that finish is rejected when tailored=False
```

Run tests:
```bash
cd job_search_agent
pytest tests/test_agent.py -v
```

Fix any failing tests before moving to Phase 3.

---

## PHASE 3 — FASTAPI BACKEND (`api_server.py`)

Create a FastAPI server that exposes the agent over HTTP so the frontend can call it.

### Endpoints:

#### `GET /health`
Returns `{"status": "ok", "model": "deepseek-r1:8b", "ollama_connected": bool}`

#### `GET /jobs`
Returns the full jobs dataset as JSON array.

#### `POST /run-agent`
**Request body:**
```json
{
  "name": "Somil",
  "skills": ["Python", "SQL", "Machine Learning"],
  "years_experience": 2,
  "preferred_location": "Remote",
  "remote_only": false,
  "exclude_companies": [],
  "current_summary": "...",
  "bullet_1": "...",
  "bullet_2": "...",
  "resume_pdf_base64": "..."   // optional: base64 encoded PDF
}
```

**Response (Server-Sent Events stream):**
The agent must stream progress events so the frontend can show live updates.
Use `StreamingResponse` with `text/event-stream` content type.

Stream events in this format:
```
data: {"type": "status", "message": "Loading dataset..."}
data: {"type": "filter_start", "message": "Filtering 25 jobs..."}
data: {"type": "filter_result", "count": 18, "jobs": [...]}
data: {"type": "rank_start", "message": "Ranking filtered jobs..."}
data: {"type": "rank_result", "top_3": [...], "all_ranked": [...]}
data: {"type": "tailor_start", "message": "Tailoring resume for NLP Engineer at Hugging Face..."}
data: {"type": "tailor_result", "professional_summary": "...", "bullet_1": "...", "bullet_2": "..."}
data: {"type": "agent_trace", "iteration": 1, "reasoning": "..."}
data: {"type": "complete", "message": "Done"}
data: {"type": "error", "message": "..."}
```

#### `POST /parse-resume`
Accepts a PDF upload (multipart form data). Extracts text from the PDF and returns:
```json
{
  "summary": "extracted professional summary...",
  "bullet_1": "extracted first bullet...",
  "bullet_2": "extracted second bullet...",
  "full_text": "...",
  "pages": 1
}
```
Use `pypdf` or `pdfplumber` for extraction.

### CORS:
Enable CORS for all origins (development mode).

### Run:
```bash
pip install fastapi uvicorn pypdf pdfplumber python-multipart
uvicorn api_server:app --reload --port 8000
```

---

## PHASE 4 — INTERACTIVE FRONTEND (`frontend/index.html`)

### Design Direction: **"Terminal Intelligence"**
Dark theme. Monospace + modern sans-serif pairing. Feels like a sophisticated AI terminal that's also beautiful. Think: a Bloomberg terminal crossed with a modern SaaS dashboard. NOT generic purple gradients.

**Color palette:**
- Background: `#0a0a0f` (near-black with blue tint)
- Surface: `#111118` 
- Panel: `#16161f`
- Border: `#2a2a3a`
- Accent: `#00d4ff` (electric cyan)
- Accent 2: `#7c3aed` (deep violet)
- Success: `#10b981`
- Text primary: `#e2e8f0`
- Text muted: `#64748b`

**Typography:**
- Display / headings: `"Syne"` from Google Fonts (geometric, modern)
- Body text: `"DM Sans"` from Google Fonts
- Code / monospace / scores: `"JetBrains Mono"` from Google Fonts

### Frontend Structure (single `index.html` file, all CSS and JS inline):

#### Section 1 — Hero Header
- Animated logo: rotating hexagon with "AI" inside, using CSS animation
- Title: "Job Search Agent" with a subtle scanning line animation underneath
- Subtitle: "Powered by DeepSeek-R1 · Local LLM · Zero Data Sent to Cloud"
- Status indicator: pulsing green dot + "Agent Ready" or red "Ollama Offline"
- On page load, call `GET /health` to set the status indicator

#### Section 2 — Candidate Profile Form
Split into two columns:

**Left column — Profile Setup:**
- Name input (text)
- Skills input: tag-style input where user types a skill and presses Enter/comma to add it as a removable chip/tag
- Years of Experience: stylized number input with +/- buttons
- Preferred Location: text input with suggestions dropdown ("Remote", "San Francisco", "New York", "Houston")
- Remote Only toggle: iOS-style toggle switch
- Exclude Companies: same tag-style chip input as skills

**Right column — Resume Upload:**
- Large drag-and-drop zone with dashed animated border
- Label: "Drop your resume PDF here" with upload icon
- On file drop/select: call `POST /parse-resume` to extract content
- Show extraction progress with animated spinner
- After extraction: show a preview card with extracted summary and bullets (editable text areas)
- **IMPORTANT**: Display a notice: "Your 1-page resume formatting will be preserved in tailored output"
- Text areas for: Professional Summary, Experience Bullet 1, Experience Bullet 2

#### Section 3 — Run Agent Button
- Full-width glowing button: "⚡ Find My Best Job Match"
- On hover: border glow pulses outward
- While running: button text changes to "Agent Running..." with animated dots
- Disabled state while agent is running

#### Section 4 — Live Agent Execution Panel (appears when agent runs)
This is the most important section. Shows the agent's reasoning in real time.

**Left side — Agent Trace Terminal:**
- Dark terminal-style panel with monospace font
- Each SSE event renders as a new line with timestamp
- Color-coded lines:
  - `[FILTER]` → cyan
  - `[RANK]` → violet
  - `[TAILOR]` → green  
  - `[TRACE]` → muted grey
  - `[ERROR]` → red
- Lines animate in one by one (typewriter effect, fast)
- Auto-scrolls to bottom
- Progress bar at top showing 3 steps: Filter → Rank → Tailor (each step lights up as it completes)

**Right side — Live Results:**
- Step 1 complete → show "Filtered Jobs" counter + small list
- Step 2 complete → show animated score cards for Top 3 jobs
  - Each card: job title, company, location, score bar (animated fill), skill/exp/loc breakdown
  - Cards slide in with staggered animation
  - Best job card has a glowing border
- Step 3 complete → show resume tailoring output (see Section 5)

#### Section 5 — Results Dashboard (appears after agent completes)

**Top 3 Jobs Cards:**
- 3 horizontal cards in a row (or vertical on mobile)
- Each card shows:
  - Rank badge (#1, #2, #3) — gold/silver/bronze colors
  - Job title (large)
  - Company name + location
  - Total score (large number) with animated count-up
  - Score breakdown: 3 mini horizontal bars for Skill / Experience / Location
  - "View Posting" button linking to job URL

**Best Job highlight:**
- Larger featured card for #1 job
- Full job description
- All required skills shown as chips (matching skills highlighted in cyan)

**Tailored Resume Output:**
- Side-by-side comparison: "Original" vs "Tailored"
- Professional Summary: original left, tailored right — with diff highlighting (changed words underlined in cyan)
- Bullet 1 comparison
- Bullet 2 comparison
- **"Export Tailored Resume" button** — generates a 1-page PDF download

#### Section 6 — Export Tailored Resume as 1-Page PDF

**CRITICAL REQUIREMENT**: When user clicks "Export Tailored Resume":

1. Take the ORIGINAL uploaded resume PDF
2. Using a client-side PDF manipulation approach OR calling a backend endpoint:
   - Replace ONLY the Professional Summary section with the tailored version
   - Replace ONLY the two experience bullet points with their tailored versions
   - Preserve ALL original formatting: fonts, layout, margins, colors, headers, other sections
   - Ensure the result is strictly 1 page (if content overflows, slightly reduce font size of the modified sections only)
3. Trigger a download of the modified PDF named `resume_tailored_{job_title}_{company}.pdf`

**Implementation approach for PDF modification:**
- Use `pdf-lib` JavaScript library (CDN: `https://cdnjs.cloudflare.com/ajax/libs/pdf-lib/1.17.1/pdf-lib.min.js`) for client-side PDF manipulation
- Alternative: add a `POST /export-resume` backend endpoint that accepts the original PDF + tailored text, and uses `reportlab` or `fpdf2` in Python to generate the 1-page output
- The backend approach is MORE RELIABLE for formatting preservation — implement it

**Backend endpoint `POST /export-resume`:**
```json
Request: {
  "original_pdf_base64": "...",
  "tailored_summary": "...",
  "tailored_bullet_1": "...",
  "tailored_bullet_2": "...",
  "job_title": "NLP Engineer",
  "company": "Hugging Face"
}
Response: PDF file binary (application/pdf)
```

The backend should:
1. Parse the original PDF with `pdfplumber` to extract layout information
2. Identify the summary and bullet point regions by text content matching
3. Use `reportlab` to recreate the page with replacements in the same positions
4. If content is >1 page, reduce font size by 0.5pt increments until it fits
5. Return the 1-page PDF

#### Additional UX Details:
- Smooth scroll between sections
- Mobile responsive (stack columns on <768px)
- All animations use CSS transitions, not JS-heavy libraries (keep it snappy)
- Error states: if Ollama is offline, show a friendly error with setup instructions
- Loading skeletons while waiting for results
- The page should feel alive even when idle (subtle background particle effect using canvas, or CSS animated gradient mesh)

---

## PHASE 5 — INTEGRATION TEST

After building everything, run this full integration test:

```bash
# Terminal 1
ollama serve

# Terminal 2
cd job_search_agent
uvicorn api_server:app --reload --port 8000

# Terminal 3
# Open frontend/index.html in browser (double-click or use live server)
# OR: python3 -m http.server 3000 --directory frontend/

# Integration test:
curl http://localhost:8000/health
# Expected: {"status":"ok","model":"deepseek-r1:8b","ollama_connected":true}

curl -X POST http://localhost:8000/run-agent \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","skills":["Python","SQL","Machine Learning"],"years_experience":2,"preferred_location":"Remote","remote_only":false,"exclude_companies":[],"current_summary":"Data scientist with Python skills","bullet_1":"Built ETL pipelines","bullet_2":"Created ML models"}'
# Expected: SSE stream with filter/rank/tailor events
```

---

## PHASE 6 — FINAL CHECKS

1. Run `pytest tests/test_agent.py -v` — ALL tests must pass
2. Run `python main.py --dry-run` — must complete without errors
3. Open `frontend/index.html` in Chrome — check that health check shows green
4. Verify `frontend/index.html` is a single self-contained file (no external local imports)
5. Update `README.md` with:
   - Final project structure
   - How to run the full stack (agent + API + frontend)
   - Any changes made to `main.py`

---

## NOTES FOR CLAUDE CODE

- The `main.py` provided uses Ollama at `http://localhost:11434`. Do not change this.
- The model is `deepseek-r1:8b`. Do not change this — it's been validated for M4 MacBook Air 16GB.
- DeepSeek-R1 wraps its reasoning in `<think>...</think>` tags — always strip these before displaying to the user but DO show them in the terminal trace panel (dimmed).
- `jobs_dataset.csv` has comma-separated skills in the `required_skills` column — the skill splitter must use `","` not `" "`.
- For the 1-page PDF export: the formatting preservation + 1-page constraint is a hard requirement. Use the backend approach (reportlab/fpdf2) for reliability.
- Use `asyncio` generators for the SSE streaming endpoint so the UI gets real-time updates.
- CORS must be open (`allow_origins=["*"]`) for local development.
- All frontend code goes in a single `frontend/index.html` — no separate CSS or JS files.
- Do not use React or any build tools for the frontend — pure HTML/CSS/JS only.
