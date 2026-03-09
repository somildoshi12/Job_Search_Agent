# AI Job Search Agent

An autonomous AI agent that filters, ranks, and tailors your resume for the best-matching jobs — using a local LLM (Llama-3.2 via Ollama). No data leaves your machine.

---

## Quick Start (From Scratch)

Follow these steps in order — takes about 10 minutes on first run (mostly model download time).

### Step 1 — Install Ollama

```bash
brew install ollama
```

Verify it installed:

```bash
ollama --version
```

### Step 2 — Download the LLM (~5 GB, one-time)

```bash
ollama pull llama3.2
```

This downloads the model. Grab a coffee — it's a 5 GB file.

### Step 3 — Clone / navigate to the project

```bash
cd "/Users/somildoshi/Somil Doshi/Projects/Job_Search_Agent"
```

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+. Check with `python3 --version`.

### Step 5 — Verify everything works (no Ollama needed)

```bash
python3 main.py --dry-run
```

You should see the filter + rank output in your terminal with no errors. If this works, your Python environment is good.

### Step 6 — Run the tests

```bash
python3 -m pytest tests/test_agent.py -v
```

All 11 tests should pass.

---

### Option A — CLI only (fastest)

Open **two terminals**:

**Terminal 1 — start Ollama:**

```bash
ollama serve
```

Leave this running.

**Terminal 2 — run the agent:**

```bash
python3 main.py
```

The agent will filter → rank → tailor your resume and print the final report.
To customize your profile, edit `CANDIDATE_PROFILE` at the bottom of `main.py`.

---

### Option B — Full web UI (recommended)

Open **three terminals**:

**Terminal 1 — start Ollama:**

```bash
ollama serve
```

**Terminal 2 — start the API server:**

```bash
cd "/Users/somildoshi/Somil Doshi/Projects/Job_Search_Agent"
uvicorn api_server:app --reload --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Terminal 3 — (optional) verify the API:**

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","model":"llama3.2","ollama_connected":true}
```

**Browser — open the UI:**

Double-click `frontend/index.html` or drag it into Chrome/Safari. The status dot in the top-right should turn **green** (Ollama connected).

Then:

1. Fill in your profile (skills are pre-filled with a sample set)
2. Optionally drag-drop your resume PDF to auto-extract summary + bullets
3. Click **⚡ Run Agent**
4. Watch the live terminal trace as the agent filters, ranks, and tailors
5. Download the tailored PDF when done

---

## Architecture

```
Candidate Profile
      │
      ▼
Llama 3.2       ──▶  filter_jobs  ──▶  rank_jobs  ──▶  tailor_resume
  (ReAct loop)        Rule-based      Scored 0–100     LLM rewrites
                      filtering         ranking        summary + bullets
                                                              │
                                              ┌───────────────┴──────────────┐
                                              ▼                              ▼
                                          CLI Report               API + Web UI
```

**Stack:**

| File                  | Purpose                                       |
| --------------------- | --------------------------------------------- |
| `main.py`             | CLI agent — ReAct loop, 3 tools, dry-run mode |
| `api_server.py`       | FastAPI backend with SSE streaming            |
| `frontend/index.html` | Self-contained interactive UI (no build step) |
| `tests/test_agent.py` | 11 pytest unit tests                          |
| `jobs_dataset.csv`    | 25 AI/ML job postings                         |

---

## Model Choice: Why `llama3.2`?

| Model           | RAM     | Speed (M4) | Reasoning              |
| --------------- | ------- | ---------- | ---------------------- |
| **llama3.2** ✓  | ~2.0 GB | ~25 tok/s  | ★★★★☆                  |
| llama3.2:3b     | ~2.5 GB | ~30 tok/s  | ★★★☆☆                  |
| llama3.1:8b     | ~5.5 GB | ~15 tok/s  | ★★★★☆                  |
| deepseek-r1:14b | ~9 GB   | ~8 tok/s   | ★★★★★ (tight on 16 GB) |

`llama3.2` is a lightweight model that runs quickly locally. The code also strips `<think>` blocks automatically for display and JSON parsing, in case reasoning models are used.

---

## Prerequisites

### 1. Install Ollama

```bash
brew install ollama
# or: curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the model (~5 GB)

```bash
ollama pull llama3.2
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Usage

### Option A — CLI dry run (no Ollama needed)

Runs filter + rank instantly without calling the LLM. Good for testing skill/location logic.

```bash
python main.py --dry-run
```

### Option B — Full CLI agent run

```bash
# Start Ollama first (separate terminal)
ollama serve

# Then run the agent
python main.py

# Optional: increase timeout for slow machines
python main.py --timeout 300
```

Edit `CANDIDATE_PROFILE` near the bottom of `main.py` to change your profile.

### Option C — Full-stack web UI

**Terminal 1:**

```bash
ollama serve
```

**Terminal 2:**

```bash
uvicorn api_server:app --reload --port 8000
```

**Browser:** Open `frontend/index.html` directly (double-click or drag into browser). No local server needed for the frontend.

---

## Project Structure

```
Job_Search_Agent/
├── main.py                    # CLI agent (ReAct loop + 3 tools)
├── api_server.py              # FastAPI backend (SSE streaming)
├── jobs_dataset.csv           # 25 AI/ML job postings (comma-sep skills)
├── requirements.txt
├── README.md
├── CLAUDE_CODE_PLAN.md        # Development plan + progress tracker
├── frontend/
│   └── index.html             # Self-contained UI (no build step)
├── tests/
│   └── test_agent.py          # 11 pytest tests
└── Assignment Requirement/
    ├── Group Assignment - Job Search Agent.pdf
    └── AI_Agent_Assignment_2 (1).pdf
```

---

## API Reference

| Method | Path             | Description                        |
| ------ | ---------------- | ---------------------------------- |
| GET    | `/health`        | Ollama connectivity check          |
| GET    | `/jobs`          | All 25 job postings as JSON        |
| POST   | `/run-agent`     | SSE stream: filter → rank → tailor |
| POST   | `/parse-resume`  | Extract text from PDF upload       |
| POST   | `/export-resume` | Generate 1-page tailored PDF       |

**Health check:**

```bash
curl http://localhost:8000/health
```

**Run agent (SSE):**

```bash
curl -N -X POST http://localhost:8000/run-agent \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Somil",
    "skills": ["Python","SQL","Machine Learning","PyTorch"],
    "years_experience": 2,
    "preferred_location": "Remote",
    "current_summary": "Data scientist with 2 years of ML experience...",
    "bullet_1": "Built ETL pipeline reducing latency by 40%...",
    "bullet_2": "Deployed NLP model achieving 90% accuracy..."
  }'
```

**Parse resume PDF:**

```bash
curl -F "file=@resume.pdf" http://localhost:8000/parse-resume
```

---

## Running Tests

```bash
pytest tests/test_agent.py -v
```

All 11 tests should pass. Coverage:

| Test                                               | What it checks                                          |
| -------------------------------------------------- | ------------------------------------------------------- |
| `test_filtering_location_remote_only`              | Only remote jobs returned with `remote_only=True`       |
| `test_filtering_experience_cap`                    | No jobs requiring >3 years for a 2-year candidate       |
| `test_filtering_company_exclusion`                 | Excluded company absent from results                    |
| `test_filtering_skill_overlap`                     | Only Python jobs returned when Python is the only skill |
| `test_ranking_score_range`                         | All scores 0–100                                        |
| `test_ranking_top_job_highest_score`               | First job has highest score                             |
| `test_ranking_skill_proportional`                  | 5/5 match beats 1/5 match                               |
| `test_ranking_top3_count`                          | `top_3` has ≤ 3 items                                   |
| `test_dataset_integrity`                           | 25 rows, 7 columns, valid years, no empty fields        |
| `test_state_machine_blocks_finish`                 | `finish` rejected when steps incomplete                 |
| `test_state_machine_allows_finish_after_all_steps` | `finish` accepted after all 3 steps                     |

---

## Candidate Profile (CLI)

Edit the dict at the bottom of `main.py`:

```python
CANDIDATE_PROFILE = {
    "name": "Your Name",
    "skills": ["Python", "SQL", "Machine Learning", ...],
    "years_experience": 2,
    "preferred_location": "Remote",  # or "San Francisco", "New York", etc.
    "remote_only": False,
    "exclude_companies": [],          # e.g. ["Meta", "TikTok"]
    "current_summary": "...",
    "bullet_1": "...",
    "bullet_2": "...",
}
```

---

## Scoring Breakdown

| Dimension      | Max Points | Calculation                                  |
| -------------- | ---------- | -------------------------------------------- |
| Skill Match    | 50         | `(matched_skills / total_job_skills) × 50`   |
| Experience Fit | 30         | 30 (exact), 22 (±1yr), 12 (±2yr), ≤5 (±3yr+) |
| Location Match | 20         | 20 (city match), 15 (remote), 5 (other)      |

---

## Troubleshooting

| Symptom                       | Fix                                          |
| ----------------------------- | -------------------------------------------- |
| `ConnectionError` on start    | Run `ollama serve` in a separate terminal    |
| `model not found`             | Run `ollama pull llama3.2`                   |
| Agent loops without finishing | Increase `MAX_AGENT_ITERATIONS` in `main.py` |
| Slow responses                | Normal on CPU; M4 Metal gives ~15 tok/s      |
| Frontend shows "API offline"  | Start: `uvicorn api_server:app --port 8000`  |
| PDF export fails              | Run: `pip install reportlab pdfplumber`      |
| Timeout on resume tailoring   | Use: `python main.py --timeout 300`          |
