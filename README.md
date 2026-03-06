# 🤖 AI Job Search Agent — Setup & Run Guide

## Model Choice: Why `deepseek-r1:8b`?

| Model | RAM Usage | Reasoning | Speed (M4) | Verdict |
|---|---|---|---|---|
| `deepseek-r1:8b` | ~5.5 GB | ⭐⭐⭐⭐⭐ Chain-of-thought | ~15 tok/s | ✅ **Recommended** |
| `llama3.2:3b` | ~2.5 GB | ⭐⭐⭐ | ~35 tok/s | Good fallback |
| `mistral:7b` | ~5 GB | ⭐⭐⭐⭐ | ~18 tok/s | Also works |
| `deepseek-r1:14b` | ~9 GB | ⭐⭐⭐⭐⭐ | ~8 tok/s | Tight on 16GB |

**DeepSeek-R1:8b is the right choice for your M4 MacBook Air 16 GB** because:
- It uses its internal `<think>` chain-of-thought reasoning (perfect for agent decision-making)
- ~5.5 GB VRAM/RAM usage leaves plenty for macOS and the Python process
- Strong instruction following for structured JSON tool-call outputs
- Runs fully on the M4 Neural Engine via Ollama's Metal backend

---

## Prerequisites

### 1. Install Ollama
```bash
# Option A — Homebrew
brew install ollama

# Option B — Direct installer
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the model (~5 GB download)
```bash
ollama pull deepseek-r1:8b
```

### 3. Start Ollama server (keep this terminal open)
```bash
ollama serve
```

### 4. Install Python dependencies
```bash
pip install requests
# OR
pip install -r requirements.txt
```

---

## Project Structure

```
job_search_agent/
├── main.py              ← Single LLM-based agent (all logic here)
├── jobs_dataset.csv     ← 25 real AI/ML job postings
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## Running the Agent

```bash
# Make sure ollama serve is running in another terminal, then:
python main.py
```

### Expected Output Flow:
1. Dataset loads (25 jobs)
2. Agent Iteration 1 → LLM reasons → calls `filter_jobs`
3. Agent Iteration 2 → LLM sees filtered results → calls `rank_jobs`
4. Agent Iteration 3 → LLM sees top 3 → calls `tailor_resume`
5. Agent Iteration 4 → LLM calls `finish`
6. Final report printed (ranked table + tailored resume)

---

## Customising the Candidate Profile

Edit the `CANDIDATE_PROFILE` dict at the bottom of `main.py`:

```python
CANDIDATE_PROFILE = {
    "name": "Your Name",
    "skills": ["Python", "SQL", "Machine Learning", ...],
    "years_experience": 2,
    "preferred_location": "Remote",   # or "San Francisco", "New York", etc.
    "remote_only": False,
    "exclude_companies": ["Meta"],    # companies to exclude
    "current_summary": "Your current resume summary...",
    "bullet_1": "Your first experience bullet...",
    "bullet_2": "Your second experience bullet...",
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CANDIDATE PROFILE INPUT                   │
│         (skills, location, years_experience, resume)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  DeepSeek-R1:8B AGENT LOOP                  │
│                  (ReAct: Reason → Act → Observe)             │
│                                                             │
│   System Prompt → LLM Reasoning → Tool Call Decision        │
│         ↑                                    │               │
│         └────── Observation Feedback ←───────┘               │
└─────────────────────────────────────────────────────────────┘
           │              │              │
           ▼              ▼              ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐
   │ TOOL 1       │ │ TOOL 2       │ │ TOOL 3               │
   │ filter_jobs  │ │ rank_jobs    │ │ tailor_resume        │
   │              │ │              │ │                      │
   │ Rule-based:  │ │ Score 0-100: │ │ LLM rewrites:        │
   │ • Location   │ │ • Skill 50pt │ │ • Prof. Summary      │
   │ • Exp cap    │ │ • Exp   30pt │ │ • Bullet 1           │
   │ • Skill ∩    │ │ • Loc   20pt │ │ • Bullet 2           │
   │ • Exclusions │ │ → Top 3 jobs │ │                      │
   └──────────────┘ └──────────────┘ └──────────────────────┘
                                              │
                                              ▼
                               ┌──────────────────────────┐
                               │     FINAL REPORT         │
                               │  • Filtered job count    │
                               │  • Ranked list w/ scores │
                               │  • Top 3 with breakdown  │
                               │  • Tailored resume       │
                               │  • Reasoning traces      │
                               └──────────────────────────┘
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ConnectionError` to Ollama | Run `ollama serve` in a separate terminal |
| Model not found | Run `ollama pull deepseek-r1:8b` |
| JSON parse errors | DeepSeek-R1 wraps output in `<think>` tags — the agent strips them automatically |
| Agent loops without finishing | Increase `MAX_AGENT_ITERATIONS` in main.py |
| Slow responses | Normal for 8B on CPU. M4 Metal acceleration should give ~15 tok/s |
