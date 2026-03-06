"""
=============================================================================
AI Job Search Agent - Single LLM-Based Agent with Tool Calling
=============================================================================
Uses Ollama (local LLM) with DeepSeek-R1:8B for reasoning and tool dispatch.
Tools: FilteringTool, RankingTool, ResumeTailoringTool

Architecture:
  Candidate Profile → Agent Loop (LLM Reasoning) → Tool Calls → Output
=============================================================================
"""

import json
import csv
import re
import sys
import textwrap
from typing import Any
import requests  # pip install requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
# DeepSeek-R1 8B is a strong reasoning model that fits comfortably on M4 16 GB.
# It uses ~5-6 GB RAM leaving plenty for the OS and tools.
MODEL_NAME = "deepseek-r1:8b"
DATASET_PATH = "jobs_dataset.csv"
MAX_AGENT_ITERATIONS = 6   # Safety cap on the ReAct loop

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS  (works on macOS terminal)
# ─────────────────────────────────────────────────────────────────────────────

class C:
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

def print_section(title: str, colour: str = C.CYAN):
    width = 70
    print(f"\n{colour}{C.BOLD}{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}{C.RESET}\n")

def print_trace(label: str, content: str):
    print(f"{C.DIM}[AGENT TRACE] {label}:{C.RESET}")
    for line in textwrap.wrap(content, width=80):
        print(f"  {C.DIM}{line}{C.RESET}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    """Load job postings CSV into a list of dicts."""
    jobs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["years_experience"] = int(row.get("years_experience", 0))
            jobs.append(dict(row))
    print(f"{C.GREEN}✓ Loaded {len(jobs)} job postings from '{path}'{C.RESET}")
    return jobs

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — FILTERING TOOL
# ─────────────────────────────────────────────────────────────────────────────

def filtering_tool(
    jobs: list[dict],
    preferred_location: str,
    max_years_experience: int,
    required_skills: list[str],
    exclude_companies: list[str] | None = None,
    remote_only: bool = False,
) -> dict:
    """
    Rule-based filtering. Returns filtered job list and a reasoning trace.

    Rules applied (in order):
      1. Remote-only filter (optional)
      2. Location preference match (city / state / "Remote")
      3. Experience cap  — drop jobs requiring MORE than candidate has
      4. Skill overlap   — keep jobs sharing ≥1 skill with candidate
      5. Company exclusion list
    """
    trace = []
    filtered = list(jobs)
    original_count = len(filtered)

    # Rule 1 – Remote-only
    if remote_only:
        filtered = [j for j in filtered if "remote" in j["location"].lower()]
        trace.append(f"Remote-only filter: {original_count} → {len(filtered)} jobs")

    # Rule 2 – Location preference
    loc_before = len(filtered)
    if preferred_location.lower() != "any":
        location_terms = preferred_location.lower().split()
        filtered = [
            j for j in filtered
            if any(t in j["location"].lower() for t in location_terms)
            or "remote" in j["location"].lower()
        ]
        trace.append(f"Location filter ('{preferred_location}'): {loc_before} → {len(filtered)} jobs")

    # Rule 3 – Experience cap
    exp_before = len(filtered)
    filtered = [j for j in filtered if j["years_experience"] <= max_years_experience + 1]
    trace.append(f"Experience cap (≤{max_years_experience}+1 yrs): {exp_before} → {len(filtered)} jobs")

    # Rule 4 – Skill overlap
    skill_before = len(filtered)
    candidate_skills_lower = {s.strip().lower() for s in required_skills}
    def has_skill_overlap(job):
        job_skills_lower = {s.strip().lower() for s in job["required_skills"].split()}
        return len(candidate_skills_lower & job_skills_lower) > 0
    filtered = [j for j in filtered if has_skill_overlap(j)]
    trace.append(f"Skill overlap filter: {skill_before} → {len(filtered)} jobs")

    # Rule 5 – Company exclusions
    if exclude_companies:
        excl_before = len(filtered)
        excl_lower = {c.lower() for c in exclude_companies}
        filtered = [j for j in filtered if j["company"].lower() not in excl_lower]
        trace.append(f"Company exclusion {exclude_companies}: {excl_before} → {len(filtered)} jobs")

    return {
        "filtered_jobs": filtered,
        "count": len(filtered),
        "reasoning_trace": trace,
    }

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — RANKING TOOL
# ─────────────────────────────────────────────────────────────────────────────

def ranking_tool(
    jobs: list[dict],
    candidate_skills: list[str],
    candidate_years: int,
    preferred_location: str,
) -> dict:
    """
    Score each job 0-100 and return a ranked list with score breakdown.

    Scoring Breakdown:
      • Skill Match       : 0-50 pts  (proportional to % skills matched)
      • Experience Fit    : 0-30 pts  (closer to candidate's years = higher)
      • Location Match    : 0-20 pts  (exact city > state > remote > other)
    """
    candidate_skills_lower = [s.strip().lower() for s in candidate_skills]
    ranked = []

    for job in jobs:
        job_skills = [s.strip().lower() for s in job["required_skills"].split(",")]
        # --- Skill Match Score (50 pts) ---
        matched = sum(1 for cs in candidate_skills_lower
                      if any(cs in js or js in cs for js in job_skills))
        skill_score = round(min(matched / max(len(job_skills), 1), 1.0) * 50, 2)

        # --- Experience Fit Score (30 pts) ---
        diff = abs(job["years_experience"] - candidate_years)
        if diff == 0:
            exp_score = 30
        elif diff == 1:
            exp_score = 22
        elif diff == 2:
            exp_score = 12
        else:
            exp_score = max(0, 5 - diff)

        # --- Location Match Score (20 pts) ---
        job_loc = job["location"].lower()
        pref_loc = preferred_location.lower()
        if "remote" in job_loc:
            loc_score = 15
        elif pref_loc in job_loc or any(t in job_loc for t in pref_loc.split()):
            loc_score = 20
        else:
            loc_score = 5

        total = round(skill_score + exp_score + loc_score, 2)
        ranked.append({
            **job,
            "score": total,
            "score_breakdown": {
                "skill_match": skill_score,
                "experience_fit": exp_score,
                "location_match": loc_score,
            },
            "skills_matched": matched,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top3 = ranked[:3]
    return {
        "ranked_jobs": ranked,
        "top_3": top3,
        "best_job": ranked[0] if ranked else None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — RESUME TAILORING TOOL  (LLM-powered)
# ─────────────────────────────────────────────────────────────────────────────

def resume_tailoring_tool(
    job: dict,
    candidate_profile: dict,
    llm_call_fn,
) -> dict:
    """
    Uses the LLM to rewrite:
      1. Professional Summary (tailored to top job)
      2. Two experience bullet points (from candidate's current resume)
    Does NOT regenerate the entire resume.
    """
    bullet_1 = candidate_profile.get("bullet_1", "Developed Python ETL pipelines processing 10M+ records daily.")
    bullet_2 = candidate_profile.get("bullet_2", "Built SQL dashboards for operational KPIs and business reporting.")

    prompt = f"""You are an expert resume writer.

=== TARGET JOB ===
Title: {job['job_title']}
Company: {job['company']}
Required Skills: {job['required_skills']}
Description: {job['job_description']}

=== CANDIDATE PROFILE ===
Skills: {', '.join(candidate_profile['skills'])}
Experience: {candidate_profile['years_experience']} years
Current Summary: {candidate_profile.get('current_summary', 'Data professional with experience in Python and analytics.')}
Bullet 1: {bullet_1}
Bullet 2: {bullet_2}

=== TASK ===
1. Write a tailored Professional Summary (3-4 sentences) that highlights the candidate's alignment with this specific role.
2. Rewrite Bullet 1 to better align with the job requirements — keep it achievement-oriented and include metrics.
3. Rewrite Bullet 2 to better align with the job requirements — keep it achievement-oriented and include metrics.

Do NOT generate a full resume. Return ONLY valid JSON in this exact format:
{{
  "professional_summary": "...",
  "bullet_1_rewritten": "...",
  "bullet_2_rewritten": "..."
}}"""

    response_text = llm_call_fn(prompt)

    # Extract JSON from response (DeepSeek-R1 may include <think> tags)
    json_match = re.search(r'\{[\s\S]*"professional_summary"[\s\S]*\}', response_text)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {"success": True, "tailored": result, "job_applied_to": job["job_title"]}
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text if JSON parsing fails
    return {
        "success": False,
        "raw_output": response_text,
        "job_applied_to": job["job_title"],
    }

# ─────────────────────────────────────────────────────────────────────────────
# LLM INTERFACE — Ollama
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(messages: list[dict], temperature: float = 0.3) -> str:
    """Send messages to Ollama and return the assistant reply string."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"{C.RED}✗ Cannot reach Ollama at {OLLAMA_URL}. Is it running?{C.RESET}")
        print(f"  Run: {C.BOLD}ollama serve{C.RESET}  in a separate terminal.")
        sys.exit(1)

def call_ollama_simple(prompt: str) -> str:
    """Convenience wrapper for a single-turn call."""
    return call_ollama([{"role": "user", "content": prompt}])

# ─────────────────────────────────────────────────────────────────────────────
# AGENT SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent AI Job Search Agent. Your role is to autonomously help
a candidate find the best matching job and tailor their resume for it.

You have access to three tools. To use a tool, respond with a JSON block like:
```tool_call
{
  "tool": "<tool_name>",
  "reason": "<why you are calling this tool>",
  "params": { ... }
}
```

Available tools:
1. filter_jobs
   params: preferred_location (str), max_years_experience (int),
           required_skills (list[str]), exclude_companies (list[str] optional),
           remote_only (bool optional, default false)

2. rank_jobs
   params: (no params needed — operates on filtered results)

3. tailor_resume
   params: (no params needed — operates on the best ranked job)

4. finish
   params: summary (str) — call this when all tasks are complete

AGENT RULES:
- Always call filter_jobs FIRST.
- Always call rank_jobs SECOND after filtering.
- Always call tailor_resume THIRD on the top-ranked job.
- Show your reasoning before each tool call.
- After tailor_resume, call finish with a brief summary.
- Be concise in your reasoning traces.
- Never skip a step or combine steps.
"""

# ─────────────────────────────────────────────────────────────────────────────
# AGENT LOOP  (ReAct-style: Reason → Act → Observe → Repeat)
# ─────────────────────────────────────────────────────────────────────────────

class JobSearchAgent:
    def __init__(self, dataset_path: str):
        self.all_jobs = load_dataset(dataset_path)
        self.filtered_jobs: list[dict] = []
        self.ranked_result: dict = {}
        self.tailored_result: dict = {}
        self.conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.candidate_profile: dict = {}

    def _add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})

    def _extract_tool_call(self, text: str) -> dict | None:
        """Parse a ```tool_call ... ``` block from LLM output."""
        match = re.search(r"```tool_call\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Fallback: try bare JSON with "tool" key
        match = re.search(r'\{\s*"tool"\s*:', text)
        if match:
            try:
                obj = json.loads(text[match.start():])
                if "tool" in obj:
                    return obj
            except Exception:
                pass
        return None

    def _dispatch_tool(self, tool_call: dict, profile: dict) -> str:
        """Execute the tool identified in tool_call and return an observation string."""
        tool = tool_call.get("tool")
        params = tool_call.get("params", {})

        if tool == "filter_jobs":
            result = filtering_tool(
                jobs=self.all_jobs,
                preferred_location=params.get("preferred_location", profile["preferred_location"]),
                max_years_experience=params.get("max_years_experience", profile["years_experience"]),
                required_skills=params.get("required_skills", profile["skills"]),
                exclude_companies=params.get("exclude_companies"),
                remote_only=params.get("remote_only", False),
            )
            self.filtered_jobs = result["filtered_jobs"]
            summary = "\n".join(result["reasoning_trace"])
            job_list = "\n".join(
                f"  {i+1}. {j['job_title']} @ {j['company']} ({j['location']}) — req. {j['years_experience']}yr"
                for i, j in enumerate(self.filtered_jobs)
            )
            return f"FILTER RESULT: {result['count']} jobs passed.\nTrace:\n{summary}\n\nFiltered Jobs:\n{job_list}"

        elif tool == "rank_jobs":
            if not self.filtered_jobs:
                return "ERROR: No filtered jobs available. Call filter_jobs first."
            result = ranking_tool(
                jobs=self.filtered_jobs,
                candidate_skills=profile["skills"],
                candidate_years=profile["years_experience"],
                preferred_location=profile["preferred_location"],
            )
            self.ranked_result = result
            top3_str = ""
            for i, j in enumerate(result["top_3"]):
                bd = j["score_breakdown"]
                top3_str += (
                    f"\n  #{i+1}  {j['job_title']} @ {j['company']}\n"
                    f"       Score: {j['score']}/100 "
                    f"(Skill={bd['skill_match']}, Exp={bd['experience_fit']}, Loc={bd['location_match']})\n"
                    f"       Skills matched: {j['skills_matched']}\n"
                )
            return f"RANK RESULT — Top 3 Jobs:{top3_str}\nBEST JOB: {result['best_job']['job_title']} @ {result['best_job']['company']} (Score: {result['best_job']['score']})"

        elif tool == "tailor_resume":
            if not self.ranked_result:
                return "ERROR: No ranked result available. Call rank_jobs first."
            best_job = self.ranked_result["best_job"]
            result = resume_tailoring_tool(
                job=best_job,
                candidate_profile=profile,
                llm_call_fn=call_ollama_simple,
            )
            self.tailored_result = result
            if result["success"]:
                t = result["tailored"]
                return (
                    f"RESUME TAILORING RESULT for '{result['job_applied_to']}':\n\n"
                    f"Professional Summary:\n{t['professional_summary']}\n\n"
                    f"Bullet 1 (rewritten):\n{t['bullet_1_rewritten']}\n\n"
                    f"Bullet 2 (rewritten):\n{t['bullet_2_rewritten']}"
                )
            else:
                return f"Resume tailoring produced raw output:\n{result.get('raw_output', 'N/A')}"

        elif tool == "finish":
            return f"AGENT COMPLETE: {params.get('summary', 'Task finished.')}"

        else:
            return f"ERROR: Unknown tool '{tool}'"

    def run(self, candidate_profile: dict):
        """Main agent execution loop."""
        self.candidate_profile = candidate_profile

        print_section("🤖  JOB SEARCH AGENT — STARTING", C.CYAN)
        print(f"Candidate: {candidate_profile.get('name', 'Candidate')}")
        print(f"Skills   : {', '.join(candidate_profile['skills'])}")
        print(f"Exp      : {candidate_profile['years_experience']} years")
        print(f"Location : {candidate_profile['preferred_location']}")

        # First message to the agent: describe the task
        initial_msg = f"""
I need you to find the best job for this candidate and tailor their resume.

Candidate Profile:
- Name: {candidate_profile.get('name', 'Candidate')}
- Skills: {', '.join(candidate_profile['skills'])}
- Years of Experience: {candidate_profile['years_experience']}
- Preferred Location: {candidate_profile['preferred_location']}
- Exclude Companies: {candidate_profile.get('exclude_companies', [])}
- Remote Only: {candidate_profile.get('remote_only', False)}

Current Resume Summary: "{candidate_profile.get('current_summary', '')}"
Bullet 1: "{candidate_profile.get('bullet_1', '')}"
Bullet 2: "{candidate_profile.get('bullet_2', '')}"

Total jobs in dataset: {len(self.all_jobs)}

Please begin. Call filter_jobs first.
"""
        self._add_message("user", initial_msg)

        # ── ReAct Loop ──────────────────────────────────────────────────────
        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            print_section(f"AGENT ITERATION {iteration}", C.YELLOW)

            # Step 1: LLM thinks and (optionally) calls a tool
            llm_response = call_ollama(self.conversation, temperature=0.2)
            self._add_message("assistant", llm_response)

            # Strip <think>...</think> blocks from DeepSeek-R1 for display
            display_response = re.sub(r"<think>[\s\S]*?</think>", "", llm_response).strip()
            print(f"{C.CYAN}Agent Reasoning & Action:{C.RESET}")
            print(display_response)

            # Step 2: Extract tool call
            tool_call = self._extract_tool_call(llm_response)

            if tool_call is None:
                print(f"{C.DIM}No tool call detected in response. Ending loop.{C.RESET}")
                break

            tool_name = tool_call.get("tool", "unknown")
            tool_reason = tool_call.get("reason", "")
            print_trace(f"Tool Selected: {tool_name}", tool_reason)

            if tool_name == "finish":
                print(f"\n{C.GREEN}{C.BOLD}✓ Agent signalled completion.{C.RESET}")
                break

            # Step 3: Execute tool
            print_section(f"TOOL EXECUTION: {tool_name.upper()}", C.GREEN)
            observation = self._dispatch_tool(tool_call, candidate_profile)
            print(observation)

            # Step 4: Feed observation back to agent
            self._add_message("user", f"Tool observation:\n{observation}\n\nContinue with the next step.")

        # ── Final Output ──────────────────────────────────────────────────────
        self._print_final_report()

    def _print_final_report(self):
        print_section("📊  FINAL AGENT REPORT", C.GREEN)

        # Ranked jobs table
        if self.ranked_result:
            print(f"{C.BOLD}TOP 3 RANKED JOBS:{C.RESET}")
            print(f"{'Rank':<5} {'Job Title':<35} {'Company':<20} {'Score':>6}  Breakdown")
            print("─" * 90)
            for i, j in enumerate(self.ranked_result.get("top_3", [])):
                bd = j["score_breakdown"]
                print(
                    f"#{i+1:<4} {j['job_title']:<35} {j['company']:<20} {j['score']:>6.1f}  "
                    f"[Skill:{bd['skill_match']:.0f} Exp:{bd['experience_fit']:.0f} Loc:{bd['location_match']:.0f}]"
                )
            print()

        # Best job detail
        best = self.ranked_result.get("best_job")
        if best:
            print(f"{C.BOLD}SELECTED JOB:{C.RESET}")
            print(f"  Title   : {best['job_title']}")
            print(f"  Company : {best['company']}")
            print(f"  Location: {best['location']}")
            print(f"  Score   : {best['score']}/100")
            print(f"  URL     : {best['url']}")
            print()

        # Tailored resume
        if self.tailored_result and self.tailored_result.get("success"):
            t = self.tailored_result["tailored"]
            print(f"{C.BOLD}TAILORED RESUME SNIPPETS (for {self.tailored_result['job_applied_to']}):{C.RESET}")
            print()
            print(f"{C.CYAN}Professional Summary:{C.RESET}")
            print(textwrap.fill(t["professional_summary"], width=80, initial_indent="  "))
            print()
            print(f"{C.CYAN}Experience Bullet 1 (rewritten):{C.RESET}")
            print(textwrap.fill(t["bullet_1_rewritten"], width=80, initial_indent="  "))
            print()
            print(f"{C.CYAN}Experience Bullet 2 (rewritten):{C.RESET}")
            print(textwrap.fill(t["bullet_2_rewritten"], width=80, initial_indent="  "))
            print()

        print(f"{C.GREEN}{C.BOLD}═══════════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.GREEN}{C.BOLD}  Agent execution complete.{C.RESET}")
        print(f"{C.GREEN}{C.BOLD}═══════════════════════════════════════════════════════════════{C.RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  — Edit your candidate profile here
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CANDIDATE PROFILE — modify this section ──────────────────────────────
    CANDIDATE_PROFILE = {
        "name": "Somil",
        "skills": [
            "Python", "SQL", "R", "Machine Learning", "Deep Learning",
            "PyTorch", "TensorFlow", "NLP", "Transformers",
            "Databricks", "Airflow", "dbt", "PostgreSQL", "MongoDB",
            "Docker", "AWS", "Power BI", "Spark", "MLflow", "Pandas",
        ],
        "years_experience": 2,
        "preferred_location": "Remote",   # or "San Francisco" / "New York" / "Remote"
        "remote_only": False,              # True = only remote jobs
        "exclude_companies": [],           # e.g. ["Meta", "TikTok"]

        # Resume content to be tailored
        "current_summary": (
            "Data Science MS student at the University of Houston with 2+ years of "
            "hands-on experience building end-to-end ML pipelines, deep learning models, "
            "and data engineering solutions using Python, SQL, PyTorch, and cloud platforms."
        ),
        "bullet_1": (
            "Developed automated ETL pipelines using Python and Apache Airflow that reduced "
            "data processing time by 40% and ensured 99.9% uptime for downstream analytics."
        ),
        "bullet_2": (
            "Built and deployed a sentiment analysis deep learning model achieving 90%+ accuracy "
            "across five sentiment classes, integrated via Flask REST API for real-time inference."
        ),
    }
    # ─────────────────────────────────────────────────────────────────────────

    agent = JobSearchAgent(dataset_path=DATASET_PATH)
    agent.run(CANDIDATE_PROFILE)
