"""
=============================================================================
AI Job Search Agent - Single LLM-Based Agent with Tool Calling
=============================================================================
Uses Ollama (local LLM) with Qwen 3.5 4B for reasoning and tool dispatch.
Tools: FilteringTool, RankingTool, ResumeTailoringTool

Architecture:
  Candidate Profile → Agent Loop (LLM Reasoning) → Tool Calls → Output
=============================================================================
"""

from __future__ import annotations

import json
import csv
import os
import re
import sys
import argparse
import textwrap
from typing import Any
from pathlib import Path
import requests  # pip install requests

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:4b"
DATASET_PATH = "jobs_dataset_with_real_urls.csv"
MAX_AGENT_ITERATIONS = 6   # Safety cap on the ReAct loop
OLLAMA_TIMEOUT = 120       # Default timeout; overridden per-call for heavy tools


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

    # Rule 4 – Skill overlap (CSV uses comma-separated skills)
    skill_before = len(filtered)
    candidate_skills_lower = {s.strip().lower() for s in required_skills}
    def has_skill_overlap(job):
        job_skills_lower = {s.strip().lower() for s in job["required_skills"].split(",")}
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
        # CSV uses comma-separated skills
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

def _parse_tailoring_json(response_text: str) -> dict | None:
    """
    Robustly extract the tailoring JSON from an LLM response.
    Cascade:
      1. Strip ```json ... ``` fences, then try json.loads()
      2. Try json.loads() on full text
      3. Regex search for JSON with required key
      4. Line-by-line manual extraction
    """
    # Step 1: strip markdown fences
    stripped = re.sub(r"```(?:json)?\s*", "", response_text)
    stripped = re.sub(r"```", "", stripped).strip()
    try:
        obj = json.loads(stripped)
        if "professional_summary" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 2: try full raw text
    try:
        obj = json.loads(response_text)
        if "professional_summary" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 3: regex for JSON block
    match = re.search(r'\{[\s\S]*?"professional_summary"[\s\S]*?\}', response_text)
    if match:
        try:
            obj = json.loads(match.group())
            if "professional_summary" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Step 4: line-by-line manual extraction
    result = {}
    current_key = None
    buffer = []
    key_map = {
        "professional_summary": "professional_summary",
        "bullet_1_rewritten": "bullet_1_rewritten",
        "bullet_2_rewritten": "bullet_2_rewritten",
    }
    for line in response_text.splitlines():
        for key in key_map:
            if f'"{key}"' in line or f"'{key}'" in line:
                if current_key and buffer:
                    result[current_key] = " ".join(buffer).strip().strip('"').strip("'")
                current_key = key
                buffer = []
                # Try to capture inline value
                m = re.search(r'["\']' + key + r'["\']:\s*["\'](.+)', line)
                if m:
                    buffer.append(m.group(1).rstrip('",'))
                break
        else:
            if current_key:
                buffer.append(line.strip().strip('"').strip(','))
    if current_key and buffer:
        result[current_key] = " ".join(buffer).strip()

    if "professional_summary" in result:
        return result

    return None


def resume_tailoring_tool(
    job: dict,
    candidate_profile: dict,
    llm_call_fn,
    timeout: int = 300,
) -> dict:
    """
    Uses the LLM to rewrite:
      1. Professional Summary (tailored to top job)
      2. Two experience bullet points (from candidate's current resume)
    Does NOT regenerate the entire resume.
    """
    bullet_1 = candidate_profile.get("bullet_1", "Developed Python ETL pipelines processing 10M+ records daily.")
    bullet_2 = candidate_profile.get("bullet_2", "Built SQL dashboards for operational KPIs and business reporting.")

    prompt = f"""You are an expert resume writer. Your task is to make MINIMAL, targeted edits to existing resume content to better align it with a specific job posting.

=== TARGET JOB ===
Title: {job['job_title']}
Company: {job['company']}
Required Skills: {job['required_skills']}
Description: {job['job_description']}

=== EXISTING RESUME CONTENT (keep structure, improve alignment) ===
Current Summary: {candidate_profile.get('current_summary', 'Data professional with experience in Python and analytics.')}
Bullet 1: {bullet_1}
Bullet 2: {bullet_2}

=== STRICT RULES ===
- Keep the same sentence structure and voice as the originals
- Only swap in keywords/technologies from the job description that the candidate already uses
- Keep all existing metrics (percentages, numbers) — you may add ONE metric if clearly supported
- Do NOT invent new projects, tools, or accomplishments
- Each bullet must remain a single sentence starting with a strong past-tense action verb
- The summary must stay 2-3 sentences, tight and specific to this role
- Do NOT rewrite from scratch — edit surgically

Return ONLY valid JSON (no markdown, no extra text):
{{
  "professional_summary": "...",
  "bullet_1_rewritten": "...",
  "bullet_2_rewritten": "..."
}}"""

    response_text = llm_call_fn(prompt, timeout=timeout)

    # Strip <think> blocks before parsing (in case of reasoning models)
    clean_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()

    parsed = _parse_tailoring_json(clean_text)
    if parsed:
        return {"success": True, "tailored": parsed, "job_applied_to": job["job_title"]}

    # Fallback: return raw text if all parsing fails
    return {
        "success": False,
        "raw_output": response_text,
        "job_applied_to": job["job_title"],
    }

# ─────────────────────────────────────────────────────────────────────────────
# LLM INTERFACE — Ollama
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(messages: list[dict], temperature: float = 0.3, timeout: int = OLLAMA_TIMEOUT) -> str:
    """Send messages to Ollama and return the assistant reply string."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
        "think": False,  # Disable Qwen3.5 chain-of-thought for speed
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"{C.RED}✗ Cannot reach Ollama at {OLLAMA_URL}. Is it running?{C.RESET}")
        print(f"  Run: {C.BOLD}ollama serve{C.RESET}  in a separate terminal.")
        sys.exit(1)

def call_ollama_simple(prompt: str, timeout: int = OLLAMA_TIMEOUT) -> str:
    """Convenience wrapper for a single-turn call."""
    return call_ollama([{"role": "user", "content": prompt}], timeout=timeout)

def get_llm_caller(provider: str):
    """Return (multi_turn_fn, simple_fn) for the given LLM provider."""
    return call_ollama, call_ollama_simple

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
    def __init__(self, dataset_path: str, provider: str = "llama"):
        self.all_jobs = load_dataset(dataset_path)
        self.filtered_jobs: list[dict] = []
        self.ranked_result: dict = {}
        self.tailored_result: dict = {}
        self.conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.candidate_profile: dict = {}
        self.provider = provider
        self._llm_fn, self._llm_simple_fn = get_llm_caller(provider)
        # State machine: all three must be True before finish is accepted
        self._state = {"filtered": False, "ranked": False, "tailored": False}
        self._consecutive_no_tool = 0  # Track iterations without a tool call

    def _add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})

    def _extract_tool_call(self, text: str) -> dict | None:
        """
        Parse a tool call from LLM output. Handles <think> tags if present.

        Strategy:
          1. Strip <think>...</think> blocks first
          2. Try ```tool_call ... ``` block
          3. Try bare JSON with "tool" key
          4. Return None if nothing found
        """
        # Strip <think> blocks before searching for tool calls
        clean = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        # Pattern 1: ```tool_call ... ```
        match = re.search(r"```tool_call\s*(\{[\s\S]*?\})\s*```", clean)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Pattern 2: bare JSON anywhere in the cleaned text
        for m in re.finditer(r'\{', clean):
            try:
                obj = json.loads(clean[m.start():])
                if "tool" in obj:
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue

        # Pattern 3: try in raw text (in case think tags are malformed)
        match = re.search(r"```tool_call\s*(\{[\s\S]*?\})\s*```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _dispatch_tool(self, tool_call: dict, profile: dict, timeout: int = OLLAMA_TIMEOUT) -> str:
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
            self._state["filtered"] = True
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
            self._state["ranked"] = True
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
            tailor_timeout = 300
            result = resume_tailoring_tool(
                job=best_job,
                candidate_profile=profile,
                llm_call_fn=self._llm_simple_fn,
                timeout=tailor_timeout,
            )
            self.tailored_result = result
            self._state["tailored"] = True
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
            # State machine: block finish if required steps not done
            missing = [step for step, done in self._state.items() if not done]
            if missing:
                return (
                    f"BLOCKED: Cannot finish yet. Missing steps: {missing}. "
                    f"Please complete these steps first."
                )
            return f"AGENT COMPLETE: {params.get('summary', 'Task finished.')}"

        else:
            return f"ERROR: Unknown tool '{tool}'"

    def run(self, candidate_profile: dict, timeout: int = OLLAMA_TIMEOUT):
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
            llm_response = self._llm_fn(self.conversation, temperature=0.2, timeout=timeout)
            self._add_message("assistant", llm_response)

            # Strip <think>...</think> blocks for display (if any)
            display_response = re.sub(r"<think>[\s\S]*?</think>", "", llm_response).strip()
            print(f"{C.CYAN}Agent Reasoning & Action:{C.RESET}")
            print(display_response)

            # Step 2: Extract tool call
            tool_call = self._extract_tool_call(llm_response)

            if tool_call is None:
                self._consecutive_no_tool += 1
                print(f"{C.DIM}No tool call detected in response (consecutive: {self._consecutive_no_tool}).{C.RESET}")
                if self._consecutive_no_tool >= 2:
                    # Inject reminder to get the agent back on track
                    missing = [step for step, done in self._state.items() if not done]
                    reminder = (
                        f"You must output a tool call in ```tool_call ... ``` format. "
                        f"Remaining steps to complete: {missing}. "
                        f"Please call the next required tool now."
                    )
                    self._add_message("user", reminder)
                    self._consecutive_no_tool = 0
                continue
            else:
                self._consecutive_no_tool = 0

            tool_name = tool_call.get("tool", "unknown")
            tool_reason = tool_call.get("reason", "")
            print_trace(f"Tool Selected: {tool_name}", tool_reason)

            if tool_name == "finish":
                # Check state machine before accepting finish
                observation = self._dispatch_tool(tool_call, candidate_profile, timeout=timeout)
                if observation.startswith("BLOCKED"):
                    print(f"{C.YELLOW}{observation}{C.RESET}")
                    self._add_message("user", f"Tool observation:\n{observation}\n\nPlease complete the missing steps.")
                    continue
                print(f"\n{C.GREEN}{C.BOLD}✓ Agent signalled completion.{C.RESET}")
                break

            # Step 3: Execute tool
            print_section(f"TOOL EXECUTION: {tool_name.upper()}", C.GREEN)
            observation = self._dispatch_tool(tool_call, candidate_profile, timeout=timeout)
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
# DRY RUN — runs filter + rank without Ollama
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(candidate_profile: dict, dataset_path: str):
    """Run filtering and ranking without calling Ollama. Useful for testing."""
    print_section("🧪  DRY RUN MODE (no LLM calls)", C.YELLOW)
    print(f"Candidate: {candidate_profile.get('name', 'Candidate')}")
    print(f"Skills   : {', '.join(candidate_profile['skills'])}")
    print(f"Exp      : {candidate_profile['years_experience']} years")
    print(f"Location : {candidate_profile['preferred_location']}\n")

    jobs = load_dataset(dataset_path)

    # Filter
    print_section("TOOL: filter_jobs", C.CYAN)
    filter_result = filtering_tool(
        jobs=jobs,
        preferred_location=candidate_profile["preferred_location"],
        max_years_experience=candidate_profile["years_experience"],
        required_skills=candidate_profile["skills"],
        exclude_companies=candidate_profile.get("exclude_companies"),
        remote_only=candidate_profile.get("remote_only", False),
    )
    for line in filter_result["reasoning_trace"]:
        print(f"  {line}")
    print(f"\n{C.GREEN}Filtered: {filter_result['count']} jobs{C.RESET}")

    # Rank
    print_section("TOOL: rank_jobs", C.CYAN)
    rank_result = ranking_tool(
        jobs=filter_result["filtered_jobs"],
        candidate_skills=candidate_profile["skills"],
        candidate_years=candidate_profile["years_experience"],
        preferred_location=candidate_profile["preferred_location"],
    )

    print(f"{'Rank':<5} {'Job Title':<35} {'Company':<22} {'Score':>6}")
    print("─" * 75)
    for i, j in enumerate(rank_result["ranked_jobs"][:10]):
        print(f"#{i+1:<4} {j['job_title']:<35} {j['company']:<22} {j['score']:>6.1f}")

    best = rank_result["best_job"]
    if best:
        print(f"\n{C.GREEN}{C.BOLD}Best Match: {best['job_title']} @ {best['company']} (Score: {best['score']}){C.RESET}")

    print_section("DRY RUN COMPLETE", C.GREEN)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  — Edit your candidate profile here
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AI Job Search Agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run filtering + ranking without calling any LLM",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=OLLAMA_TIMEOUT,
        help=f"Request timeout in seconds for Ollama (default: {OLLAMA_TIMEOUT})",
    )
    args = parser.parse_args()

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

    if args.dry_run:
        dry_run(CANDIDATE_PROFILE, DATASET_PATH)
    else:
        print(f"{C.CYAN}Using LLM provider: {C.BOLD}LLAMA{C.RESET}")
        agent = JobSearchAgent(dataset_path=DATASET_PATH, provider="llama")
        agent.run(CANDIDATE_PROFILE, timeout=args.timeout)
