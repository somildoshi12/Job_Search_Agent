"""
=============================================================================
Unit Tests for AI Job Search Agent
=============================================================================
Run with: pytest tests/test_agent.py -v
=============================================================================
"""

import csv
import sys
import os
import pytest

# Add project root to path so we can import main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import filtering_tool, ranking_tool, JobSearchAgent, DATASET_PATH


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def make_job(
    title="ML Engineer",
    company="TestCo",
    location="Remote USA",
    skills="Python,SQL,machine learning",
    years=2,
    description="Test job.",
    url="https://example.com",
):
    return {
        "job_title": title,
        "company": company,
        "location": location,
        "required_skills": skills,
        "years_experience": years,
        "job_description": description,
        "url": url,
    }


SAMPLE_JOBS = [
    make_job(title="Remote ML Engineer",   company="Alpha",  location="Remote USA",       skills="Python,machine learning", years=2),
    make_job(title="Onsite Data Scientist", company="Beta",   location="San Francisco CA", skills="Python,SQL",              years=2),
    make_job(title="Remote Data Engineer",  company="Gamma",  location="Remote USA",       skills="SQL,Spark,Airflow",        years=1),
    make_job(title="Senior ML Researcher",  company="Google", location="Mountain View CA", skills="Python,TensorFlow,research", years=5),
    make_job(title="Junior Analyst",        company="Meta",   location="New York NY",      skills="SQL,Tableau,statistics",   years=0),
]


# ─────────────────────────────────────────────────────────────────────────────
# FILTERING TOOL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFilteringTool:

    def test_filtering_location_remote_only(self):
        """Test 1: remote_only=True returns only remote jobs."""
        result = filtering_tool(
            jobs=SAMPLE_JOBS,
            preferred_location="Remote",
            max_years_experience=5,
            required_skills=["Python"],
            remote_only=True,
        )
        jobs = result["filtered_jobs"]
        assert len(jobs) > 0, "Should return at least one remote job"
        for job in jobs:
            assert "remote" in job["location"].lower(), (
                f"Non-remote job found: {job['location']}"
            )

    def test_filtering_experience_cap(self):
        """Test 2: candidate with 2 years should not see jobs requiring >3 years."""
        result = filtering_tool(
            jobs=SAMPLE_JOBS,
            preferred_location="any",
            max_years_experience=2,
            required_skills=["Python"],
        )
        for job in result["filtered_jobs"]:
            assert job["years_experience"] <= 3, (
                f"Job requiring {job['years_experience']} years slipped through for a 2yr candidate"
            )

    def test_filtering_company_exclusion(self):
        """Test 3: excluded company should not appear in results."""
        result = filtering_tool(
            jobs=SAMPLE_JOBS,
            preferred_location="any",
            max_years_experience=10,
            required_skills=["Python", "SQL"],
            exclude_companies=["Google"],
        )
        companies = [j["company"] for j in result["filtered_jobs"]]
        assert "Google" not in companies, "Google should be excluded"

    def test_filtering_skill_overlap(self):
        """Test 4: only jobs sharing at least one skill with candidate are returned."""
        result = filtering_tool(
            jobs=SAMPLE_JOBS,
            preferred_location="any",
            max_years_experience=10,
            required_skills=["Python"],
        )
        for job in result["filtered_jobs"]:
            job_skills = {s.strip().lower() for s in job["required_skills"].split(",")}
            assert "python" in job_skills, (
                f"Job without Python returned: {job['job_title']} skills={job['required_skills']}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# RANKING TOOL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRankingTool:

    def _get_ranked(self, jobs=None, skills=None, years=2, location="Remote"):
        jobs = jobs or SAMPLE_JOBS
        skills = skills or ["Python", "SQL", "machine learning"]
        return ranking_tool(
            jobs=jobs,
            candidate_skills=skills,
            candidate_years=years,
            preferred_location=location,
        )

    def test_ranking_score_range(self):
        """Test 5: all scores must be 0–100."""
        result = self._get_ranked()
        for job in result["ranked_jobs"]:
            assert 0 <= job["score"] <= 100, (
                f"Score out of range: {job['score']} for {job['job_title']}"
            )

    def test_ranking_top_job_highest_score(self):
        """Test 6: the first item in ranked_jobs has the highest score."""
        result = self._get_ranked()
        ranked = result["ranked_jobs"]
        if len(ranked) > 1:
            assert ranked[0]["score"] >= ranked[1]["score"], (
                "First job does not have the highest score"
            )

    def test_ranking_skill_proportional(self):
        """Test 7: a job matching 5/5 candidate skills should outscore one matching 1/5."""
        high_match = make_job(
            title="High Match",
            skills="Python,SQL,machine learning,TensorFlow,Spark",
            years=2,
            location="Remote USA",
        )
        low_match = make_job(
            title="Low Match",
            skills="Python,COBOL,Fortran,Java,Ruby",
            years=2,
            location="Remote USA",
        )
        result = ranking_tool(
            jobs=[high_match, low_match],
            candidate_skills=["Python", "SQL", "machine learning", "TensorFlow", "Spark"],
            candidate_years=2,
            preferred_location="Remote",
        )
        scores = {j["job_title"]: j["score"] for j in result["ranked_jobs"]}
        assert scores["High Match"] > scores["Low Match"], (
            f"High match score ({scores['High Match']}) should exceed low match ({scores['Low Match']})"
        )

    def test_ranking_top3_count(self):
        """Test 8: top_3 has at most 3 items (fewer if <3 jobs passed in)."""
        result = self._get_ranked()
        assert len(result["top_3"]) <= 3, "top_3 should have at most 3 items"
        # With only 1 job
        single = ranking_tool(
            jobs=[SAMPLE_JOBS[0]],
            candidate_skills=["Python"],
            candidate_years=2,
            preferred_location="Remote",
        )
        assert len(single["top_3"]) == 1, "top_3 should have exactly 1 item when only 1 job"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET INTEGRITY TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetIntegrity:

    def test_dataset_integrity(self):
        """Test 9: CSV has 40 rows, 7 required columns, numeric years, no empty titles/companies."""
        required_columns = {
            "job_title", "company", "location", "required_skills",
            "years_experience", "job_description", "url",
        }
        rows = []
        csv_path = os.path.join(os.path.dirname(__file__), "..", DATASET_PATH)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert required_columns.issubset(set(reader.fieldnames or [])), (
                f"Missing columns: {required_columns - set(reader.fieldnames or [])}"
            )
            for row in reader:
                rows.append(row)

        assert len(rows) == 40, f"Expected 40 job rows, got {len(rows)}"

        for row in rows:
            assert row["job_title"].strip(), f"Empty job_title found: {row}"
            assert row["company"].strip(), f"Empty company found: {row}"
            try:
                int(row["years_experience"])
            except ValueError:
                pytest.fail(f"Non-numeric years_experience: {row['years_experience']} in {row['job_title']}")


# ─────────────────────────────────────────────────────────────────────────────
# STATE MACHINE TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestStateMachine:

    def test_state_machine_blocks_finish(self):
        """Test 10: finish is rejected (returns BLOCKED) when tailored=False."""
        agent = JobSearchAgent(dataset_path=DATASET_PATH)
        profile = {
            "name": "Test",
            "skills": ["Python"],
            "years_experience": 2,
            "preferred_location": "Remote",
            "remote_only": False,
            "exclude_companies": [],
        }
        # Call finish before any steps
        finish_call = {"tool": "finish", "params": {"summary": "done"}}
        result = agent._dispatch_tool(finish_call, profile)
        assert result.startswith("BLOCKED"), (
            f"Expected BLOCKED response, got: {result}"
        )

    def test_state_machine_allows_finish_after_all_steps(self):
        """Bonus: finish is accepted after all three steps are marked done."""
        agent = JobSearchAgent(dataset_path=DATASET_PATH)
        # Manually mark all steps as done
        agent._state["filtered"] = True
        agent._state["ranked"] = True
        agent._state["tailored"] = True
        profile = {
            "name": "Test",
            "skills": ["Python"],
            "years_experience": 2,
            "preferred_location": "Remote",
        }
        finish_call = {"tool": "finish", "params": {"summary": "All done!"}}
        result = agent._dispatch_tool(finish_call, profile)
        assert result.startswith("AGENT COMPLETE"), (
            f"Expected AGENT COMPLETE, got: {result}"
        )
