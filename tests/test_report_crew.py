from fastapi.testclient import TestClient

from projects.report_crew.agents import CriticAgent, PlannerAgent, WriterAgent
from projects.report_crew.api import app
from projects.report_crew.crew import ReportCrew

client = TestClient(app)


# --- Unit tests: individual agents -------------------------------------------


def test_planner_produces_five_sections_mentioning_topic():
    outline = PlannerAgent().plan("zero-trust API security")
    assert len(outline) == 5
    assert outline[0] == "Introduction"
    assert outline[-1] == "Conclusion"
    assert any("zero-trust API security" in section for section in outline)


def test_writer_first_pass_is_thin():
    outline = PlannerAgent().plan("observability")
    report = WriterAgent().draft(outline, "observability")
    for section in outline:
        sentence_count = report[section].count(".")
        assert sentence_count == WriterAgent.BASE_SENTENCES


def test_writer_expands_only_flagged_sections():
    outline = ["Introduction", "Conclusion"]
    writer = WriterAgent()
    report = writer.draft(outline, "topic", revision_sections={"Introduction"})
    assert report["Introduction"].count(".") == WriterAgent.EXPANDED_SENTENCES
    assert report["Conclusion"].count(".") == WriterAgent.BASE_SENTENCES


def test_critic_flags_short_sections():
    critic = CriticAgent(min_sentences=3)
    report = {"Introduction": "One sentence. Two sentences."}
    result = critic.review(report, ["Introduction"])
    assert result.approved is False
    assert "Introduction" in result.issues[0]


def test_critic_flags_missing_sections():
    critic = CriticAgent(min_sentences=3)
    result = critic.review({}, ["Introduction"])
    assert result.approved is False
    assert "missing" in result.issues[0].lower()


def test_critic_approves_sufficiently_long_report():
    critic = CriticAgent(min_sentences=2)
    report = {"Introduction": "One sentence. Two sentences."}
    result = critic.review(report, ["Introduction"])
    assert result.approved is True
    assert result.issues == []


# --- Crew orchestration --------------------------------------------------------


def test_crew_converges_in_exactly_two_iterations():
    crew = ReportCrew(PlannerAgent(), WriterAgent(), CriticAgent())
    result = crew.run("zero-trust API security", max_iterations=5)
    assert result.approved is True
    assert len(result.iterations) == 2
    assert result.iterations[0].critique.approved is False
    assert result.iterations[1].critique.approved is True


def test_crew_respects_max_iterations_budget():
    class _StubbornCritic:
        def review(self, report, outline):
            from projects.report_crew.agents import CritiqueResult

            return CritiqueResult(approved=False, issues=["never satisfied"])

    crew = ReportCrew(PlannerAgent(), WriterAgent(), _StubbornCritic())
    result = crew.run("anything", max_iterations=2)
    assert result.approved is False
    assert len(result.iterations) == 2


def test_crew_final_report_covers_every_outline_section():
    crew = ReportCrew(PlannerAgent(), WriterAgent(), CriticAgent())
    result = crew.run("data pipeline reliability")
    assert set(result.final_report.keys()) == set(result.outline)


# --- API tests -----------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_create_report_endpoint():
    resp = client.post("/crew/report", json={"topic": "prompt injection defenses"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["approved"] is True
    assert body["iterations_taken"] == 2
    assert set(body["final_report"].keys()) == set(body["outline"])


def test_create_report_rejects_empty_topic():
    resp = client.post("/crew/report", json={"topic": ""})
    assert resp.status_code == 422
