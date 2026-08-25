import pytest

import src.agents.scientist_crew as scientist_crew_module
from src.agents.scientist_crew import (
    _build_crew,
    render_evaluation_report_markdown,
    render_model_card_markdown,
    run_scientist_crew,
)
from src.services.ml_trainer import ArtifactPersistenceError, CandidateOutcome


class _FakeTaskOutput:
    def __init__(self, text="ok"):
        self._text = text

    def __str__(self):
        return self._text


class _FakeTask:
    def __init__(self):
        self.output = _FakeTaskOutput()


class _FakeCrew:
    def kickoff(self):
        return None


def _fake_build_crew(training_result, feature_columns):
    return _FakeCrew(), (_FakeTask(), _FakeTask(), _FakeTask())


def _outcome(model_id, status="success", rmse=None):
    metrics = None if status != "success" else {"MAE": 1.0, "RMSE": rmse, "R2": 0.9}
    return CandidateOutcome(
        model_id=model_id,
        status=status,
        metrics=metrics,
        fitted_model=object() if status == "success" else None,
        reason_code=None if status == "success" else "non_finite_metric",
        reason_params=None if status == "success" else {"model_id": model_id, "non_finite_metric_names": ["RMSE"]},
    )


def _two_candidate_result(tie_break=None):
    return {
        "selected_model_name": "RandomForestRegressor",
        "candidate_metrics": {
            "Ridge": {"MAE": 5.0, "RMSE": 6.0, "R2": 0.8},
            "RandomForestRegressor": {"MAE": 3.0, "RMSE": 4.0, "R2": 0.9},
        },
        "candidate_outcomes": {
            "Ridge": _outcome("Ridge", rmse=6.0),
            "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=4.0),
        },
        "usable_candidates": ("RandomForestRegressor", "Ridge"),
        "tie_break": tie_break,
        "feature_columns": ["Store", "Dept"],
        "target_column": "Weekly_Sales",
        "train_rows": 100,
        "test_rows": 20,
        "split_cutoff_date": "2010-06-01",
    }


def _three_candidate_result():
    return {
        "selected_model_name": "GradientBoostingRegressor",
        "candidate_metrics": {
            "Ridge": {"MAE": 5.0, "RMSE": 6.0, "R2": 0.8},
            "RandomForestRegressor": {"MAE": 4.0, "RMSE": 5.0, "R2": 0.85},
            "GradientBoostingRegressor": {"MAE": 2.0, "RMSE": 3.0, "R2": 0.95},
        },
        "candidate_outcomes": {
            "Ridge": _outcome("Ridge", rmse=6.0),
            "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=5.0),
            "GradientBoostingRegressor": _outcome("GradientBoostingRegressor", rmse=3.0),
        },
        "usable_candidates": ("GradientBoostingRegressor", "RandomForestRegressor", "Ridge"),
        "tie_break": None,
        "feature_columns": ["Store", "Dept"],
        "target_column": "Weekly_Sales",
        "train_rows": 100,
        "test_rows": 20,
        "split_cutoff_date": "2010-06-01",
    }


_EXACT_TIE_BREAK = {
    "reason_code": "tie_break_applied",
    "reason_params": {"tied_model_ids": ["Ridge", "RandomForestRegressor"], "winner": "Ridge", "criterion": "model_id_alphabetical"},
}


def _modeling_population():
    return {
        "total_clean_rows": 421570,
        "eligible_rows": 420924,
        "excluded_rows": 646,
        "excluded_departments": [47],
        "exclusion_reasons": {47: "Unresolved business semantics combined with verified structural anomalies."},
        "watchlist": {78: "Elevated anomaly rate; retained for modeling."},
    }


# ---------------------------------------------------------------------------
# evaluation_report.md
# ---------------------------------------------------------------------------


def test_evaluation_report_renders_for_two_candidates():
    md = render_evaluation_report_markdown(_two_candidate_result())
    assert "Ridge" in md
    assert "RandomForestRegressor" in md
    assert "**Selected model:** `RandomForestRegressor`" in md


def test_evaluation_report_renders_for_three_candidates():
    md = render_evaluation_report_markdown(_three_candidate_result())
    assert "Ridge" in md
    assert "RandomForestRegressor" in md
    assert "GradientBoostingRegressor" in md


def test_evaluation_report_unique_winner_has_no_tie_break_line():
    md = render_evaluation_report_markdown(_two_candidate_result(tie_break=None))
    assert "Tie-break" not in md


def test_evaluation_report_exact_tie_renders_tie_break_line():
    md = render_evaluation_report_markdown(_two_candidate_result(tie_break=_EXACT_TIE_BREAK))
    assert "Tie-break applied" in md
    assert "model_id_alphabetical" in md
    assert "-> `Ridge`" in md


def test_evaluation_report_has_no_stale_fixed_two_model_wording():
    md = render_evaluation_report_markdown(_three_candidate_result())
    assert "Exactly two candidates" not in md
    assert "No other model families were evaluated" not in md


def test_evaluation_report_metrics_are_supplied_not_recomputed():
    result = _two_candidate_result()
    md = render_evaluation_report_markdown(result)
    assert "4.00" in md  # RandomForestRegressor RMSE from candidate_metrics, unmodified
    assert "6.00" in md  # Ridge RMSE from candidate_metrics, unmodified


# ---------------------------------------------------------------------------
# model_card.md
# ---------------------------------------------------------------------------


def test_model_card_renders_for_two_candidates():
    md = render_model_card_markdown(_two_candidate_result())
    assert "Ridge" in md
    assert "RandomForestRegressor" in md
    assert "2 candidate model(s) were selected" in md


def test_model_card_renders_for_three_candidates():
    md = render_model_card_markdown(_three_candidate_result())
    assert "3 candidate model(s) were selected" in md
    assert "GradientBoostingRegressor" in md


def test_model_card_lists_selected_candidates_dynamically():
    md = render_model_card_markdown(_three_candidate_result())
    assert "`Ridge`" in md
    assert "`RandomForestRegressor`" in md
    assert "`GradientBoostingRegressor`" in md


def test_model_card_shows_winner_correctly():
    md = render_model_card_markdown(_three_candidate_result())
    assert "# Model Card — GradientBoostingRegressor" in md
    assert "**Model:** GradientBoostingRegressor" in md


def test_model_card_has_no_stale_exactly_two_candidates_statement():
    md = render_model_card_markdown(_three_candidate_result())
    assert "Exactly two candidates" not in md


def test_model_card_has_no_stale_no_other_model_families_claim():
    md = render_model_card_markdown(_three_candidate_result())
    assert "No other model families were evaluated" not in md


def test_model_card_excludes_unusable_candidates_from_usable_list():
    result = _two_candidate_result()
    result["candidate_outcomes"]["Ridge"] = _outcome("Ridge", status="failed")
    result["usable_candidates"] = ("RandomForestRegressor",)
    # Re-derive candidate_metrics to only contain the usable candidate, matching
    # what train_and_select_model() actually produces on a real run.
    result["candidate_metrics"] = {"RandomForestRegressor": {"MAE": 3.0, "RMSE": 4.0, "R2": 0.9}}
    md = render_model_card_markdown(result)
    assert "Excluded from comparison" in md
    assert "`Ridge` (`non_finite_metric`)" in md
    assert "**Usable for comparison:** `RandomForestRegressor`" in md


def test_model_card_modeling_population_section_renders_correctly():
    result = _two_candidate_result()
    result["modeling_population"] = _modeling_population()
    md = render_model_card_markdown(result)
    assert "## Modeling Population" in md
    assert "421,570 total descriptive rows" in md
    assert "420,924 eligible for predictive training" in md
    assert "646 excluded" in md


def test_model_card_department_47_exclusion_wording_is_descriptive_not_data_cleaning():
    result = _two_candidate_result()
    result["modeling_population"] = _modeling_population()
    md = render_model_card_markdown(result)
    assert "Department 47" in md
    assert "modeling-eligibility decision, NOT a data-cleaning decision" in md
    assert "remain in clean_data.csv and all descriptive/EDA outputs" in md


def test_model_card_watchlist_section_renders_when_present():
    result = _two_candidate_result()
    result["modeling_population"] = _modeling_population()
    md = render_model_card_markdown(result)
    assert "Documented (non-exclusionary) watchlist" in md
    assert "Department 78" in md


def test_model_card_modeling_population_absent_renders_no_section():
    result = _two_candidate_result()
    md = render_model_card_markdown(result)
    assert "## Modeling Population" not in md


def test_model_card_rendering_is_deterministic():
    result = _three_candidate_result()
    first = render_model_card_markdown(result)
    second = render_model_card_markdown(result)
    assert first == second


# ---------------------------------------------------------------------------
# Scientist Crew boundary (no real LLM calls)
# ---------------------------------------------------------------------------


def test_eval_task_prompt_is_n_model_generic_not_ridge_vs_randomforest():
    _crew, (feature_task, eval_task, governance_task) = _build_crew(_three_candidate_result(), ["Store", "Dept"])
    assert "Ridge vs RandomForest" not in eval_task.description
    assert "Ridge vs RandomForest" not in eval_task.agent.goal


def test_eval_task_instructs_no_recomputation_or_model_override():
    _crew, (feature_task, eval_task, governance_task) = _build_crew(_three_candidate_result(), ["Store", "Dept"])
    assert "do not recompute" in eval_task.description.lower()
    assert "model choice" in eval_task.description.lower()


def test_eval_task_carries_tie_break_evidence_when_present():
    result = _two_candidate_result(tie_break=_EXACT_TIE_BREAK)
    _crew, (feature_task, eval_task, governance_task) = _build_crew(result, ["Store", "Dept"])
    assert "tie_break_applied" in eval_task.description


def test_scientist_crew_agents_have_no_winner_authority_in_goals():
    _crew, (feature_task, eval_task, governance_task) = _build_crew(_three_candidate_result(), ["Store", "Dept"])
    for agent in (feature_task.agent, eval_task.agent, governance_task.agent):
        goal_lower = agent.goal.lower()
        assert "decide" not in goal_lower
        assert "choose" not in goal_lower


def test_scientist_crew_has_exactly_three_agents():
    crew, _tasks = _build_crew(_three_candidate_result(), ["Store", "Dept"])
    assert len(crew.agents) == 3


# ---------------------------------------------------------------------------
# Required-artifact write-failure semantics (evaluation_report.md / model_card.md)
# ---------------------------------------------------------------------------


def test_evaluation_report_md_write_failure_raises_artifact_generation_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(scientist_crew_module, "_build_crew", _fake_build_crew)
    original_write_text = scientist_crew_module.Path.write_text

    def _broken_write_text(self, *args, **kwargs):
        if self.name == "evaluation_report.md":
            raise OSError("disk full (synthetic)")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(scientist_crew_module.Path, "write_text", _broken_write_text)

    result = _two_candidate_result()
    with pytest.raises(ArtifactPersistenceError) as excinfo:
        run_scientist_crew(result, tmp_path)

    assert excinfo.value.reason_code == "artifact_generation_failed"
    assert excinfo.value.reason_params["artifact"] == "evaluation_report.md"
    assert excinfo.value.reason_params["stage"] == "markdown_write"
    assert excinfo.value.reason_params["exception_type"] == "OSError"


def test_model_card_md_write_failure_raises_artifact_generation_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(scientist_crew_module, "_build_crew", _fake_build_crew)
    original_write_text = scientist_crew_module.Path.write_text

    def _broken_write_text(self, *args, **kwargs):
        if self.name == "model_card.md":
            raise OSError("disk full (synthetic)")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(scientist_crew_module.Path, "write_text", _broken_write_text)

    result = _two_candidate_result()
    with pytest.raises(ArtifactPersistenceError) as excinfo:
        run_scientist_crew(result, tmp_path)

    assert excinfo.value.reason_code == "artifact_generation_failed"
    assert excinfo.value.reason_params["artifact"] == "model_card.md"
    assert excinfo.value.reason_params["stage"] == "markdown_write"


def test_evaluation_report_md_succeeds_before_model_card_md_failure_is_still_a_failure(tmp_path, monkeypatch):
    # evaluation_report.md write succeeds (real write), model_card.md fails --
    # proves partial success is not silently swallowed.
    monkeypatch.setattr(scientist_crew_module, "_build_crew", _fake_build_crew)
    original_write_text = scientist_crew_module.Path.write_text

    def _broken_write_text(self, *args, **kwargs):
        if self.name == "model_card.md":
            raise OSError("disk full (synthetic)")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(scientist_crew_module.Path, "write_text", _broken_write_text)

    result = _two_candidate_result()
    with pytest.raises(ArtifactPersistenceError):
        run_scientist_crew(result, tmp_path)

    assert (tmp_path / "evaluation_report.md").exists()  # partial artifact, written before the failure
    assert not (tmp_path / "model_card.md").exists()


def test_markdown_write_failure_does_not_change_selected_model_name(tmp_path, monkeypatch):
    monkeypatch.setattr(scientist_crew_module, "_build_crew", _fake_build_crew)
    original_write_text = scientist_crew_module.Path.write_text

    def _broken_write_text(self, *args, **kwargs):
        if self.name == "model_card.md":
            raise OSError("disk full (synthetic)")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(scientist_crew_module.Path, "write_text", _broken_write_text)

    result = _two_candidate_result()
    original_winner = result["selected_model_name"]
    with pytest.raises(ArtifactPersistenceError):
        run_scientist_crew(result, tmp_path)
    assert result["selected_model_name"] == original_winner


def test_llm_kickoff_failure_is_not_conflated_with_artifact_write_failure(tmp_path):
    # No monkeypatch of Path.write_text here -- only _build_crew's kickoff()
    # is broken. This must be handled by the existing narration-failure
    # branch (producing narratives["error"]) and must NOT raise
    # ArtifactPersistenceError -- writes should still succeed normally.
    class _BrokenCrew:
        def kickoff(self):
            raise RuntimeError("synthetic LLM/network failure")

    def _broken_kickoff_build_crew(training_result, feature_columns):
        return _BrokenCrew(), (_FakeTask(), _FakeTask(), _FakeTask())

    import src.agents.scientist_crew as scm

    original_build_crew = scm._build_crew
    scm._build_crew = _broken_kickoff_build_crew
    try:
        result = run_scientist_crew(_two_candidate_result(), tmp_path)
    finally:
        scm._build_crew = original_build_crew

    assert (tmp_path / "evaluation_report.md").exists()
    assert (tmp_path / "model_card.md").exists()
    assert "error" in result["narratives"]
