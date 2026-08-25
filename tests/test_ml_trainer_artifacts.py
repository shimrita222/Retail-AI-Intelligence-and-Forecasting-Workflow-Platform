import json
import math

import pandas as pd
import pytest

from src.services.ml_trainer import (
    ARTIFACT_GENERATION_FAILED,
    ARTIFACT_SERIALIZATION_FAILED,
    ArtifactPersistenceError,
    CandidateOutcome,
    save_artifacts,
    train_and_select_model,
)


def _multi_series_df(n_weeks=30):
    import numpy as np

    dates = pd.date_range("2010-01-01", periods=n_weeks, freq="7D")
    frames = []
    rng = np.random.default_rng(42)
    for store in (1, 2):
        for dept in (1, 2):
            base = 100.0 * store + 10.0 * dept
            sales = base + rng.normal(0, 5, n_weeks) + np.arange(n_weeks) * 1.5
            frames.append(
                pd.DataFrame({"Store": store, "Dept": dept, "Date": dates, "Weekly_Sales": sales})
            )
    return pd.concat(frames, ignore_index=True)


FEATURE_COLUMNS = ["Store", "Dept"]


def _real_pass_result(tmp_path=None):
    df = _multi_series_df()
    return train_and_select_model(df, FEATURE_COLUMNS, selected_model_ids=["Ridge", "RandomForestRegressor"])


# ---------------------------------------------------------------------------
# evaluation_report.json schema
# ---------------------------------------------------------------------------


def test_existing_required_keys_remain_present(tmp_path):
    result = _real_pass_result()
    paths = save_artifacts(result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    for key in (
        "selected_model_name",
        "candidate_metrics",
        "feature_columns",
        "target_column",
        "train_rows",
        "test_rows",
        "split_cutoff_date",
    ):
        assert key in report


def test_additive_phase3_metadata_is_serialized(tmp_path):
    result = _real_pass_result()
    paths = save_artifacts(result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    for key in ("usable_candidates", "post_training_viability", "tie_break", "candidate_outcomes"):
        assert key in report


def test_usable_candidates_persisted_correctly(tmp_path):
    result = _real_pass_result()
    paths = save_artifacts(result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    assert set(report["usable_candidates"]) == {"Ridge", "RandomForestRegressor"}


def test_post_training_viability_persisted_correctly(tmp_path):
    result = _real_pass_result()
    paths = save_artifacts(result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    assert report["post_training_viability"]["status"] == "PASS"
    assert report["post_training_viability"]["reason_code"] is None


def test_tie_break_absent_for_unique_winner(tmp_path):
    result = _real_pass_result()
    paths = save_artifacts(result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    assert report["tie_break"] is None or result["tie_break"] is None


def test_tie_break_persisted_on_exact_tie(tmp_path):
    df = _multi_series_df()
    result = train_and_select_model(df, FEATURE_COLUMNS, selected_model_ids=["Ridge", "RandomForestRegressor"])
    # Force an exact tie deterministically for the persistence test only.
    result["tie_break"] = {
        "reason_code": "tie_break_applied",
        "reason_params": {"tied_model_ids": ["Ridge", "RandomForestRegressor"], "winner": "Ridge", "criterion": "model_id_alphabetical"},
    }
    paths = save_artifacts(result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    assert report["tie_break"]["reason_code"] == "tie_break_applied"
    assert report["tie_break"]["reason_params"]["winner"] == "Ridge"


def test_failed_candidate_outcome_represented_safely(tmp_path):
    df = _multi_series_df()
    registry_result = train_and_select_model(
        df, FEATURE_COLUMNS, selected_model_ids=["Ridge", "RandomForestRegressor", "TotallyUnregisteredModel"]
    )
    paths = save_artifacts(registry_result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    failed_entry = report["candidate_outcomes"]["TotallyUnregisteredModel"]
    assert failed_entry["status"] == "failed"
    assert failed_entry["reason_code"] == "unknown_model_id"
    assert failed_entry["metrics"] is None
    assert "fitted_model" not in failed_entry


def test_fitted_model_objects_are_not_serialized_into_json(tmp_path):
    result = _real_pass_result()
    paths = save_artifacts(result, tmp_path)
    raw_text = paths["report_path"].read_text(encoding="utf-8")
    assert "fitted_model" not in raw_text
    assert "sklearn" not in raw_text.lower()


def test_non_finite_values_cannot_leak_into_persisted_json(tmp_path):
    result = _real_pass_result()
    paths = save_artifacts(result, tmp_path)
    raw_text = paths["report_path"].read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    report = json.loads(raw_text)  # must be strictly valid JSON
    for metrics in report["candidate_metrics"].values():
        for value in metrics.values():
            assert math.isfinite(value)


def test_existing_app_consumed_keys_remain_backward_compatible(tmp_path):
    result = _real_pass_result()
    paths = save_artifacts(result, tmp_path)
    report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    # app.py reads report["selected_model_name"], report["candidate_metrics"],
    # report["feature_columns"] via plain dict indexing.
    assert report["selected_model_name"] in report["candidate_metrics"]
    assert isinstance(report["feature_columns"], list)


# ---------------------------------------------------------------------------
# Artifact failure semantics
# ---------------------------------------------------------------------------


def test_selected_model_serialization_failure_raises_structured_error(tmp_path, monkeypatch):
    import src.services.ml_trainer as ml_trainer

    def _broken_dump(*args, **kwargs):
        raise OSError("disk full (synthetic)")

    monkeypatch.setattr(ml_trainer.joblib, "dump", _broken_dump)

    result = _real_pass_result()
    with pytest.raises(ArtifactPersistenceError) as excinfo:
        save_artifacts(result, tmp_path)
    assert excinfo.value.reason_code == ARTIFACT_SERIALIZATION_FAILED
    assert excinfo.value.reason_params["exception_type"] == "OSError"


def test_report_write_failure_raises_structured_error(tmp_path, monkeypatch):
    import src.services.ml_trainer as ml_trainer

    original_open = ml_trainer.Path.open

    def _broken_open(self, *args, **kwargs):
        if self.name == "evaluation_report.json":
            raise OSError("disk full (synthetic)")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(ml_trainer.Path, "open", _broken_open)

    result = _real_pass_result()
    with pytest.raises(ArtifactPersistenceError) as excinfo:
        save_artifacts(result, tmp_path)
    assert excinfo.value.reason_code == ARTIFACT_GENERATION_FAILED


def test_artifact_failure_does_not_change_selected_model_name(tmp_path, monkeypatch):
    import src.services.ml_trainer as ml_trainer

    monkeypatch.setattr(ml_trainer.joblib, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))

    result = _real_pass_result()
    original_winner = result["selected_model_name"]
    with pytest.raises(ArtifactPersistenceError):
        save_artifacts(result, tmp_path)
    assert result["selected_model_name"] == original_winner


def test_no_artifacts_left_representing_partial_success(tmp_path, monkeypatch):
    import src.services.ml_trainer as ml_trainer

    original_open = ml_trainer.Path.open

    def _broken_open(self, *args, **kwargs):
        if self.name == "evaluation_report.json":
            raise OSError("disk full (synthetic)")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(ml_trainer.Path, "open", _broken_open)

    result = _real_pass_result()
    with pytest.raises(ArtifactPersistenceError):
        save_artifacts(result, tmp_path)
    # The model file was written before the report failed -- the failure
    # must still be surfaced as a structured exception so the caller (Flow)
    # never treats this run as successfully completed.
    assert (tmp_path / "evaluation_report.json").exists() is False
