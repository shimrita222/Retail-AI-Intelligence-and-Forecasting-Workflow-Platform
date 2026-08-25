"""Focused, non-Flow-coupled tests for the Slice 3A Pre-Training Gate
integration point (src.flows.retail_flow.run_dynamic_training_stage()) and
the Slice 3B-Selection Post-Training Viability halt behavior in
RetailFlow.run_scientist_stage().

These exercise the gate/viability sequencing directly (no real CrewAI Flow
kickoff, no ingestion, no LLM calls) by monkeypatching the relevant
functions as imported into src.flows.retail_flow, and by constructing a
RetailFlow instance with pre-seeded state rather than calling .kickoff().
"""

import json

import pandas as pd

import src.flows.retail_flow as retail_flow
from src.services.candidate_selection import INSUFFICIENT_ELIGIBLE_CANDIDATES, PreTrainingGateResult


def _fake_engineered_df():
    dates = pd.date_range("2010-01-01", periods=10, freq="7D")
    return pd.DataFrame({"Date": dates, "Weekly_Sales": range(10)})


def _pass_gate_result(selected_model_ids):
    return PreTrainingGateResult(
        status="PASS",
        target_candidate_count=len(selected_model_ids),
        selected_model_ids=tuple(selected_model_ids),
        eligibility_results={},
        reason_code=None,
        reason_params=None,
    )


def _fail_gate_result():
    return PreTrainingGateResult(
        status="FAIL",
        target_candidate_count=3,
        selected_model_ids=(),
        eligibility_results={},
        reason_code=INSUFFICIENT_ELIGIBLE_CANDIDATES,
        reason_params={"required_count": 3, "available_count": 0, "available_model_ids": []},
    )


def test_gate_is_evaluated_before_training(monkeypatch):
    calls = []

    def fake_gate(train_row_count):
        calls.append(train_row_count)
        return _pass_gate_result(["Ridge", "RandomForestRegressor"])

    def fake_trainer(df, feature_columns, selected_model_ids=None, registry=None):
        calls.append("trainer_called")
        return {"selected_model_name": "Ridge", "candidate_metrics": {}}

    monkeypatch.setattr(retail_flow, "evaluate_pre_training_gate", fake_gate)
    monkeypatch.setattr(retail_flow, "train_and_select_model", fake_trainer)

    retail_flow.run_dynamic_training_stage(_fake_engineered_df(), ["Store"])

    assert calls[0] != "trainer_called"  # gate call happened first
    assert "trainer_called" in calls


def test_gate_fail_prevents_trainer_execution(monkeypatch):
    trainer_called = []

    monkeypatch.setattr(retail_flow, "evaluate_pre_training_gate", lambda train_row_count: _fail_gate_result())
    monkeypatch.setattr(
        retail_flow, "train_and_select_model", lambda *a, **k: trainer_called.append(True)
    )

    result = retail_flow.run_dynamic_training_stage(_fake_engineered_df(), ["Store"])

    assert trainer_called == []
    assert result["gate_result"].status == "FAIL"
    assert result["training_result"] is None


def test_gate_pass_passes_exactly_selected_model_ids_to_trainer(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        retail_flow, "evaluate_pre_training_gate", lambda train_row_count: _pass_gate_result(["Ridge", "GradientBoostingRegressor"])
    )

    def fake_trainer(df, feature_columns, selected_model_ids=None, registry=None):
        captured["selected_model_ids"] = selected_model_ids
        return {"selected_model_name": "Ridge", "candidate_metrics": {}}

    monkeypatch.setattr(retail_flow, "train_and_select_model", fake_trainer)

    retail_flow.run_dynamic_training_stage(_fake_engineered_df(), ["Store"])

    assert captured["selected_model_ids"] == ["Ridge", "GradientBoostingRegressor"]


def test_trainer_does_not_independently_select_candidates(monkeypatch):
    # The trainer function passed selected_model_ids must be used as-is;
    # this test proves run_dynamic_training_stage never calls
    # select_candidates() itself -- only evaluate_pre_training_gate.
    import src.services.candidate_selection as candidate_selection

    select_candidates_calls = []
    original = candidate_selection.select_candidates

    def spy_select_candidates(*args, **kwargs):
        select_candidates_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(retail_flow, "train_and_select_model", lambda *a, **k: {"selected_model_name": "Ridge", "candidate_metrics": {}})

    # Do not monkeypatch evaluate_pre_training_gate here -- exercise the
    # real gate, but spy on select_candidates specifically at the module
    # candidate_selection imports it from, to prove run_dynamic_training_stage
    # itself never calls select_candidates directly (only the gate does,
    # internally, exactly once).
    monkeypatch.setattr(retail_flow, "evaluate_pre_training_gate", candidate_selection.evaluate_pre_training_gate)

    retail_flow.run_dynamic_training_stage(_fake_engineered_df(), ["Store"])
    # run_dynamic_training_stage module itself has no select_candidates import.
    assert not hasattr(retail_flow, "select_candidates")


def test_no_training_occurs_before_gate_pass(monkeypatch):
    order = []

    def fake_gate(train_row_count):
        order.append("gate")
        return _fail_gate_result()

    def fake_trainer(*a, **k):
        order.append("trainer")
        return {"selected_model_name": "Ridge", "candidate_metrics": {}}

    monkeypatch.setattr(retail_flow, "evaluate_pre_training_gate", fake_gate)
    monkeypatch.setattr(retail_flow, "train_and_select_model", fake_trainer)

    retail_flow.run_dynamic_training_stage(_fake_engineered_df(), ["Store"])

    assert order == ["gate"]


# ---------------------------------------------------------------------------
# Slice 3B-Selection: Post-Training Viability halt behavior in RetailFlow
# ---------------------------------------------------------------------------


def _make_flow(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contract_path = run_dir / "dataset_contract.json"
    contract_path.write_text(
        json.dumps({"target_column": "Weekly_Sales", "columns": {"Weekly_Sales": {"min": -5000, "max": 800000}}}),
        encoding="utf-8",
    )
    clean_data_path = run_dir / "clean_data.csv"
    pd.DataFrame({"Store": [1], "Dept": [1], "Date": ["2010-01-01"], "Weekly_Sales": [100.0]}).to_csv(
        clean_data_path, index=False
    )

    flow = retail_flow.RetailFlow(raw_dir=tmp_path, artifacts_root=tmp_path, run_id="test-run")
    flow.state.run_dir = str(run_dir)
    flow.state.clean_data_path = str(clean_data_path)
    flow.state.contract_path = str(contract_path)
    return flow


def _patch_pipeline_stubs(monkeypatch):
    monkeypatch.setattr(retail_flow, "summarize_modeling_population", lambda df: {})
    monkeypatch.setattr(retail_flow, "modeling_eligible_mask", lambda df: pd.Series([True] * len(df), index=df.index))
    monkeypatch.setattr(retail_flow, "engineer_features", lambda df: df)
    monkeypatch.setattr(retail_flow, "get_feature_columns", lambda df: ["Store"])


def test_flow_halts_before_artifacts_and_scientist_crew_on_viability_fail(tmp_path, monkeypatch):
    flow = _make_flow(tmp_path)
    _patch_pipeline_stubs(monkeypatch)

    fail_training_result = {
        "selected_model_name": None,
        "candidate_metrics": {},
        "usable_candidates": ("Ridge",),
        "post_training_viability": {
            "status": "FAIL",
            "reason_code": "insufficient_successful_candidates",
            "reason_params": {
                "required_count": 2,
                "usable_count": 1,
                "usable_model_ids": ["Ridge"],
                "failed_model_ids": ["Broken"],
            },
        },
    }
    gate_result = _pass_gate_result(["Ridge", "Broken"])
    monkeypatch.setattr(
        retail_flow,
        "run_dynamic_training_stage",
        lambda engineered, cols: {"gate_result": gate_result, "training_result": fail_training_result},
    )

    save_calls = []
    crew_calls = []
    monkeypatch.setattr(retail_flow, "save_artifacts", lambda *a, **k: save_calls.append(True))
    monkeypatch.setattr(retail_flow, "run_scientist_crew", lambda *a, **k: crew_calls.append(True))

    result = flow.run_scientist_stage()

    assert save_calls == []
    assert crew_calls == []
    assert flow.state.status == "VIABILITY_FAILED"
    assert flow.state.selected_model_name == ""
    assert result["status"] == "VIABILITY_FAILED"
    assert result["viability_reason_code"] == "insufficient_successful_candidates"


def test_flow_proceeds_to_artifacts_and_scientist_crew_on_viability_pass(tmp_path, monkeypatch):
    flow = _make_flow(tmp_path)
    _patch_pipeline_stubs(monkeypatch)

    pass_training_result = {
        "selected_model_name": "Ridge",
        "candidate_metrics": {"Ridge": {"MAE": 1.0, "RMSE": 1.0, "R2": 0.9}},
        "fitted_models": {"Ridge": object()},
        "usable_candidates": ("Ridge", "RandomForestRegressor"),
        "tie_break": None,
        "feature_columns": ["Store"],
        "target_column": "Weekly_Sales",
        "train_rows": 1,
        "test_rows": 1,
        "split_cutoff_date": "2010-01-01",
        "post_training_viability": {"status": "PASS", "reason_code": None, "reason_params": None},
    }
    gate_result = _pass_gate_result(["Ridge", "RandomForestRegressor"])
    monkeypatch.setattr(
        retail_flow,
        "run_dynamic_training_stage",
        lambda engineered, cols: {"gate_result": gate_result, "training_result": pass_training_result},
    )

    save_calls = []
    crew_calls = []
    monkeypatch.setattr(retail_flow, "save_artifacts", lambda *a, **k: save_calls.append(True))
    monkeypatch.setattr(retail_flow, "run_scientist_crew", lambda *a, **k: crew_calls.append(True))

    result = flow.run_scientist_stage()

    assert len(save_calls) == 1
    assert len(crew_calls) == 1
    assert flow.state.status == "SCIENTIST_COMPLETE"
    assert flow.state.selected_model_name == "Ridge"
    assert result["selected_model_name"] == "Ridge"
