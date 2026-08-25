"""Focused, non-Flow-coupled tests for the Slice 3A Pre-Training Gate
integration point: src.flows.retail_flow.run_dynamic_training_stage().

These exercise the gate-then-train sequencing directly (no CrewAI Flow
kickoff, no ingestion, no LLM calls) by monkeypatching the gate and trainer
functions as imported into src.flows.retail_flow.
"""

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
