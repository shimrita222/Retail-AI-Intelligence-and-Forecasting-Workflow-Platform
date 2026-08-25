import numpy as np
import pandas as pd

from src.services.ml_trainer import (
    INSUFFICIENT_SUCCESSFUL_CANDIDATES,
    MINIMUM_USABLE_CANDIDATES,
    TIE_BREAK_APPLIED,
    CandidateOutcome,
    evaluate_post_training_viability,
    select_winner,
)


def _outcome(model_id, status="success", rmse=None, mae=1.0, r2=0.5):
    metrics = None if status != "success" else {"MAE": mae, "RMSE": rmse, "R2": r2}
    return CandidateOutcome(
        model_id=model_id,
        status=status,
        metrics=metrics,
        fitted_model=object() if status == "success" else None,
        reason_code=None if status == "success" else "candidate_training_failed",
        reason_params=None if status == "success" else {"model_id": model_id},
    )


# ---------------------------------------------------------------------------
# Post-Training Viability Check (Checkpoint 2)
# ---------------------------------------------------------------------------


def test_two_successful_usable_candidates_pass():
    outcomes = {"Ridge": _outcome("Ridge", rmse=1.0), "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=2.0)}
    result = evaluate_post_training_viability(outcomes)
    assert result.status == "PASS"
    assert result.usable_model_ids == ("RandomForestRegressor", "Ridge")


def test_three_successful_usable_candidates_pass():
    outcomes = {
        "Ridge": _outcome("Ridge", rmse=1.0),
        "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=2.0),
        "GradientBoostingRegressor": _outcome("GradientBoostingRegressor", rmse=3.0),
    }
    result = evaluate_post_training_viability(outcomes)
    assert result.status == "PASS"


def test_one_successful_usable_candidate_fails():
    outcomes = {"Ridge": _outcome("Ridge", rmse=1.0), "Broken": _outcome("Broken", status="failed")}
    result = evaluate_post_training_viability(outcomes)
    assert result.status == "FAIL"


def test_zero_successful_usable_candidates_fails():
    outcomes = {"Broken1": _outcome("Broken1", status="failed"), "Broken2": _outcome("Broken2", status="failed")}
    result = evaluate_post_training_viability(outcomes)
    assert result.status == "FAIL"


def test_fail_reason_code_is_insufficient_successful_candidates():
    outcomes = {"Ridge": _outcome("Ridge", rmse=1.0), "Broken": _outcome("Broken", status="failed")}
    result = evaluate_post_training_viability(outcomes)
    assert result.reason_code == INSUFFICIENT_SUCCESSFUL_CANDIDATES


def test_fail_reason_params_are_structured_and_auditable():
    outcomes = {"Ridge": _outcome("Ridge", rmse=1.0), "Broken": _outcome("Broken", status="failed")}
    result = evaluate_post_training_viability(outcomes, required_count=2)
    assert result.reason_params == {
        "required_count": 2,
        "usable_count": 1,
        "usable_model_ids": ["Ridge"],
        "failed_model_ids": ["Broken"],
    }


def test_failed_and_non_finite_candidates_excluded_from_usable():
    outcomes = {
        "Ridge": _outcome("Ridge", rmse=1.0),
        "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=2.0),
        "NonFinite": _outcome("NonFinite", status="failed"),  # non_finite_metric outcomes are status="failed"
    }
    result = evaluate_post_training_viability(outcomes)
    assert "NonFinite" not in result.usable_model_ids
    assert "NonFinite" in result.failed_model_ids


def test_post_training_check_uses_only_given_outcomes_no_retraining():
    # No df/feature_columns/registry are accepted by the function at all --
    # its signature alone proves it cannot re-run training.
    import inspect

    sig = inspect.signature(evaluate_post_training_viability)
    assert set(sig.parameters.keys()) == {"outcomes", "required_count"}


def test_post_training_check_does_not_import_candidate_selection():
    # Behavioral check, not a doc-text scan: neither name is bound in this
    # module's namespace at all (docstrings may still name them in prose to
    # explain the checkpoint-separation boundary -- that is documentation,
    # not an import).
    import src.services.ml_trainer as ml_trainer

    assert not hasattr(ml_trainer, "select_candidates")
    assert not hasattr(ml_trainer, "evaluate_pre_training_gate")


def test_minimum_usable_candidates_constant_is_two():
    assert MINIMUM_USABLE_CANDIDATES == 2


# ---------------------------------------------------------------------------
# Unique winner (no tie)
# ---------------------------------------------------------------------------


def test_two_model_unique_lowest_rmse_wins():
    outcomes = {"Ridge": _outcome("Ridge", rmse=5.0), "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=3.0)}
    result = select_winner(outcomes)
    assert result["winner_model_id"] == "RandomForestRegressor"
    assert result["tie_break"] is None


def test_three_model_unique_lowest_rmse_wins():
    outcomes = {
        "Ridge": _outcome("Ridge", rmse=5.0),
        "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=3.0),
        "GradientBoostingRegressor": _outcome("GradientBoostingRegressor", rmse=1.0),
    }
    result = select_winner(outcomes)
    assert result["winner_model_id"] == "GradientBoostingRegressor"
    assert result["tie_break"] is None


def test_runtime_cost_tier_is_irrelevant_to_unique_winner():
    # Structural check, not a doc-text scan: CandidateOutcome (the only
    # input select_winner() reads) carries no runtime_cost_tier field, so
    # there is nothing for the function to read even if it wanted to.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(CandidateOutcome)}
    assert "runtime_cost_tier" not in field_names


# ---------------------------------------------------------------------------
# Exact ties
# ---------------------------------------------------------------------------


def test_exact_two_model_tie_breaks_alphabetically():
    outcomes = {"Zeta": _outcome("Zeta", rmse=4.0), "Alpha": _outcome("Alpha", rmse=4.0)}
    result = select_winner(outcomes)
    assert result["winner_model_id"] == "Alpha"
    assert result["tie_break"] is not None
    assert result["tie_break"]["reason_code"] == TIE_BREAK_APPLIED
    assert result["tie_break"]["reason_params"] == {
        "tied_model_ids": ["Alpha", "Zeta"],
        "winner": "Alpha",
        "criterion": "model_id_alphabetical",
    }


def test_exact_three_model_tie_breaks_alphabetically():
    outcomes = {
        "Zeta": _outcome("Zeta", rmse=4.0),
        "Middle": _outcome("Middle", rmse=4.0),
        "Alpha": _outcome("Alpha", rmse=4.0),
    }
    result = select_winner(outcomes)
    assert result["winner_model_id"] == "Alpha"
    assert result["tie_break"]["reason_params"]["tied_model_ids"] == ["Alpha", "Middle", "Zeta"]
    assert result["tie_break"]["reason_params"]["winner"] == "Alpha"


def test_runtime_cost_tier_cannot_override_alphabetical_tie_break():
    # Construct a tie where an imaginary "prefer lower runtime_cost_tier"
    # ordering would pick a different winner than model_id alphabetical.
    # CandidateOutcome carries no runtime_cost_tier field at all, so there
    # is nothing for select_winner() to read even if it wanted to --
    # proving the boundary structurally, not just by absence of a branch.
    outcomes = {
        "ZLowCostWouldWinIfConsidered": _outcome("ZLowCostWouldWinIfConsidered", rmse=4.0),
        "AHighCostButAlphabeticallyFirst": _outcome("AHighCostButAlphabeticallyFirst", rmse=4.0),
    }
    assert not hasattr(list(outcomes.values())[0], "runtime_cost_tier")
    result = select_winner(outcomes)
    assert result["winner_model_id"] == "AHighCostButAlphabeticallyFirst"


def test_near_tie_is_not_a_tie():
    outcomes = {"Ridge": _outcome("Ridge", rmse=4.0000001), "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=4.0)}
    result = select_winner(outcomes)
    assert result["winner_model_id"] == "RandomForestRegressor"
    assert result["tie_break"] is None


def test_no_rounding_applied_before_comparison():
    outcomes = {
        "Ridge": _outcome("Ridge", rmse=4.00004),
        "RandomForestRegressor": _outcome("RandomForestRegressor", rmse=4.00003),
    }
    result = select_winner(outcomes)
    # If rounding to e.g. 2dp were applied, these would appear tied (4.00).
    assert result["winner_model_id"] == "RandomForestRegressor"
    assert result["tie_break"] is None
