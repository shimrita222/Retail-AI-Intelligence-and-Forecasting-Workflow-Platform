"""Deterministic MODEL candidate eligibility, selection, and pre-training gate.

This module is distinct from `src.services.modeling_eligibility`, which
governs DATA modeling eligibility (which dataset rows/departments enter the
training population -- e.g. Department 47). This module governs MODEL
candidate eligibility (which registered model implementations may enter a
training candidate pool). The two concepts are never mixed here; this module
does not read or alter Department 47 / watchlist policy in any way.

Five distinct concepts, computed here without collapsing into one opaque
boolean (see `src.services.model_registry` module docstring for the full
definitions): registered, enabled, dependency_available,
problem_type_compatible, eligible.

This module implements only:
  1. Eligibility computation over MODEL_REGISTRY (or a supplied registry).
  2. The frozen PRE-TRAINING candidate-count policy (2-vs-3, provisional
     50,000-row heuristic).
  3. Deterministic candidate selection over eligible models only, anchored
     on Ridge + RandomForestRegressor, with a boosting-family slot filled by
     the frozen Fallback Order (runtime_cost_tier ascending, then model_id
     alphabetical -- explicitly permitted at this pre-training stage only).
  4. The frozen Pre-Training Candidate Availability / Viability Check
     (Checkpoint 1), reason_code="insufficient_eligible_candidates".

It does NOT train models, does NOT implement the Post-Training Comparison
Viability Check (a different checkpoint, a different reason_code,
"insufficient_successful_candidates", owned by a later slice), and does NOT
implement or touch final-winner selection (RMSE ascending / model_id
alphabetical) -- runtime_cost_tier must never reach that later stage, per
the frozen contract amendment. No LLM participates in any decision here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.services.model_registry import (
    DEPENDENCY_UNAVAILABLE,
    FAMILY_BOOSTING_ENSEMBLE,
    MODEL_REGISTRY,
    PROBLEM_TYPE_REGRESSION,
    ModelEntry,
    is_available,
    unavailable_reason,
)

# Additional structured reason codes for Slice 2, minimal set. Never bare
# English prose used as decision state -- always reason_code + reason_params.
MODEL_DISABLED = "model_disabled"
PROBLEM_TYPE_INCOMPATIBLE = "problem_type_incompatible"
INSUFFICIENT_ELIGIBLE_CANDIDATES = "insufficient_eligible_candidates"

# Mandatory anchors: always attempted first, in this order, when eligible.
MANDATORY_ANCHOR_MODEL_IDS: tuple[str, ...] = ("Ridge", "RandomForestRegressor")

# PROVISIONAL, CONFIGURABLE heuristic -- not a measured optimum, not a
# VERIFIED FACT, not a performance claim. See the frozen Phase 3 contract.
CANDIDATE_COUNT_ROW_THRESHOLD = 50_000
TARGET_CANDIDATE_COUNT_BELOW_THRESHOLD = 2
TARGET_CANDIDATE_COUNT_AT_OR_ABOVE_THRESHOLD = 3

# Fallback Order (pre-training only): runtime_cost_tier ascending. This
# ranking is operational/resource-cost ordering, never a quality ranking.
_RUNTIME_COST_TIER_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class EligibilityResult:
    model_id: str
    registered: bool
    enabled: bool
    dependency_available: bool
    problem_type_compatible: bool
    eligible: bool
    reason_code: str | None
    reason_params: dict[str, Any] | None


@dataclass(frozen=True)
class PreTrainingGateResult:
    status: str  # "PASS" | "FAIL"
    target_candidate_count: int
    selected_model_ids: tuple[str, ...]
    eligibility_results: dict[str, EligibilityResult]
    reason_code: str | None
    reason_params: dict[str, Any] | None


def _evaluate_single(entry: ModelEntry, problem_type: str) -> EligibilityResult:
    """Priority order for a single blocking reason mirrors the frozen concept
    order (enabled -> dependency_available -> problem_type_compatible): the
    first failing check in that order is reported. A model is never
    reported eligible for more than one simultaneous failure reason.
    """
    dependency_available = is_available(entry)
    problem_type_compatible = problem_type in entry.problem_types
    eligible = entry.enabled and dependency_available and problem_type_compatible

    reason_code: str | None = None
    reason_params: dict[str, Any] | None = None
    if not entry.enabled:
        reason_code = MODEL_DISABLED
        reason_params = {"model_id": entry.model_id}
    elif not dependency_available:
        dep_reason = unavailable_reason(entry)
        assert dep_reason is not None  # dependency_available is False here
        reason_code = dep_reason["reason_code"]
        reason_params = dep_reason["reason_params"]
    elif not problem_type_compatible:
        reason_code = PROBLEM_TYPE_INCOMPATIBLE
        reason_params = {
            "model_id": entry.model_id,
            "requested_problem_type": problem_type,
            "supported_problem_types": list(entry.problem_types),
        }

    return EligibilityResult(
        model_id=entry.model_id,
        registered=True,
        enabled=entry.enabled,
        dependency_available=dependency_available,
        problem_type_compatible=problem_type_compatible,
        eligible=eligible,
        reason_code=reason_code,
        reason_params=reason_params,
    )


def compute_eligibility(
    problem_type: str = PROBLEM_TYPE_REGRESSION,
    registry: dict[str, ModelEntry] | None = None,
) -> dict[str, EligibilityResult]:
    """Deterministic, auditable eligibility for every entry in `registry`
    (defaults to the live MODEL_REGISTRY). Never mutates the registry.
    """
    source = registry if registry is not None else MODEL_REGISTRY
    return {model_id: _evaluate_single(entry, problem_type) for model_id, entry in source.items()}


def target_candidate_count(train_row_count: int) -> int:
    """Frozen, provisional 2-vs-3 candidate-count policy."""
    if train_row_count < 0:
        raise ValueError(f"train_row_count must be non-negative, got {train_row_count}")
    if train_row_count >= CANDIDATE_COUNT_ROW_THRESHOLD:
        return TARGET_CANDIDATE_COUNT_AT_OR_ABOVE_THRESHOLD
    return TARGET_CANDIDATE_COUNT_BELOW_THRESHOLD


def _fallback_order_key(entry: ModelEntry) -> tuple[int, str]:
    return (_RUNTIME_COST_TIER_RANK.get(entry.runtime_cost_tier, len(_RUNTIME_COST_TIER_RANK)), entry.model_id)


def select_candidates(
    eligibility_results: dict[str, EligibilityResult],
    target_count: int,
    registry: dict[str, ModelEntry] | None = None,
) -> list[str]:
    """Deterministic candidate selection over ELIGIBLE models only.

    Anchors (Ridge, RandomForestRegressor) are included first when eligible.
    Any remaining slots up to target_count are filled from eligible
    boosting-family models not already selected, ordered by the frozen
    Fallback Order (runtime_cost_tier ascending, then model_id alphabetical).
    Never trains anything; returns model_ids only. May return fewer than
    target_count if not enough eligible models exist -- the Pre-Training
    Gate is responsible for judging sufficiency, not this function.
    """
    source = registry if registry is not None else MODEL_REGISTRY

    selected: list[str] = [
        model_id
        for model_id in MANDATORY_ANCHOR_MODEL_IDS
        if model_id in eligibility_results and eligibility_results[model_id].eligible
    ]

    remaining_slots = target_count - len(selected)
    if remaining_slots > 0:
        boosting_candidates = sorted(
            (
                entry
                for model_id, entry in source.items()
                if entry.family == FAMILY_BOOSTING_ENSEMBLE
                and model_id not in selected
                and model_id in eligibility_results
                and eligibility_results[model_id].eligible
            ),
            key=_fallback_order_key,
        )
        selected.extend(entry.model_id for entry in boosting_candidates[:remaining_slots])

    return selected


def evaluate_pre_training_gate(
    train_row_count: int,
    problem_type: str = PROBLEM_TYPE_REGRESSION,
    registry: dict[str, ModelEntry] | None = None,
) -> PreTrainingGateResult:
    """Checkpoint 1: PRE-TRAINING CANDIDATE AVAILABILITY / VIABILITY CHECK.

    Answers "are enough eligible candidates available to satisfy the
    deterministic candidate-count policy?" -- nothing else. Never trains,
    never touches ml_trainer.py, never computes or reads any metric. This is
    NOT the Post-Training Comparison Viability Check (different question,
    different reason_code, a later slice's responsibility).
    """
    eligibility_results = compute_eligibility(problem_type=problem_type, registry=registry)
    target = target_candidate_count(train_row_count)
    selected = select_candidates(eligibility_results, target, registry=registry)

    if len(selected) >= target:
        return PreTrainingGateResult(
            status="PASS",
            target_candidate_count=target,
            selected_model_ids=tuple(selected[:target]),
            eligibility_results=eligibility_results,
            reason_code=None,
            reason_params=None,
        )

    eligible_model_ids = sorted(model_id for model_id, r in eligibility_results.items() if r.eligible)
    return PreTrainingGateResult(
        status="FAIL",
        target_candidate_count=target,
        selected_model_ids=tuple(selected),
        eligibility_results=eligibility_results,
        reason_code=INSUFFICIENT_ELIGIBLE_CANDIDATES,
        reason_params={
            "required_count": target,
            "available_count": len(eligible_model_ids),
            "available_model_ids": eligible_model_ids,
        },
    )
