# CLAUDE.md

## Project Purpose

Retail AI Intelligence & Forecasting Workflow Platform — a CrewAI Flow application for retail sales analysis and forecasting on the approved "Retail Data Analytics" Kaggle dataset.

LLM agents interpret and narrate precomputed evidence. Deterministic Python code performs data processing, validation, metrics, feature engineering, training, evaluation, and model-selection control.

## Git Workflow

feature/* or fix/* → dev → staging validation → main → production

Rules:
- Never develop directly on `main`.
- New implementation work must branch from `dev` unless explicitly instructed otherwise.
- `dev` is the current integration branch.
- Before any significant code change, commit, merge, rebase, or branch cleanup, verify the current branch and working-tree state.
- Do not merge feature/fix work directly into `main`.
- Do not delete a feature/fix branch until its work is merged, verified, and safely contained in the target branch.

## Test Baseline

Run:

    python -m pytest tests/ -v

Current verified Phase 2 baseline on `dev`:
40 tests passing.

Rules:
- Do not weaken, skip, or delete tests merely to make a change pass.
- When changing correctness-sensitive behavior, add or update tests that protect the behavior, not just implementation details.

## Dataset Source of Truth

Approved dataset:

Retail Data Analytics

Approved Kaggle listing:

https://www.kaggle.com/datasets/manjeetsingh/retaildataset

Project-recorded uploader:
Manjeet Singh

Project-recorded license:
CC0: Public Domain

Primary source of truth:
`data/dataset_manifest.json`

Preserve:
- provenance
- exact listing identity
- schema
- expected source files
- join keys
- license record

Never silently substitute another similarly named Walmart/Retail dataset or a different version.

Project-recorded provenance must not be upgraded into an independently verified external fact unless it has actually been re-verified.

## Raw Data

`data/raw/` is immutable source data.

Never modify raw CSV files to solve analytical, modeling, validation, or reporting problems.

Fix processing, validation, modeling eligibility, feature engineering, or reporting logic instead.

## Analytical Guardrails

- Deterministic Python code controls validation, metrics, feature engineering, training, evaluation, and model-selection execution.
- LLM agents may reason about and explain computed evidence but must not fabricate statistics, metrics, business impact, or validation outcomes.
- Agents cannot convert a deterministic contract FAIL into PASS.
- Association is not causation.
- Unknown business semantics must remain documented as unknown rather than being replaced by plausible assumptions.
- Do not infer undocumented business meaning from department codes, negative sales values, or anonymous features.

## MarkDown Policy

- `MarkDown1-5` source grain is `Store + Date`, not `Store + Dept + Date`.
- Missing MarkDown values do not automatically mean "no promotion"; describe them as unrecorded/unavailable unless authoritative evidence establishes more.
- After the join to Sales, one Store-Date MarkDown value may repeat across multiple departments.
- Never count those repeated department rows as independent MarkDown events.

## Weekly_Sales Policy

- Zero and negative values remain visible in clean/descriptive data unless a separately approved policy says otherwise.
- Do not automatically delete them.
- Do not claim negative sales represent returns, refunds, or corrections without authoritative evidence.

## Modeling Eligibility

Single source of truth:

`src/services/modeling_eligibility.py`

Current policy:
- Department 47: included in raw/clean/descriptive/EDA data, excluded only from predictive training.
- This is a modeling-eligibility decision, not data cleaning.
- Departments 78, 18, and 54 remain modeling-eligible and are documented/watchlisted.
- Do not create exclusion rules from surface traits such as round values, symmetric ±X values, or mere presence of negatives.

## Week-over-Week Rule

"Week-over-week" means observations exactly 7 calendar days apart within the same `Store + Dept` series.

Never assume adjacent dataframe rows are consecutive weeks.

When the previous value is 0:
- preserve absolute change
- percentage change = N/A
- never emit `inf%` or `-inf%`

## Agent Security Boundary

Agents must never receive:
- database passwords
- connection strings
- service-role keys
- API secrets
- authentication secrets
- unrestricted arbitrary SQL capability

Credentials belong only in deterministic backend/integration layers.

## Architecture Boundary

Do not implement future roadmap phases opportunistically.

Completed:
- Phase 1 — Architecture Specification
- Phase 2 — Correctness Remediation, merged into `dev`

Future capabilities require explicit scoping before implementation, including:
- Supabase data integration
- Supabase Auth
- generic external data connectors
- expanded Analyst Crew
- expanded Scientist Crew
- dynamic 2–3 ML model selection
- frontend/backend separation
- Railway deployment
- generic-dataset architecture

## Known Deferred Issues

Keep these visible until explicitly resolved:

1. `compute_micro_inspection()` holiday-spike logic can fail when input contains only one `IsHoliday` category.
2. The rendered `model_card.md` "Modeling Population" section lacks direct rendering coverage.
3. A genuine full-dataset end-to-end ML integration run with real model training and artifact generation has not yet been completed.

## Mandatory Production Checkpoint

After the future dynamic 2–3 model-selection architecture is implemented and before Production deployment, run a genuine full-dataset end-to-end integration workflow including:

- ingestion
- validation
- Analyst Crew
- modeling eligibility
- feature engineering
- real training of all selected candidate models
- model comparison
- final model selection
- serialization
- prediction path
- artifact generation
- model card
- evaluation artifacts
- run metadata
- frontend/backend integration where applicable

Unit tests, reduced-data runs, or feature-engineering-only verification are not substitutes for this checkpoint.
