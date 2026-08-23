# Retail AI Intelligence & Forecasting Workflow Platform

A CrewAI Flow application for retail sales analysis and forecasting, built on
the Kaggle **"Retail Data Analytics"** dataset. The system pairs LLM-driven
narration (CrewAI agents) with **fully deterministic** statistics, validation,
feature engineering, and model selection — the LLM never decides whether
data passes validation, never trains a model, and never computes a metric.

## Contents

- [Dataset](#dataset)
- [Architecture](#architecture)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Running the workflow from the CLI](#running-the-workflow-from-the-cli)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Roles & credentials](#roles--credentials)
- [Design decisions & guardrails](#design-decisions--guardrails)

## Dataset

This project uses the **"Retail Data Analytics"** dataset from Kaggle:

- **Dataset**: Retail Data Analytics
- **Uploader**: [Manjeet Singh](https://www.kaggle.com/manjeetsingh)
- **URL**: https://www.kaggle.com/datasets/manjeetsingh/retaildataset
- **License**: CC0: Public Domain (as declared on the Kaggle listing)

## Architecture

```
data/raw/{stores,features,train}.csv
        │
        ▼
 Data Ingestion (deterministic join)
        │
        ▼
 Data Analyst Crew ──► clean_data.csv, eda_report.html,
   (3 CrewAI agents        insights.md, dataset_contract.json
    narrate precomputed
    deterministic stats)
        │
        ▼
 Contract Validation Gate (pure Python, PASS/FAIL)
        │
   ┌────┴────┐
  FAIL      PASS
   │          │
 HALT         ▼
        Feature Pipeline (lags, rolling stats, imputation, one-hot)
        Ridge vs RandomForestRegressor (chronological 80/20 split)
        Data Scientist Crew ──► evaluation_report.md, model_card.md
        selected_model.joblib, evaluation_report.json
        │
        ▼
 Finalize (run_metadata.json, status = COMPLETED)
```

All of this is orchestrated by a stateful **CrewAI Flow**
(`src/flows/retail_flow.py`, class `RetailFlow`) with one artifact directory
per run under `artifacts/<run_id>/`.

**Deterministic guardrail:** every number that ends up in `insights.md`,
`eda_report.html`, `dataset_contract.json`, `evaluation_report.md`, or
`model_card.md` is computed first in plain Python/pandas/scikit-learn. The
CrewAI agents are only ever given those precomputed numbers as task context
and asked to narrate them — they cannot alter a statistic, decide a
validation outcome, or change which model gets selected. If no LLM
credentials are configured, the crews fall back to deterministic-only output
(no narration text) rather than failing the run.

**Micro-inspection policy:** the Data Profiler and Business Intelligence
agents are deliberately fed record-level findings, not just global
aggregates — explicit zero/negative sales rows, per-store holiday spikes,
IQR-based markdown outliers, and week-over-week anomalies, each tied to a
specific `(Store, Dept, Date)`. These are always written into `insights.md`
and `eda_report.html`, independent of what any LLM narrative says.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The raw dataset is already included under `data/raw/` (`stores.csv`,
`features.csv`, `train.csv` — see `data/dataset_manifest.json` for
provenance). No download step is required.

### LLM credentials (optional)

The Analyst and Scientist crews use CrewAI's default LLM (OpenAI via
LiteLLM). To get narrative text in `insights.md`, `evaluation_report.md`,
and `model_card.md`, set an API key before running:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
# macOS/Linux
export OPENAI_API_KEY="sk-..."
```

Without a key, the workflow still completes end-to-end — the crews catch the
LLM failure, skip narration, and every deterministic artifact (contract,
EDA report, model metrics, model card) is still generated normally. Expect
the run to take longer without a key configured, since CrewAI retries the
LLM call several times before giving up.

## Running the app

```bash
streamlit run app.py
```

Log in with one of the seeded accounts (see [Roles & credentials](#roles--credentials)).

- **Admin**: click "Start New Run" to kick off the full CrewAI Flow in the
  background; live logs stream into the page every ~2 seconds. Past runs are
  listed with their status, and any run that failed the contract validation
  gate can be expanded to see the exact errors.
- **Descriptive Analytics** ("What Has Happened?"): KPI cards, monthly sales
  trend, a markdown/sales correlation heatmap, the analyst crew's
  `insights.md`, and the full `eda_report.html` embedded in the page.
- **Predictive Intelligence** ("What Will Happen?"): the Ridge vs
  RandomForest comparison table, the selected model's metrics, an
  actual-vs-predicted forecast chart over the chronological hold-out weeks,
  and the `model_card.md` / `evaluation_report.md` viewers.

Only pages the logged-in role has permission for (per `config.yaml` →
`roles`) appear in the sidebar.

**Note on model size:** `RandomForestRegressor(n_estimators=100,
random_state=42)` trained on the *full* ~421K-row dataset with unconstrained
tree depth (as specified) produces a `selected_model.joblib` several
gigabytes in size. Loading it in the Predictive Intelligence page requires
enough free RAM; on constrained machines, a small pre-generated UI demo
fixture is available at `fixtures/ui-smoketest/` (built the same way as
`tests/test_ml_pipeline.py`'s `test_full_pipeline_on_real_ingested_data`,
filtered to a few stores). **This fixture intentionally lives outside
`artifacts/`** so it can never be discovered as a genuine workflow run by
`app.py`'s run selector (see [Known Data Semantics & Modeling
Policies](#known-data-semantics--modeling-policies)) — do not copy it into
`artifacts/`.

## Running the workflow from the CLI

```python
from src.flows.retail_flow import run_retail_flow

final_state = run_retail_flow(raw_dir="data/raw", artifacts_root="artifacts")
print(final_state.status)          # COMPLETED or FAILED
print(final_state.run_dir)         # artifacts/<run_id>
```

## Testing

```bash
pytest tests/ -v
```

40 tests across five files:

- `tests/test_validator.py` (13 tests) — deterministic ingestion join
  correctness, `contract_validator` PASS/FAIL behavior for missing columns,
  null primary keys/target, duplicate primary keys, out-of-bounds values,
  wrong dtypes, and the non-blocking negative/zero-target WARNING behavior.
- `tests/test_ml_pipeline.py` (9 tests) — lag/rolling feature correctness
  and leakage prevention, markdown imputation, one-hot encoding, the
  chronological split boundary, training both candidates and selecting the
  lowest-RMSE one, `.joblib` serialization, and an end-to-end run against
  the real ingested dataset.
- `tests/test_analyst_crew.py` (6 tests) — MarkDown Store-Date grain
  deduplication, the missingness indicator columns, and the week-over-week
  zero-denominator / non-consecutive-week-gap corrections.
- `tests/test_modeling_eligibility.py` (6 tests) — Department 47's exclusion
  from the modeling population while remaining in descriptive data,
  Departments 78/18/54 remaining modeling-eligible, and traceability of the
  exclusion summary.
- `tests/test_run_discovery.py` (6 tests) — real run-id recognition and
  fixture exclusion from `list_runs()`.

Both `test_validator.py` and `test_ml_pipeline.py` run against the real CSVs
in `data/raw/`, not synthetic mocks-only fixtures, so a passing suite
reflects the actual dataset.

## Project structure

```
config.yaml                      # roles, credentials (sha256 hashes), paths
requirements.txt
data/
  dataset_manifest.json          # Kaggle provenance, schema, license
  raw/{stores,features,train}.csv
src/
  services/
    data_ingestion.py            # deterministic joins + null cleaning
    contract_validator.py        # pure-Python PASS/FAIL validation gate + warnings
    feature_pipeline.py          # lags, rolling stats, imputation, one-hot
    ml_trainer.py                # chronological split, Ridge vs RF, selection
    modeling_eligibility.py      # descriptive-vs-modeling population policy (Dept 47)
    run_registry.py              # real-run-id recognition for app.py's run selector
  agents/
    analyst_crew.py              # Data Profiler / BI / Contract Architect
    scientist_crew.py            # Feature Strategy / Model Eval / Governance
  flows/
    retail_flow.py               # RetailFlow: full run lifecycle + gate
  utils/
    auth.py                      # config.yaml-based Streamlit auth
fixtures/
  ui-smoketest/                  # small pre-generated fixture for fast UI demos
                                  # (outside artifacts/ so it can't masquerade as a real run)
tests/
  test_validator.py
  test_ml_pipeline.py
  test_analyst_crew.py
  test_modeling_eligibility.py
  test_run_discovery.py
app.py                           # Streamlit multi-role dashboard
```

## Roles & credentials

Seeded in `config.yaml` (passwords are the username + `123`, hashed with
SHA-256 — change these before any real deployment):

| Username     | Password       | Role       | Access                              |
|--------------|----------------|------------|--------------------------------------|
| `admin`      | `admin123`     | Admin      | Run workflow, both dashboards, logs |
| `analyst`    | `analyst123`   | Analyst    | Descriptive Analytics only          |
| `scientist`  | `scientist123` | Scientist  | Predictive Intelligence only        |
| `business`   | `business123`  | Business   | Both dashboards                     |

## Design decisions & guardrails

- **No dataset substitution.** Only `stores.csv`, `features.csv`,
  `train.csv` from the Kaggle "Retail Data Analytics" listing are read
  (`src/services/data_ingestion.py` fails fast if any is missing).
- **Exactly two candidate models.** `Ridge(alpha=1.0)` and
  `RandomForestRegressor(n_estimators=100, random_state=42)` — no other
  model family is trained or considered.
- **Leakage prevention.** Lag and rolling features are built with
  `shift(1)` before any window aggregation, so a row's features only ever
  see data strictly before that row's own date.
- **Chronological split, not random.** Train/test is split on a single
  global date cutoff (first 80% of the calendar timeline vs. the final
  20%), never a random shuffle.
- **The validation gate halts the flow.** If `contract_validator` returns
  `FAIL`, `RetailFlow` routes to `handle_validation_failure`, writes
  `run_metadata.json` with `status: FAILED` and the exact errors, and never
  runs the Scientist stage.

## Known Data Semantics & Modeling Policies

Corrections made during the Phase 2 correctness remediation (2026-08-17):

- **Fixture vs. real runs.** `app.py`'s `list_runs()` (via
  `src/services/run_registry.py`) only recognizes directories whose name
  matches the exact pattern `RetailFlow.initiate_flow()` generates
  (`YYYYMMDDTHHMMSSZ-<8 hex chars>`). The hand-built UI demo fixture lives at
  `fixtures/ui-smoketest/`, outside `artifacts/`, and can never be surfaced
  as the latest/genuine analytical result regardless of what its
  `run_metadata.json` claims.
- **MarkDown missing values are not "no promotion".** The Kaggle source
  documents only that MarkDown data is unavailable before Nov 2011 and not
  recorded for all stores at all times — it does not state what a missing
  value means for the underlying business event. `data/dataset_manifest.json`
  and all generated reports describe a missing MarkDown value as
  "not recorded/unavailable", never as a confirmed absence of a promotion.
  `feature_pipeline.engineer_features()` still zero-fills MarkDown nulls for
  model training, but first records a `MarkDown{N}_was_missing` indicator
  column so "unrecorded" and "observed zero" remain distinguishable.
- **MarkDown grain.** MarkDown1-5 are Store+Date-level features; Weekly_Sales
  is Store+Dept+Date-level. After the ingestion join, one Store-Date markdown
  value is repeated once per department present that week.
  `analyst_crew.compute_micro_inspection()` deduplicates to (Store, Date)
  before computing markdown outliers so a single markdown observation is
  never reported as N independent department-level events; the markdown/sales
  comparison in `compute_business_intelligence()` still operates at the
  Store+Dept+Date sales grain (that comparison is valid) but now carries an
  explicit `grain_note` explaining the mismatch, surfaced in `insights.md`,
  `eda_report.html`, and the Streamlit dashboard.
- **Zero and negative Weekly_Sales are retained, not deleted.** The source
  documentation does not explain what a negative Weekly_Sales value
  represents (not confirmed as returns, refunds, or corrections) or whether
  zero is invalid. Both are kept in `clean_data.csv`, EDA, and (subject to the
  modeling-eligibility policy below) the modeling population.
  `contract_validator.validate_dataframe()` now emits non-blocking
  **warnings** (not failures) when negative or zero target values are
  present, and the Analyst Crew's LLM task prompts are explicitly instructed
  not to assert an unverified cause for either.
- **Department 47 modeling eligibility.** `src/services/modeling_eligibility.py`
  is the single source of truth for the difference between "included in
  descriptive/EDA data" (always true, every department) and "included in the
  predictive modeling population". Department 47 is excluded from the
  modeling population only — it remains in `clean_data.csv`, `insights.md`,
  and `eda_report.html` — because of unresolved business semantics (no
  authoritative source explains what Department 47 represents) combined with
  verified structural target behavior (>100x baseline negative-sales rate
  across most of the stores that carry it, and calendar-corrected true-7-day
  week-over-week transitions showing persistent instability, not ordinary
  stability). This is a modeling-eligibility decision, not a data-cleaning
  decision, and is fully traceable via `training_result["modeling_population"]`
  → `evaluation_report.json` / `run_metadata.json` / `model_card.md`'s
  "Modeling Population" section. Departments 78, 18, and 54 show milder,
  non-exclusionary anomaly rates and remain modeling-eligible; they are
  documented (not filtered) via `DOCUMENTED_WATCHLIST` in the same module.
- **Week-over-week calculations never report `inf%`/`-inf%`.** When the
  previous week's sales were exactly zero, a percentage change is
  mathematically undefined; `compute_micro_inspection()` reports those cases
  separately (absolute change, explicit "previous value is zero" reason)
  instead of a division-by-zero infinity. Week-over-week comparisons are only
  computed for Store/Dept observations whose dates are exactly 7 days apart —
  a larger calendar gap is excluded rather than mislabeled as a one-week
  change.
