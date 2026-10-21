# churnlab-2026

Churn prediction project built on the IBM Telco Customer Churn dataset (ESILV MLOps course).

## Business question

A telecommunications company wants to identify the customers who are likely to leave its service,
so that retention actions can be targeted. This is a binary classification task.

- **Target column:** `Churn` (`Yes` / `No`)
- `customerID` is an identifier, not a model feature. `Churn` must never be used as an input.

## Environment setup

- **Python:** 3.11 (3.12 is also accepted by `pyproject.toml`; the project is pinned to 3.11 in `.python-version`)
- **Package manager:** [uv](https://docs.astral.sh/uv/). The lock file was produced with uv 0.13.4.

From the project root:

```sh
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"   # expected: 0.1.0
uv run python scripts/record_environment.py                         # writes evidence/tp01/environment.json
```

`uv sync --locked` fails instead of silently updating `uv.lock` if the lock no longer matches `pyproject.toml`.
`.venv/` is local and never committed; recreate it with the command above.

> **Windows note:** if uv-managed Python fails with exit code `0xc0e90002` (Smart App Control),
> install the official Python 3.11 from python.org and set `UV_PYTHON_PREFERENCE=only-system`.

## Data

| Property | Value |
|---|---|
| Source | IBM `telco-customer-churn-on-icp4d`, pinned to commit `1fd6fd7` |
| URL | https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv |
| Local file | `data/raw/telco.csv` (not committed) |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Shape | 7,043 rows × 21 columns |

Download (if absent), verify the hash and inspect:

```sh
uv run python scripts/download_data.py    # fetches the CSV, checks the hash, writes data/raw/telco.source.json
uv run python scripts/inspect_telco.py    # fails on hash mismatch, writes reports/tp01_data_overview.json
```

Expected inspection output: 5,174 `No` / 1,869 `Yes` (≈ 26.54 % churn), 0 duplicate customer IDs,
11 whitespace-only `TotalCharges` values. Do not edit the raw CSV: any change breaks the hash check.

A filename is not a data version: the SHA-256 identifies the exact bytes used. It does not prove that the labels are correct.

### What this dataset is not

This is a **single customer snapshot**: one row per customer, with no observation date for each field
and no record of when a customer churned. It is **not** an event-time log (dated customer events) and
**not** an intervention dataset (randomized retention offers with treatment assignment).
It therefore supports predicting churn from current customer attributes, but not time-based
validation or causal statements about the effect of a retention action.

## Project layout

| Path | Purpose |
|---|---|
| `src/churnlab/` | Importable data, training and serving code |
| `scripts/` | Executable workflow commands (download, inspection, environment record) |
| `tests/` | Small behavior checks (`uv run pytest`) |
| `data/raw/` | Original input snapshot and its source metadata (`telco.source.json` is committed, the CSV is not) |
| `data/processed/` | Generated splits (later TPs) |
| `artifacts/` | Generated model files (later TPs, not committed) |
| `reports/` | Generated reports (not committed; copy small reports to `evidence/`) |
| `evidence/` | Non-sensitive evidence submitted for each TP |
| `k8s/`, `pipelines/` | Deployment and orchestration (later TPs) |
