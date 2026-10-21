# churnlab-2026

Telco customer churn prediction project (IBM Telco Customer Churn dataset).

## Setup

```sh
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"
```

## Data

```sh
uv run python scripts/download_data.py
uv run python scripts/inspect_telco.py
```
