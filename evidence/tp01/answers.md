# TP01 - Answers and evidence

## D2. What has been reproduced

| Item | Evidence produced in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit `<COMMIT_HASH>` on `main` (scripts `record_environment.py`, `inspect_telco.py`, `download_data.py`, package `src/churnlab/`) | Uncommitted edits are not identified by that commit |
| Package selections | `pyproject.toml` (`requires-python >=3.11,<3.13`, numpy/pandas/scikit-learn) and `uv.lock` (20 resolved packages), unchanged after recreation | Platform/interpreter selection can affect installations: the lock was generated with uv 0.13.4 on Windows with CPython 3.11.9, not with the course's uv 0.12.23 |
| Installed environment | `environment-before.json` and `environment.json`: identical after deleting and recreating `.venv` with `uv sync --locked` (Python 3.11.9, churnlab 0.1.0, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1) | This does not capture every OS or hardware detail (CPU, BLAS libraries, drivers, system DLLs, locale) |
| Source data | `data/raw/telco.source.json`: commit-pinned IBM URL and SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`; `data-overview.json` identical before/after (7,043 rows, 21 columns, 5,174 No / 1,869 Yes, 0 duplicate IDs, 11 blank `TotalCharges`) | This does not establish label quality or absence of leakage |
| Trained model | Not produced in TP01 | Requires later training/configuration/artifact evidence |


