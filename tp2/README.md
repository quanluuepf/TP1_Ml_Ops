# churnlab - TP02: reusable training pipeline

Churn prediction on the IBM Telco Customer Churn snapshot. TP02 turns a first working model into a
reusable scikit-learn pipeline: **working model → preprocessing pipeline → fair comparison → reusable artifact**.

- **Positive class:** `Churn = Yes`, encoded as `1`.
- `customerID` is only used for identification and split checks. It is never a model input, and neither is `Churn`.

## Setup

Run all commands from `tp2/`:

```sh
uv sync --locked
uv run python scripts/download_data.py   # only if data/raw/telco.csv is missing; checks SHA-256 16320c9c…5e91
```

Python 3.11 (see `.python-version`). On Windows with Smart App Control, use the official python.org
interpreter and set `UV_PYTHON_PREFERENCE=only-system` (see `tp1/README.md`).

## Workflow (terminal entry points)

| Step | Command | Output |
|---|---|---|
| 1. First model (2 features + baseline) | `uv run python scripts/first_model.py` | `reports/tp02_first_model.json` |
| 2-4. Train, compare, select, save | `uv run python scripts/train.py --config configs/train.json` | `reports/tp02_comparison.{csv,json}`, `artifacts/churn_pipeline.joblib`, `artifacts/churn_pipeline.metadata.json`, `data/processed/split_allocations.csv` |
| 4. Inference in a fresh process | `uv run python scripts/predict.py --input data/examples/new_customers.csv` | Scores and Yes/No predictions, plus an agreement check against the scores recorded at training time |
| 5. Reliability checks | `uv run pytest -v` | 9 tests |

Training is deterministic (seed 42 for the split and for the tree models). All settings are in `configs/train.json`.

## Data and split

- Source: `data/raw/telco.csv`. The file is never modified, and `load_raw` refuses to run if its SHA-256 differs.
- The split is stratified on `Churn` and made by customer: 60/20/20 → train 4,225 / validation 1,409 / test 1,409.
  Each customer's split is written to `data/processed/split_allocations.csv`, and the hash of that file is recorded in the metadata.
- **The test set is reserved.** It is created and recorded, but no script evaluates it. Model selection uses validation only.

## Features and preprocessing

| Numeric | Categorical |
|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges` | `Contract`, `InternetService`, `PaymentMethod` |

**Input parsing (stateless, `churnlab.data.parse_features`).** The same rules apply at training and at inference:
- a missing required column raises `KeyError`;
- blank or whitespace-only values become `NaN`: these are allowed missing values (11 rows of `TotalCharges` in the raw data);
- numeric columns are converted with `pandas.to_numeric(errors="raise")`, so malformed text such as `"abc"` **fails**
  instead of silently becoming `NaN`.

**Pipeline (`churnlab.pipeline.build_pipeline`).** The builder returns a new, unfitted object. It does not read data and does not train.

```
Pipeline
├── preprocessor: ColumnTransformer
│   ├── num: SimpleImputer(median) → StandardScaler
│   └── cat: SimpleImputer(constant="__missing__") → OneHotEncoder(handle_unknown="ignore")
└── classifier
```

- A single `fit` call on the training rows learns the medians, the scaling statistics, the category list and the classifier.
  Validation data and new records reuse these fitted transformations; nothing is refitted.
- **Unknown category at inference:** it is encoded as all zeros, so the prediction falls back on the other features.
  This keeps the service running but **does not make the prediction reliable** for that customer.
- **Missing categorical value:** it becomes `__missing__`. Training had no missing categories, so this level is unknown
  to the encoder and also gets all zeros.

## Model comparison (validation set, threshold 0.5)

The primary metric was **chosen before the comparison: average precision (AP)** on the positive-class probability.
Churners are the minority (26.5 %), so accuracy is misleading: the majority baseline already reaches 73.5 % accuracy
while catching no churners. AP summarizes the precision/recall trade-off across all thresholds.

| Model | AP | Precision | Recall | F1 | Accuracy | TN | FP | FN | TP | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.265 | 0.000 | 0.000 | 0.000 | 0.735 | 1035 | 0 | 374 | 0 | 0.01 |
| Logistic regression | 0.647 | 0.664 | 0.529 | 0.589 | 0.804 | 935 | 100 | 176 | 198 | 0.02 |
| Decision tree (depth 5, leaf 20) | 0.617 | 0.636 | 0.602 | 0.618 | 0.803 | 906 | 129 | 149 | 225 | 0.02 |
| **Random forest (300 trees, depth 8, leaf 10)** | **0.664** | 0.653 | 0.484 | 0.556 | 0.795 | 939 | 96 | 193 | 181 | 0.37 |

All candidates use the same 6 features, the same split and the same metric code. Fit times were measured on a single
laptop and are only indicative. Confusion matrices have rows = true class [0, 1] and columns = predicted class [0, 1].

### Selection note

- **Rule (in `configs/train.json`, set before comparing):** highest validation AP among learned models. If another model is
  within 0.005 AP of the best, keep the simpler one (earlier in the config order: logistic < tree < forest).
- **Decision: random forest.** AP 0.664 vs 0.647 for logistic regression; the gap (0.017) is larger than the tolerance.
- **Caveats:**
  - The gap is small and comes from a single validation split, so logistic regression remains a reasonable and simpler alternative.
  - At the 0.5 threshold the forest has the **lowest recall** (48 %). If the team's priority is catching churners,
    the threshold must be chosen on validation for that goal; this changes the decision policy, not the trained model.
  - The probabilities have not been calibrated: a score of 0.8 does not mean an 80 % risk.
- **First model vs this comparison:** the first model (`tenure` + `MonthlyCharges`, logistic regression) reached 0.783 validation
  accuracy against 0.735 for the baseline, and caught 153 of the 374 churners. Its results cannot be compared directly with the table
  above to judge the algorithm, because the feature set also changed.

## Saved artifact

`artifacts/churn_pipeline.joblib` contains the **fitted preprocessing and classifier together**. Saving only the classifier
would lose the medians, the scaling statistics and the category list it depends on.

`artifacts/churn_pipeline.metadata.json` records:
- the hashes of the artifact, the source CSV, the split file and the config;
- the seed and split sizes, the feature roles and the positive class;
- the threshold, the preprocessing settings and the learned medians;
- the model parameters and validation metrics;
- the Git commit and whether the working tree had uncommitted changes, plus the Python and library versions;
- reference scores for `data/examples/new_customers.csv`.

`scripts/predict.py` checks the artifact hash against this record before loading. Only load joblib files from a trusted source:
loading one can execute code. Reload with compatible versions (scikit-learn 1.9.1, joblib, numpy and pandas as recorded).

The artifact (~4 MB) is git-ignored. It can be regenerated by re-running `scripts/train.py`; the hash in the metadata identifies the exact file, or shared through a
release or course storage. A copy of its metadata is in `evidence/tp02/`.

## Reliability evidence (`tests/test_reliability.py`, `evidence/tp02/`)

| Check | Result |
|---|---|
| Customer IDs do not overlap across train, validation and test; the split is identical when re-run | passed |
| The learned imputer medians equal the **training** medians (tenure 29, MonthlyCharges 70.55, TotalCharges 1373.625) and differ from the full-dataset medians | passed |
| `predict_proba` leaves the imputer medians and scaler means unchanged | passed |
| Blank `TotalCharges` becomes `NaN` and is imputed; an unseen `Contract` value is encoded as all zeros; the score stays in [0, 1] | passed |
| Malformed numeric text raises `ValueError` | passed |
| A missing `Contract` column raises an error, both in parsing and in the pipeline | passed |
| Reload in a separate Python process gives the same scores (atol 1e-9) | passed |
| `scripts/predict.py` (no labels, no `fit`) agrees with the scores recorded at training: max difference 2.8e-17 | AGREE |

## Limitations and next steps

- Only one validation split was used, with no confidence interval; the small gap between models is uncertain.
- No hyperparameter search was done, and probabilities are not calibrated.
- The 0.5 threshold is a convention, not a business decision.
- This is a snapshot dataset: there is no time-based validation and no causal claim about retention actions.
- The test set has not been used yet. It should be evaluated once, on the frozen selected pipeline.
- Experiment tracking and model versioning come in TP03.

## Peer review

- Reviewer: `<name>`
- Issue found while reproducing: `<issue>`
- Resolution: `<resolution>`
