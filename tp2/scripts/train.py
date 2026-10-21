"""Train, compare on validation, select and save the whole fitted pipeline.

Usage (from tp2/):  uv run python scripts/train.py [--config configs/train.json]
The test split is created and recorded but never evaluated here.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from churnlab.data import (
    CATEGORICAL_FEATURES, EXPECTED_SHA256, FEATURES, ID_COLUMN, NUMERIC_FEATURES,
    POSITIVE_LABEL, RAW_PATH, TARGET, load_raw, parse_features, parse_target, sha256_file,
)
from churnlab.evaluate import evaluate_model, positive_scores
from churnlab.pipeline import build_pipeline
from churnlab.split import allocation_table, check_disjoint, split_customers

ESTIMATORS = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "DummyClassifier": DummyClassifier,
}
EXAMPLES_PATH = Path("data/examples/new_customers.csv")
ARTIFACT_PATH = Path("artifacts/churn_pipeline.joblib")
METADATA_PATH = Path("artifacts/churn_pipeline.metadata.json")
ALLOCATION_PATH = Path("data/processed/split_allocations.csv")


def git_identity() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "."], text=True).strip())
        return {"commit": commit, "dirty_working_tree": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_working_tree": None}


def select_model(results: list[dict], metric: str, tolerance: float) -> str:
    """Best metric wins; within tolerance, the earlier (simpler) config entry wins."""
    learned = [r for r in results if not r["is_baseline"]]
    best = max(r[metric] for r in learned)
    return next(r["model"] for r in learned if r[metric] >= best - tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed, threshold = config["seed"], config["threshold"]

    raw = load_raw()
    train_df, valid_df, test_df = split_customers(raw, seed, config["valid_size"], config["test_size"])
    check_disjoint(train_df, valid_df, test_df)
    ALLOCATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    allocation_table(train_df, valid_df, test_df).to_csv(ALLOCATION_PATH, index=False)

    X_train, y_train = parse_features(train_df), parse_target(train_df)
    X_valid, y_valid = parse_features(valid_df), parse_target(valid_df)

    candidates = {name: (spec, False) for name, spec in config["models"].items()}
    candidates["majority_baseline"] = (config["baseline"], True)

    results, fitted = [], {}
    for name, (spec, is_baseline) in candidates.items():
        pipe = build_pipeline(ESTIMATORS[spec["class"]](**spec["params"]))
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        metrics = evaluate_model(pipe, X_valid, y_valid, threshold)
        results.append({
            "model": name, "is_baseline": is_baseline, "estimator": spec["class"],
            "params": json.dumps(spec["params"], sort_keys=True),
            "fit_seconds": round(fit_seconds, 4), **metrics,
        })
        fitted[name] = pipe
        print(f"{name:20s} AP={metrics['average_precision']:.4f} "
              f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} fit={fit_seconds:.2f}s")

    selection = config["selection"]
    selected = select_model(results, selection["metric"], selection["tie_tolerance"])
    selected_pipeline = fitted[selected]
    print(f"Selected: {selected}")

    Path("reports").mkdir(exist_ok=True)
    table = pd.DataFrame(results)
    table.to_csv("reports/tp02_comparison.csv", index=False)
    Path("reports/tp02_comparison.json").write_text(
        json.dumps({"split": "validation", "threshold": threshold,
                    "selection_metric": selection["metric"], "selected": selected,
                    "results": results}, indent=2),
        encoding="utf-8",
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, ARTIFACT_PATH)

    examples = pd.read_csv(EXAMPLES_PATH, dtype=str, keep_default_na=False)
    reference_scores = positive_scores(selected_pipeline, parse_features(examples))

    preprocessor = selected_pipeline.named_steps["preprocessor"]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {"path": ARTIFACT_PATH.as_posix(), "sha256": sha256_file(ARTIFACT_PATH),
                     "format": "joblib (trusted source only)"},
        "data": {"source_file": RAW_PATH.as_posix(), "sha256": EXPECTED_SHA256,
                 "rows": int(len(raw))},
        "split": {"seed": seed, "sizes": {"train": len(train_df), "valid": len(valid_df),
                                          "test": len(test_df)},
                  "stratified_on": TARGET, "allocation_file": ALLOCATION_PATH.as_posix(),
                  "allocation_sha256": sha256_file(ALLOCATION_PATH),
                  "test_evaluated": False},
        "features": {"id_column": ID_COLUMN, "target": TARGET, "numeric": NUMERIC_FEATURES,
                     "categorical": CATEGORICAL_FEATURES, "input_order": FEATURES},
        "positive_class": {"label": POSITIVE_LABEL, "encoded_as": 1},
        "threshold": threshold,
        "preprocessing": {
            "parsing": "blank/whitespace -> NaN; numeric via pandas.to_numeric(errors='raise')",
            "numeric": "SimpleImputer(median) -> StandardScaler",
            "categorical": "SimpleImputer(constant='__missing__') -> OneHotEncoder(handle_unknown='ignore')",
            "learned_numeric_medians": dict(zip(
                NUMERIC_FEATURES,
                preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_.tolist())),
        },
        "selected_model": {"name": selected, **{k: v for k, v in config["models"][selected].items()},
                           "validation_metrics": next(r for r in results if r["model"] == selected)},
        "selection": selection,
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "code": git_identity(),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {p: version(p) for p in
                                     ["churnlab", "scikit-learn", "pandas", "numpy", "joblib"]}},
        "reference_predictions": {"examples_file": EXAMPLES_PATH.as_posix(),
                                  "customer_ids": examples[ID_COLUMN].tolist(),
                                  "positive_scores": reference_scores.tolist()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {ARTIFACT_PATH} and {METADATA_PATH}")


if __name__ == "__main__":
    main()
