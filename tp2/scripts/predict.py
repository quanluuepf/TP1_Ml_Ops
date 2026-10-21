"""Inference only: load the trusted fitted pipeline and score new customers.

Usage (from tp2/):  uv run python scripts/predict.py [--input data/examples/new_customers.csv]
No training labels are read and no fit is called.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnlab.data import ID_COLUMN, parse_features, sha256_file
from churnlab.evaluate import positive_scores

ARTIFACT_PATH = Path("artifacts/churn_pipeline.joblib")
METADATA_PATH = Path("artifacts/churn_pipeline.metadata.json")
TOLERANCE = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/examples/new_customers.csv")
    args = parser.parse_args()

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if sha256_file(ARTIFACT_PATH) != metadata["artifact"]["sha256"]:
        raise ValueError("Artifact hash differs from its metadata record; refusing to load")
    pipeline = joblib.load(ARTIFACT_PATH)  # trusted artifact only

    batch = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    X = parse_features(batch)  # raises on missing columns or malformed numbers
    scores = positive_scores(pipeline, X)
    threshold = metadata["threshold"]
    output = pd.DataFrame({
        ID_COLUMN: batch[ID_COLUMN],
        "churn_score": scores.round(6),
        "predicted_churn": np.where(scores >= threshold, "Yes", "No"),
    })
    print(output.to_string(index=False))

    reference = metadata["reference_predictions"]
    if Path(args.input).as_posix() == reference["examples_file"]:
        max_diff = float(np.max(np.abs(scores - np.array(reference["positive_scores"]))))
        agree = max_diff <= TOLERANCE
        print(f"Max |loaded - original| score difference: {max_diff:.3e} "
              f"(tolerance {TOLERANCE:g}) -> {'AGREE' if agree else 'DISAGREE'}")
        if not agree:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
