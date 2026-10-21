"""Loading, validation and stateless parsing of the raw Telco snapshot."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/telco.csv")
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

ID_COLUMN = "customerID"
TARGET = "Churn"
POSITIVE_LABEL = "Yes"
TARGET_MAPPING = {"No": 0, "Yes": 1}

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path = RAW_PATH, expected_sha256: str = EXPECTED_SHA256) -> pd.DataFrame:
    """Read the immutable CSV as text after checking its bytes."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Source hash mismatch for {path}: {actual}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted({ID_COLUMN, TARGET, *FEATURES} - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate customer IDs in source")
    return df


def _blank_to_nan(value):
    if isinstance(value, str):
        value = value.strip()
        return value if value else np.nan
    return np.nan if pd.isna(value) else value


def parse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the six model inputs with documented types.

    Rules (identical at training and inference, nothing is learned here):
    - a missing required column raises a KeyError;
    - blank or whitespace-only values become NaN (allowed missing values);
    - numeric columns are converted with errors="raise", so malformed
      nonblank text such as "abc" fails instead of becoming NaN;
    - categorical columns are stripped strings (object dtype).
    """
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        values = df[col].astype(object).map(_blank_to_nan)
        try:
            out[col] = pd.to_numeric(values, errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric value in {col!r}: {exc}") from exc
    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype(object).map(_blank_to_nan).astype(object)
    return out


def parse_target(df: pd.DataFrame) -> pd.Series:
    """Map Churn to 1 (Yes, positive class) / 0 (No); fail on anything else."""
    labels = df[TARGET].astype(str).str.strip()
    unknown = sorted(set(labels) - set(TARGET_MAPPING))
    if unknown:
        raise ValueError(f"Unexpected target labels: {unknown}")
    return labels.map(TARGET_MAPPING).astype(int)
