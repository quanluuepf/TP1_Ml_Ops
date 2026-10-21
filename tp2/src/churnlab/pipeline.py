"""Construction of unfitted preprocessing + classifier pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnlab.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MISSING_CATEGORY = "__missing__"


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median imputation then standard scaling.

    Categorical: missing values become an explicit "__missing__" level, then
    one-hot encoding. Categories never seen in training (including
    "__missing__" when training had no missing values) are encoded as all
    zeros (handle_unknown="ignore") instead of failing.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    """Return a new, unfitted pipeline. Does not read data or train."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
