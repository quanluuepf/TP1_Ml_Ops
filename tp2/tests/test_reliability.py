"""Focused reliability checks for TP02 (section 5)."""

import copy
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churnlab.data import FEATURES, ID_COLUMN, NUMERIC_FEATURES, load_raw, parse_features, parse_target
from churnlab.evaluate import positive_scores
from churnlab.pipeline import build_pipeline
from churnlab.split import split_customers

SEED = 42


@pytest.fixture(scope="module")
def splits():
    return split_customers(load_raw(), SEED)


@pytest.fixture(scope="module")
def fitted(splits):
    train_df = splits[0]
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(parse_features(train_df), parse_target(train_df))
    return pipe


def numeric_imputer(pipe):
    return pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]


def test_customer_ids_do_not_overlap(splits):
    train_ids, valid_ids, test_ids = (set(part[ID_COLUMN]) for part in splits)
    assert not train_ids & valid_ids
    assert not train_ids & test_ids
    assert not valid_ids & test_ids
    assert len(train_ids) + len(valid_ids) + len(test_ids) == 7043


def test_split_is_reproducible(splits):
    again = split_customers(load_raw(), SEED)
    for first, second in zip(splits, again):
        assert list(first[ID_COLUMN]) == list(second[ID_COLUMN])


def test_imputer_medians_come_from_training_only(splits, fitted):
    train_medians = parse_features(splits[0])[NUMERIC_FEATURES].median().to_numpy()
    full_medians = parse_features(load_raw())[NUMERIC_FEATURES].median().to_numpy()
    learned = numeric_imputer(fitted).statistics_
    np.testing.assert_allclose(learned, train_medians)
    # The full-dataset TotalCharges median differs, so this check is discriminative.
    assert not np.allclose(learned, full_medians)


def test_prediction_does_not_refit(splits, fitted):
    before = copy.deepcopy(numeric_imputer(fitted).statistics_)
    scaler_mean = copy.deepcopy(
        fitted.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"].mean_)
    fitted.predict_proba(parse_features(splits[1]))
    np.testing.assert_array_equal(numeric_imputer(fitted).statistics_, before)
    np.testing.assert_array_equal(
        fitted.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"].mean_,
        scaler_mean)


def test_missing_numeric_and_unknown_category_follow_policy(fitted):
    batch = pd.DataFrame([{
        "tenure": "5", "MonthlyCharges": "70.0", "TotalCharges": " ",
        "Contract": "Three year", "InternetService": "Fiber optic", "PaymentMethod": "Crypto wallet",
    }])
    X = parse_features(batch)
    assert np.isnan(X.loc[0, "TotalCharges"])
    transformed = fitted.named_steps["preprocessor"].transform(X)
    names = list(fitted.named_steps["preprocessor"].get_feature_names_out())
    contract_cols = [i for i, n in enumerate(names) if n.startswith("cat__Contract_")]
    assert transformed[0, contract_cols].sum() == 0  # unseen category -> all zeros
    score = positive_scores(fitted, X)
    assert 0.0 <= score[0] <= 1.0


def test_malformed_numeric_text_fails():
    batch = pd.DataFrame([dict.fromkeys(FEATURES, "x") | {
        "tenure": "abc", "MonthlyCharges": "1", "TotalCharges": "1"}])
    with pytest.raises(ValueError, match="tenure"):
        parse_features(batch)


def test_missing_required_feature_fails_visibly(splits, fitted):
    X = parse_features(splits[1]).drop(columns=["Contract"])
    with pytest.raises(KeyError, match="Contract"):
        parse_features(splits[1].drop(columns=["Contract"]))
    with pytest.raises(ValueError):
        fitted.predict(X)  # the pipeline itself also refuses


def test_reload_in_fresh_process_preserves_predictions(tmp_path, splits, fitted):
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(fitted, artifact)
    X = parse_features(splits[1].head(20))
    batch_path = tmp_path / "batch.pkl"
    X.to_pickle(batch_path)
    expected = positive_scores(fitted, X)

    code = (
        "import sys, joblib, pandas as pd\n"
        "from churnlab.evaluate import positive_scores\n"
        "pipe = joblib.load(sys.argv[1])\n"
        "X = pd.read_pickle(sys.argv[2])\n"
        "print(','.join(repr(float(s)) for s in positive_scores(pipe, X)))\n"
    )
    out = subprocess.check_output([sys.executable, "-c", code, str(artifact), str(batch_path)], text=True)
    loaded = np.array([float(v) for v in out.strip().split(",")])
    np.testing.assert_allclose(loaded, expected, rtol=0, atol=1e-9)
