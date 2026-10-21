"""Validation metrics for binary churn classifiers (positive class = 1 = Churn Yes)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def positive_scores(model, X) -> np.ndarray:
    """Probability of class 1, located through classes_ rather than assumed."""
    proba = model.predict_proba(X)
    return proba[:, list(model.classes_).index(1)]


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """Average precision on continuous scores; other metrics at a fixed threshold."""
    scores = positive_scores(model, X)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
