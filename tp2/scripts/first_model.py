"""Milestone 1: two numeric features, logistic regression vs majority baseline."""

import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from churnlab.data import load_raw, parse_features, parse_target
from churnlab.split import check_disjoint, split_customers

CONFIG = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
FIRST_FEATURES = ["tenure", "MonthlyCharges"]

raw = load_raw()
train_df, valid_df, _ = split_customers(
    raw, CONFIG["seed"], CONFIG["valid_size"], CONFIG["test_size"]
)
check_disjoint(train_df, valid_df, _)

X_train = parse_features(train_df)[FIRST_FEATURES]
X_valid = parse_features(valid_df)[FIRST_FEATURES]
y_train, y_valid = parse_target(train_df), parse_target(valid_df)

report = {"positive_class": "Churn=Yes (1)", "features": FIRST_FEATURES,
          "confusion_matrix_layout": "rows=true [0,1], cols=predicted [0,1]"}
for name, model in [
    ("majority_baseline", DummyClassifier(strategy="most_frequent")),
    ("logistic_regression", LogisticRegression(max_iter=1000)),
]:
    model.fit(X_train, y_train)  # both fitted on training data only
    predicted = model.predict(X_valid)
    report[name] = {
        "validation_accuracy": float(accuracy_score(y_valid, predicted)),
        "confusion_matrix": confusion_matrix(y_valid, predicted, labels=[0, 1]).tolist(),
    }

out = Path("reports/tp02_first_model.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
