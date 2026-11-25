"""Evaluate the saved model on the project's fixed held-out test split."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "customer_churn.csv"
MODEL = ROOT / "models" / "churn_model.pkl"
METRICS = ROOT / "model_metrics.json"
EVALUATION = ROOT / "evaluation_results.json"
RANDOM_STATE = 42

metrics = json.loads(METRICS.read_text(encoding="utf-8"))
threshold_info = metrics["threshold_selection"]
if threshold_info.get("method") != "5-fold out-of-fold predictions on 80% training split":
    raise ValueError("Expected a training-only OOF threshold selection record")
if not threshold_info.get("frozen_before_test_evaluation"):
    raise ValueError("Threshold must be frozen before test evaluation")
threshold = float(threshold_info["threshold"])

df = pd.read_csv(DATA).dropna(subset=["Churn"]).copy()
df["Churn"] = df["Churn"].astype(int)
X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

model = joblib.load(MODEL)
if "CustomerID" in model.feature_names_in_:
    raise ValueError("CustomerID unexpectedly appears in saved model inputs")
model_probability = model.predict_proba(X_test)[:, 1]
model_prediction = (model_probability >= threshold).astype(int)

baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
baseline_prediction = baseline.predict(X_test)
baseline_probability = baseline.predict_proba(X_test)[:, 1]


def summarize(y_true, prediction, probability):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
        "classification_report": classification_report(
            y_true,
            prediction,
            output_dict=True,
            zero_division=0,
        ),
    }


result = {
    "evaluation": {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_fraction": 0.20,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "customer_id_excluded": "CustomerID" not in model.feature_names_in_,
        "threshold": threshold,
        "threshold_selection_method": threshold_info["method"],
        "threshold_frozen_before_test_evaluation": True,
        "test_used_for_model_or_threshold_selection": False,
    },
    "model": summarize(y_test, model_prediction, model_probability),
    "dummy_baseline": summarize(y_test, baseline_prediction, baseline_probability),
}

# Keep existing experiment, feature audit, and leakage audit fields intact.
metrics["final_test"] = {
    key: result["model"][key]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
}
metrics["baseline"] = {
    key: result["dummy_baseline"][key]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
}
metrics["confusion_matrix"] = result["model"]["confusion_matrix"]
metrics["classification_report"] = result["model"]["classification_report"]
metrics["baseline_confusion_matrix"] = result["dummy_baseline"]["confusion_matrix"]
metrics["baseline_classification_report"] = result["dummy_baseline"]["classification_report"]
metrics["evaluation"] = result["evaluation"]
METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
EVALUATION.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(json.dumps(result, indent=2))
print(f"Saved updated metrics: {METRICS}")
print(f"Saved full evaluation: {EVALUATION}")
