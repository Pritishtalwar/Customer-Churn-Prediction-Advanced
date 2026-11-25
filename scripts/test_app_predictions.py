import json
from pathlib import Path

import joblib
import pandas as pd
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_model.pkl"
DATA = ROOT / "dataset" / "customer_churn.csv"
METRICS = ROOT / "model_metrics.json"
sys.path.insert(0, str(ROOT / "app"))
from app import validate_batch

model = joblib.load(MODEL)
df = pd.read_csv(DATA)
threshold = json.loads(METRICS.read_text(encoding="utf-8"))["threshold_selection"]["threshold"]
feature_columns = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]

single = df.iloc[[0]][feature_columns].copy()
single_probability = float(model.predict_proba(single)[0, 1])
single_prediction = int(single_probability >= threshold)

batch = df.iloc[:100][feature_columns].copy()
batch_probabilities = model.predict_proba(batch)[:, 1]
batch_predictions = (batch_probabilities >= threshold).astype(int)

# Exercise the app's batch CSV validation directly, without starting a browser.
missing_column_batch = batch.drop(columns=[feature_columns[0]])
validated, error = validate_batch(missing_column_batch, df, feature_columns)
assert validated is None and "Missing required columns" in error
print("batch_missing_column: rejected")

extra_column_batch = batch.assign(UnusedColumn=0)
validated, error = validate_batch(extra_column_batch, df, feature_columns)
assert error is None and "UnusedColumn" not in validated.columns
print("batch_extra_column: ignored; expected model columns selected")

invalid_numeric_batch = batch.copy()
invalid_numeric_batch["Age"] = invalid_numeric_batch["Age"].astype(object)
invalid_numeric_batch.loc[invalid_numeric_batch.index[0], "Age"] = "not-a-number"
validated, error = validate_batch(invalid_numeric_batch, df, feature_columns)
assert validated is None and "finite numeric values" in error
print("batch_invalid_numeric: rejected")

print("single_probability:", round(single_probability, 6))
print("single_prediction:", single_prediction)
print("batch_rows:", len(batch))
print("batch_churn_predictions:", int(batch_predictions.sum()))
print("batch_probability_range:", (
    round(float(batch_probabilities.min()), 6),
    round(float(batch_probabilities.max()), 6),
))
