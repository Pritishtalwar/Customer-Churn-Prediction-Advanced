import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_model.pkl"
DATA = ROOT / "dataset" / "customer_churn.csv"
METRICS = ROOT / "model_metrics.json"

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

print("single_probability:", round(single_probability, 6))
print("single_prediction:", single_prediction)
print("batch_rows:", len(batch))
print("batch_churn_predictions:", int(batch_predictions.sum()))
print("batch_probability_range:", (
    round(float(batch_probabilities.min()), 6),
    round(float(batch_probabilities.max()), 6),
))
