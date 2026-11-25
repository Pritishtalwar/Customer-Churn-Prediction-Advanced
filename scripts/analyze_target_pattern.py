import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "customer_churn.csv"
OUTPUT = ROOT / "target_pattern_analysis.json"

df = pd.read_csv(DATA)
X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"].astype(int)

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric = X.select_dtypes(include=[np.number]).columns.tolist()
categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical),
])

depth_results = {}
for depth in (1, 2, 3, 4, 5):
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("model", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    depth_results[str(depth)] = {
        "training_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "validation_accuracy": float(
            accuracy_score(y_holdout, model.predict(X_holdout))
        ),
    }

support_ge5 = (df["Support Calls"] >= 5).astype(int)
payment_delay_thresholds = {}
for threshold in (10, 15, 20, 25):
    payment_delay_thresholds[str(threshold)] = float(
        accuracy_score(y, (df["Payment Delay"] >= threshold).astype(int))
    )

numeric_correlations = df.select_dtypes(include=[np.number]).corr()["Churn"].drop(
    "Churn",
    errors="ignore",
).sort_values().to_dict()

result = {
    "holdout_split": "20% validation, random_state=42",
    "no_customer_id_used": True,
    "exact_duplicate_feature_rows": int(
        X.duplicated().sum()
    ),
    "conflicting_duplicate_feature_labels": int(
        df.drop(columns=["CustomerID"])
        .groupby(X.columns.tolist(), dropna=False)["Churn"]
        .nunique()
        .gt(1)
        .sum()
    ),
    "numeric_target_correlations": numeric_correlations,
    "simple_support_calls_ge5_accuracy": float(
        accuracy_score(y, support_ge5)
    ),
    "payment_delay_rule_accuracies": payment_delay_thresholds,
    "shallow_decision_tree_results": depth_results,
    "interpretation": (
        "The dataset contains strong rule-like relationships, especially in "
        "Support Calls and Payment Delay. This is evidence of strong "
        "target structure, not by itself evidence of target leakage. "
        "CustomerID is excluded and no conflicting duplicate feature/label "
        "groups were found."
    ),
}

OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
