"""Audit target structure and feature importance without using test data for modeling."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "customer_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
METRICS_PATH = ROOT / "model_metrics.json"
AUDIT_PATH = ROOT / "data_leakage_audit.json"
NATIVE_IMPORTANCE_PATH = ROOT / "feature_importance_report.csv"
PERMUTATION_PATH = ROOT / "permutation_importance_report.csv"
UNIVARIATE_PATH = ROOT / "univariate_target_signal_report.csv"
RANDOM_STATE = 42

df = pd.read_csv(DATA).dropna(subset=["Churn"]).copy()
df["Churn"] = df["Churn"].astype(int)
X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"]

# Reproduce the project split first, then reserve validation rows within its
# training partition. No test rows or labels enter model/importance analysis.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_fit, X_validation, y_fit, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
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

saved_model = joblib.load(MODEL_PATH)
feature_inputs = list(saved_model.feature_names_in_)
assert "CustomerID" not in feature_inputs
validation_model = clone(saved_model)
validation_model.fit(X_fit, y_fit)
validation_prediction = validation_model.predict(X_validation)
validation_accuracy = float(accuracy_score(y_validation, validation_prediction))

# Native LightGBM feature_importances_ is split count importance. Aggregate
# one-hot columns to source feature names for a readable report.
native_estimator = saved_model.named_steps["model"]
encoded_names = saved_model.named_steps["preprocessing"].get_feature_names_out()
native_encoded = pd.DataFrame({
    "encoded_feature": encoded_names,
    "split_count_importance": native_estimator.feature_importances_,
})
native_encoded.to_csv(NATIVE_IMPORTANCE_PATH, index=False)
native_by_feature = {}
for raw_feature in X.columns:
    native_by_feature[raw_feature] = int(native_encoded.loc[
        native_encoded["encoded_feature"].str.startswith(
            ("numeric__" + raw_feature, "categorical__" + raw_feature + "_")
        ), "split_count_importance"
    ].sum())
pd.DataFrame([
    {"feature": name, "split_count_importance": value}
    for name, value in sorted(native_by_feature.items(), key=lambda item: item[1], reverse=True)
]).to_csv(NATIVE_IMPORTANCE_PATH.with_name("feature_importance_by_source_feature.csv"), index=False)

# Permutation importance uses an independent validation subset of the 80%
# training partition. Test data is not passed to permutation_importance.
permutation = permutation_importance(
    validation_model,
    X_validation,
    y_validation,
    scoring="accuracy",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
permutation_table = pd.DataFrame({
    "feature": X.columns,
    "mean_validation_accuracy_drop": permutation.importances_mean,
    "std_validation_accuracy_drop": permutation.importances_std,
}).sort_values("mean_validation_accuracy_drop", ascending=False)
permutation_table.to_csv(PERMUTATION_PATH, index=False)

# One-feature shallow trees test whether an individual field nearly determines
# the label. These are trained only on X_fit/y_fit and measured on validation.
univariate_rows = []
for feature in X.columns:
    feature_numeric = pd.api.types.is_numeric_dtype(X[feature])
    one_feature_preprocessor = ColumnTransformer([
        ("value", SimpleImputer(strategy="median" if feature_numeric else "most_frequent"), [feature])
        if feature_numeric else
        ("value", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), [feature])
    ])
    stump = Pipeline([
        ("preprocessing", one_feature_preprocessor),
        ("model", DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)),
    ])
    stump.fit(X_fit[[feature]], y_fit)
    train_accuracy = accuracy_score(y_fit, stump.predict(X_fit[[feature]]))
    val_accuracy = accuracy_score(y_validation, stump.predict(X_validation[[feature]]))
    univariate_rows.append({
        "feature": feature,
        "fit_accuracy": float(train_accuracy),
        "validation_accuracy": float(val_accuracy),
    })
univariate_table = pd.DataFrame(univariate_rows).sort_values("validation_accuracy", ascending=False)
univariate_table.to_csv(UNIVARIATE_PATH, index=False)

# ID-ordering summaries use only the 80% training partition and are never
# passed to the predictive model.
train_rows = df.loc[X_train_full.index]
id_decile = pd.qcut(train_rows["CustomerID"], 10, labels=False, duplicates="drop")
id_churn_rates = train_rows.assign(customer_id_decile=id_decile).groupby(
    "customer_id_decile", observed=True
)["Churn"].agg(["mean", "size"])
train_numeric = train_rows.select_dtypes(include=[np.number])
id_feature_correlations = train_numeric.corr(numeric_only=True)["CustomerID"].drop(
    ["CustomerID", "Churn"], errors="ignore"
).sort_values(key=lambda values: values.abs(), ascending=False)

# Duplicate and overlap checks include full-source integrity only; importance,
# target signal models, and their labels use fit/validation rows only.
feature_groups = df.groupby(list(X.columns), dropna=False)["Churn"].agg(["size", "nunique"])
numeric_corr = X_train_full.select_dtypes(include=[np.number]).corr().abs()
high_corr = [
    {"feature_a": a, "feature_b": b, "absolute_correlation": float(numeric_corr.loc[a, b])}
    for i, a in enumerate(numeric_corr.columns)
    for b in numeric_corr.columns[i + 1:]
    if numeric_corr.loc[a, b] >= 0.90
]

report = {
    "dataset": {
        "rows": int(len(df)),
        "source_recorded_in_project": "Kaggle: muhammadshahidazeem/customer-churn-dataset",
        "target_generation_code_or_documentation_present": False,
        "synthetic_or_rule_generated_status": "Cannot be proven from the CSV or project code. Extremely strong ID ordering and feature-target structure are consistent with possible source ordering or rule-generated/synthetic labels, but are not proof.",
    },
    "split_and_resampling": {
        "outer_split": "Stratified 80:20 before model fitting/resampling",
        "training_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
        "fit_rows_for_validation_importance": int(len(X_fit)),
        "validation_rows_for_permutation_importance": int(len(X_validation)),
        "random_state": RANDOM_STATE,
        "resampling_or_synthetic_oversampling_in_training_code": False,
        "resampling_dependency_in_requirements": False,
        "test_data_used_for_importance": False,
    },
    "target_and_feature_relationships": {
        "direct_churn_target_feature_present": "No separate churn date/status source field exists in the 10 model inputs; the CSV does not document how Churn was assigned.",
        "customer_id_in_model_inputs": "CustomerID" in feature_inputs,
        "customer_id_churn_correlation_descriptive_only_training_rows": float(train_rows["CustomerID"].corr(train_rows["Churn"])),
        "numeric_predictor_customer_id_correlations_training_rows": {
            feature: float(correlation)
            for feature, correlation in id_feature_correlations.items()
        },
        "churn_rate_by_customer_id_decile": {
            str(int(idx)): {"churn_rate": float(row["mean"]), "rows": int(row["size"])}
            for idx, row in id_churn_rates.iterrows()
        },
        "single_feature_stump_fit_and_validation_accuracy": univariate_rows,
        "interpretation": "No input is a documented direct copy of Churn, and CustomerID is excluded. The very strong label ordering by CustomerID and high univariate signal in account/behavior fields warrant concern about source construction, temporal ordering, or target rules. Without source-generation logic, leakage cannot be confirmed or ruled out.",
    },
    "record_integrity": {
        "duplicate_full_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["CustomerID"].duplicated().sum()),
        "duplicate_feature_vectors_excluding_id_and_target": int(X.duplicated().sum()),
        "duplicate_feature_groups_with_conflicting_labels": int((feature_groups["nunique"] > 1).sum()),
        "train_validation_customer_id_overlap": int(len(set(df.loc[X_fit.index, "CustomerID"]) & set(df.loc[X_validation.index, "CustomerID"]))),
        "train_test_customer_id_overlap": int(len(set(df.loc[X_train_full.index, "CustomerID"]) & set(df.loc[X_test.index, "CustomerID"]))),
        "train_test_feature_vector_overlap": int(len(set(map(tuple, X_train_full.to_numpy())) & set(map(tuple, X_test.to_numpy())))),
    },
    "feature_diagnostics": {
        "native_importance": "LightGBM split counts from the saved final model, fitted on training rows only; detailed encoded and aggregated CSV reports saved.",
        "permutation_importance": "Accuracy drop on the 10,300-row validation subset; diagnostic pipeline fitted only on the 41,199-row fit subset.",
        "diagnostic_validation_accuracy": validation_accuracy,
        "high_numeric_correlation_pairs_abs_ge_0_90_training_only": high_corr,
        "features_removed_or_added": [],
    },
    "limitations": [
        "The Kaggle CSV does not include source code, label-generation rules, a churn date, or a source customer/time key that would establish causal feature timing.",
        "Random stratified splitting can preserve source ordering structure; it does not test forward-in-time generalization.",
        "Permutation importance measures predictive dependence on this validation subset, not causality; correlated predictors can share or mask importance.",
        "A high feature-target association is a warning signal, not proof that a feature was used to construct the target.",
    ],
}
AUDIT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("Saved:", AUDIT_PATH, NATIVE_IMPORTANCE_PATH, PERMUTATION_PATH, UNIVARIATE_PATH)
