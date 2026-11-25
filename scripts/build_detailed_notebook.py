from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "churn_analysis.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}


def md(text):
    nb["cells"].append(nbf.v4.new_markdown_cell(text))


def code(text):
    nb["cells"].append(nbf.v4.new_code_cell(text))


md("# Customer Churn Prediction - Complete Analysis")
md("""
## Project Overview

This notebook predicts whether a telecom customer is likely to churn.
The analysis follows a leakage-safe machine-learning workflow:

1. Verify and understand the original data.
2. Audit quality and target relationships.
3. Perform exploratory data analysis.
4. Preserve a fully untouched test set.
5. Compare eleven machine-learning models.
6. Use out-of-fold predictions to select a probability threshold.
7. Refit the selected pipeline on the full training split.
8. Evaluate the selected pipeline once on the untouched test set.

The notebook deliberately distinguishes cross-validation, validation, out-of-fold threshold selection and final test metrics.
""")

md("## 1. Dataset and Business Objective")
md("""
The verified Kaggle Customer Churn Dataset contains customer demographics and account information.
The machine-learning objective is binary classification:

- `0` means the customer is likely to stay.
- `1` means the customer is likely to churn.

`CustomerID` is an identifier, not a behavioral predictor, and is excluded from every model.
""")

md("## 2. Import Libraries")
code("""from pathlib import Path
import json
import subprocess
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)

sns.set_theme(style="whitegrid")
""")

md("## 3. Define Portable Project Paths")
code("""ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

DATASET_PATH = ROOT / "dataset" / "customer_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
METRICS_PATH = ROOT / "model_metrics.json"
COMPARISON_PATH = ROOT / "model_comparison.csv"

print("Project root:", ROOT)
print("Dataset:", DATASET_PATH)
""")

md("## 4. Load the Original Dataset")
code("""df = pd.read_csv(DATASET_PATH)
print("Dataset shape:", df.shape)
display(df.head())
""")

md("## 5. Inspect the Last Records")
code("""display(df.tail())
""")

md("## 6. Verify Dataset Dimensions")
code("""print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
""")

md("## 7. Dataset Schema")
code("""display(pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Non-Null Count": df.notna().sum().values,
}).set_index("Column"))
""")

md("## 8. DataFrame Information")
code("""display(df.info())
""")

md("## 9. Statistical Summary")
code("""display(df.describe(include="all").T)
""")

md("## 10. Identify Numerical and Categorical Features")
code("""numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

numerical_columns = [c for c in numerical_columns if c not in ("Churn", "CustomerID")]
categorical_columns = [c for c in categorical_columns if c != "CustomerID"]

print("Numerical features:", numerical_columns)
print("Categorical features:", categorical_columns)
""")

md("## 11. Target and Identifier")
code("""print("Target:", "Churn")
print("Identifier:", "CustomerID")
print("Target values:", df["Churn"].value_counts(dropna=False).to_dict())
""")

md("## 12. Missing Values")
code("""missing = df.isna().sum().sort_values(ascending=False)
missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
display(pd.DataFrame({
    "Missing Count": missing,
    "Missing Percentage": missing_percent,
}))
""")

md("## 13. Missing-Value Visualization")
code("""fig, ax = plt.subplots(figsize=(10, 5))
missing_with_values = missing_percent[missing_percent > 0]
if missing_with_values.empty:
    ax.text(
        0.5,
        0.5,
        "No missing values",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
    )
else:
    missing_with_values.plot(kind="bar", ax=ax, color="#64748b")
ax.set_title("Missing Values by Column")
ax.set_ylabel("Percentage")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
""")

md("If no columns appear in the chart, the dataset has no missing values.")

md("## 14. Duplicate Rows")
code("""print("Exact duplicate rows:", df.duplicated().sum())
""")

md("## 15. Duplicate Customer IDs")
code("""print("Duplicate CustomerID values:", df["CustomerID"].duplicated().sum())
print("Unique CustomerID values:", df["CustomerID"].nunique())
""")

md("## 16. Constant and Low-Cardinality Columns")
code("""cardinality = df.nunique(dropna=True).sort_values()
display(cardinality.to_frame("Unique Values"))
print("Constant columns:", df.columns[df.nunique(dropna=False) <= 1].tolist())
""")

md("## 17. Target Distribution")
code("""target_counts = df["Churn"].value_counts().sort_index()
display(target_counts.to_frame("Count"))
display((target_counts / len(df)).to_frame("Proportion"))

fig, ax = plt.subplots(figsize=(7, 5))
target_counts.plot(kind="bar", ax=ax, color=["#94a3b8", "#ef4444"])
ax.set_xticklabels(["Stay", "Churn"], rotation=0)
ax.set_title("Churn Distribution")
ax.set_ylabel("Customers")
plt.tight_layout()
plt.show()
""")

md("## 18. Target Balance Interpretation")
md("A balanced or moderately imbalanced target determines whether accuracy alone is sufficient. This project reports precision, recall, F1 and ROC-AUC in addition to accuracy.")

md("## 19. Age Distribution")
code("""fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Age", hue="Churn", bins=40, kde=True, ax=ax)
ax.set_title("Age Distribution by Churn")
plt.tight_layout()
plt.show()
""")

md("## 20. Age Boxplot")
code("""fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="Age", ax=ax)
ax.set_xticklabels(["Stay", "Churn"])
ax.set_title("Age by Churn")
plt.tight_layout()
plt.show()
""")

md("## 21. Tenure Distribution")
code("""fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Tenure", hue="Churn", bins=40, kde=True, ax=ax)
ax.set_title("Tenure Distribution by Churn")
plt.tight_layout()
plt.show()
""")

md("## 22. Tenure vs Churn")
code("""fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="Tenure", ax=ax)
ax.set_xticklabels(["Stay", "Churn"])
ax.set_title("Tenure by Churn")
plt.tight_layout()
plt.show()
""")

md("## 23. Usage Frequency Distribution")
code("""fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Usage Frequency", hue="Churn", bins=40, kde=True, ax=ax)
ax.set_title("Usage Frequency by Churn")
plt.tight_layout()
plt.show()
""")

md("## 24. Support Calls Distribution")
code("""fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Support Calls", hue="Churn", bins=30, kde=True, ax=ax)
ax.set_title("Support Calls by Churn")
plt.tight_layout()
plt.show()
""")

md("## 25. Payment Delay Distribution")
code("""fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Payment Delay", hue="Churn", bins=40, kde=True, ax=ax)
ax.set_title("Payment Delay by Churn")
plt.tight_layout()
plt.show()
""")

md("## 26. Total Spend Distribution")
code("""fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Total Spend", hue="Churn", bins=40, kde=True, ax=ax)
ax.set_title("Total Spend by Churn")
plt.tight_layout()
plt.show()
""")

md("## 27. Last Interaction Distribution")
code("""fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Last Interaction", hue="Churn", bins=40, kde=True, ax=ax)
ax.set_title("Last Interaction by Churn")
plt.tight_layout()
plt.show()
""")

md("## 28. Gender vs Churn")
code("""gender_rate = pd.crosstab(df["Gender"], df["Churn"], normalize="index")
display(gender_rate)

fig, ax = plt.subplots(figsize=(8, 5))
gender_rate.plot(kind="bar", ax=ax)
ax.set_title("Gender vs Churn")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
""")

md("## 29. Subscription Type vs Churn")
code("""subscription_rate = pd.crosstab(df["Subscription Type"], df["Churn"], normalize="index")
display(subscription_rate)

fig, ax = plt.subplots(figsize=(9, 5))
subscription_rate.plot(kind="bar", ax=ax)
ax.set_title("Subscription Type vs Churn")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
""")

md("## 30. Contract Length vs Churn")
code("""contract_rate = pd.crosstab(df["Contract Length"], df["Churn"], normalize="index")
display(contract_rate)

fig, ax = plt.subplots(figsize=(9, 5))
contract_rate.plot(kind="bar", ax=ax)
ax.set_title("Contract Length vs Churn")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
""")

md("## 31. Numerical Correlation Heatmap")
code("""correlation = df[numerical_columns + ["Churn"]].corr(numeric_only=True)

plt.figure(figsize=(11, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Numerical Correlation Heatmap")
plt.tight_layout()
plt.show()
""")

md("## 32. Correlation Ranking")
code("""ranking = correlation["Churn"].drop("Churn").abs().sort_values(ascending=False)
display(ranking.to_frame("Absolute Correlation"))

ranking.plot(kind="barh", figsize=(9, 6))
plt.title("Numerical Features Ranked by Absolute Churn Correlation")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
""")

md("## 33. Target-Pattern Audit")
code("""pattern_result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "analyze_target_pattern.py")],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
)
print(pattern_result.stdout)
""")

md("## 34. Interpretation of Target Patterns")
md("""
Strong relationships between customer behavior fields and `Churn` explain why multiple models perform very well.
The audit also checks:

- Duplicate feature rows.
- Conflicting labels for the same feature combination.
- Whether `CustomerID` is included.
- Whether simple thresholds achieve unusually high accuracy.

Strong feature relationships are not automatically evidence of leakage.
There is no evidence of conflicting duplicate labels or identifier leakage in this pipeline.
""")

md("## 35. Feature Engineering Decision")
md("""
`CustomerID` is removed because it is an identifier rather than a customer behavior variable.
No target-derived features are created.
No synthetic rows or oversampling are used.
""")

md("## 36. Separate Features and Target")
code("""X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"].astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
""")

md("## 37. Reidentify Model Feature Types")
code("""numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)
""")

md("## 38. Leakage-Safe Preprocessing")
md("""
The pipeline applies transformations inside cross-validation folds.
Numeric features use median imputation and standard scaling.
Categorical features use most-frequent imputation and one-hot encoding.
""")

md("## 39. Build the ColumnTransformer")
code("""from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])
""")

md("## 40. Stratified 80:20 Split")
code("""from sklearn.model_selection import train_test_split

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Full training rows:", X_train_full.shape[0])
print("Untouched test rows:", X_test.shape[0])
""")

md("## 41. Keep the Test Set Untouched")
md("""
The test set is not used to select models, tune hyperparameters or choose thresholds.
It is used exactly once for the final evaluation.
""")

md("## 42. Dummy Baseline")
code("""from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

baseline = DummyClassifier(strategy="most_frequent", random_state=42)
baseline.fit(X_train_full, y_train_full)
baseline_prediction = baseline.predict(X_test)

print("Dummy baseline accuracy:", accuracy_score(y_test, baseline_prediction))
""")

md("## 43. Execute the Eleven-Model Training Script")
code("""train_result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "train_models.py")],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
)
print(train_result.stdout[-14000:])
""")

md("## 44. Model Comparison Table")
code("""comparison = pd.read_csv(COMPARISON_PATH)
display(comparison)
""")

md("## 45. Cross-Validation Accuracy Comparison")
code("""successful = comparison[comparison["Status"] == "Success"].copy()
if not successful.empty:
    successful.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot(
        kind="bar", figsize=(14, 7)
    )
    plt.title("Cross-Validation Model Comparison")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.show()
""")

md("## 46. Validation Accuracy Comparison")
code("""if "Validation Accuracy" in successful.columns:
    successful.sort_values("Validation Accuracy", ascending=False).set_index("Model")[
        ["Validation Accuracy", "Validation Precision", "Validation Recall",
         "Validation F1", "Validation ROC-AUC"]
    ].plot(kind="bar", figsize=(14, 7))
    plt.title("Validation Metrics")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.show()
""")

md("## 47. Training-Time Comparison")
code("""if "Training Seconds" in successful.columns:
    successful.sort_values("Training Seconds").set_index("Model")["Training Seconds"].plot(
        kind="barh", figsize=(9, 7), color="#2563eb"
    )
    plt.title("Model Training Time")
    plt.tight_layout()
    plt.show()
""")

md("## 48. Load Final Metrics")
code("""metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
print("Selected model:", metrics["selected_model"])
print("OOF threshold:", metrics["threshold_selection"]["threshold"])
print("Baseline:", metrics["baseline"])
print("Final test:", metrics["final_test"])
""")

md("## 49. Cross-Validation vs Validation vs Test")
md("""
These metrics measure different stages:

- **Cross-validation:** folds inside the 30,000-row subset.
- **Validation:** 20% holdout inside the 80% training split.
- **Out-of-fold threshold:** 5-fold probabilities on the full 80% training split.
- **Test:** 12,875 rows evaluated once after model and threshold selection.
""")

md("## 50. Hyperparameter Tuning")
code("""display(metrics["hyperparameter_tuning"])
""")

md("## 51. OOF Threshold Method")
code("""display(metrics["threshold_selection"])
""")

md("## 52. Load the Saved Pipeline")
code("""model = joblib.load(MODEL_PATH)
print("Model type:", type(model).__name__)
print("Fitted pipeline steps:", list(model.named_steps))
""")

md("## 53. Verify CustomerID Is Not a Model Input")
code("""preprocessor = model.named_steps["preprocessing"]
feature_names = model.feature_names_in_.tolist()
print("CustomerID in pipeline inputs:", "CustomerID" in feature_names)
print("Model input count:", len(feature_names))
print("First model inputs:", feature_names[:15])
""")

md("## 54. Feature Importance")
code("""estimator = model.named_steps["model"]
if hasattr(estimator, "feature_importances_"):
    names = preprocessor.get_feature_names_out()
    importance = pd.Series(estimator.feature_importances_, index=names)
    importance = importance.sort_values(ascending=False).head(20)
    importance.iloc[::-1].plot(kind="barh", figsize=(10, 8), color="#2563eb")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()
else:
    print("The selected model does not expose native feature importances.")
""")

md("## 55. Final Test Predictions")
code("""probability = model.predict_proba(X_test)[:, 1]
threshold = metrics["threshold_selection"]["threshold"]
prediction = (probability >= threshold).astype(int)

print("Prediction threshold:", threshold)
print("Predicted churn count:", int(prediction.sum()))
print("Predicted stay count:", int((1 - prediction).sum()))
""")

md("## 56. Confusion Matrix")
code("""fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    prediction,
    display_labels=["Stay", "Churn"],
    cmap="Blues",
    ax=ax,
)
ax.set_title("Final Untouched Test Confusion Matrix")
plt.tight_layout()
plt.show()
""")

md("## 57. Classification Report")
code("""print(classification_report(y_test, prediction, target_names=["Stay", "Churn"], digits=4))
""")

md("## 58. ROC Curve")
code("""fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, probability, ax=ax)
ax.set_title("Final Untouched Test ROC Curve")
plt.tight_layout()
plt.show()
""")

md("## 59. Precision-Recall Curve")
code("""fig, ax = plt.subplots(figsize=(7, 6))
PrecisionRecallDisplay.from_predictions(y_test, probability, ax=ax)
ax.set_title("Final Untouched Test Precision-Recall Curve")
plt.tight_layout()
plt.show()
""")

md("## 60. Actual vs Predicted Distribution")
code("""pd.DataFrame({
    "Actual": y_test.map({0: "Stay", 1: "Churn"}),
    "Predicted": pd.Series(prediction).map({0: "Stay", 1: "Churn"}).values,
}).value_counts().rename("Count").to_frame().plot(
    kind="bar", figsize=(9, 5)
)
plt.title("Actual vs Predicted Outcome Counts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
""")

md("## 61. False Positives and False Negatives")
md("""
- False positive: predicted churn, but the customer stayed.
- False negative: predicted stay, but the customer churned.

For retention, false negatives are usually important because missed churners do not receive intervention.
""")

md("## 62. Limitations")
md("""
Near-perfect performance must be interpreted cautiously.
This audit found strong rule-like relationships in the target structure, but no direct proof of target leakage.
Strong synthetic patterns can produce very high accuracy even though the feature values are ordinary account measurements.
""")

md("## 63. Final Model Metrics")
code("""final = metrics["final_test"]
print("Selected model:", metrics["selected_model"])
print("Accuracy:", f"{final['accuracy']:.6%}")
print("Precision:", f"{final['precision']:.6%}")
print("Recall:", f"{final['recall']:.6%}")
print("F1:", f"{final['f1']:.6f}")
print("ROC-AUC:", f"{final['roc_auc']:.6f}")
print("90% target achieved:", metrics["target_90_percent_achieved"])
""")

md("## 64. Smoke Test the Reloaded Model")
code("""smoke_predictions = model.predict(X_test.iloc[:20])
print("Reloaded model test predictions:", smoke_predictions.tolist())
""")

md("## 65. Conclusion")
md("""
The final selected pipeline is LightGBM with an out-of-fold-selected threshold.
The untouched test set reported near-perfect accuracy.

The model result should be reported honestly with:

- Exact test metrics.
- Confusion matrix.
- Target-pattern findings.
- Dummy-baseline comparison.
- The possibility of strong synthetic or rule-like structure.

This is the complete end-to-end analysis; deployment decisions should not rely on accuracy alone.
""")

nb["cells"] = nb["cells"]
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
