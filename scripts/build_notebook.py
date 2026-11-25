from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}


def md(text):
    notebook["cells"].append(nbf.v4.new_markdown_cell(text))


def code(text):
    notebook["cells"].append(nbf.v4.new_code_cell(text))


md("# Customer Churn Prediction")
md("## 1. Project Introduction")
md("This project predicts customer churn using the verified Kaggle Customer Churn Dataset. The workflow prevents test-set leakage: preprocessing, model selection, tuning and threshold selection use training/validation data only.")

md("## 2. Library Imports")
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

md("## 3. Dataset Loading")
code("""ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

DATA = ROOT / "dataset" / "customer_churn.csv"
df = pd.read_csv(DATA)
print("Shape:", df.shape)
display(df.head())
display(df.tail())
""")

md("## 4. Data Understanding")
code("""display(df.info())
display(df.describe(include="all").T)

numerical = df.select_dtypes(include=np.number).columns.drop(["Churn", "CustomerID"], errors="ignore")
categorical = df.select_dtypes(exclude=np.number).columns
print("Numerical features:", numerical.tolist())
print("Categorical features:", categorical.tolist())
""")

md("## 5. Data Quality Analysis")
code("""display((df.isna().mean() * 100).sort_values(ascending=False).to_frame("missing_percentage"))
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate CustomerID:", df["CustomerID"].duplicated().sum())
print("Constant columns:", df.columns[df.nunique(dropna=False) <= 1].tolist())
display(df.nunique(dropna=True).sort_values().to_frame("unique_values"))
display(df["Churn"].value_counts().to_frame("count"))
""")

md("## 6. Exploratory Data Analysis")
code("""eda_result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "eda_analysis.py")],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
)
print(eda_result.stdout)
""")

md("## 7. Feature Engineering")
md("CustomerID is excluded. The supplied customer, usage and billing fields are retained unless feature selection demonstrates a change.")
code("""X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"].astype(int)
print("Feature shape:", X.shape)

pattern_result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "analyze_target_pattern.py")],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
)
print(pattern_result.stdout[-10000:])
""")

md("## 8. Data Preprocessing")
code("""from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_standard = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
numeric_minmax = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
])
categorical_preprocessing = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_standard, numeric),
    ("categorical", categorical_preprocessing, categorical),
])
""")

md("## 9. Train-Test Split")
code("""from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print("Train:", X_train.shape)
print("Test:", X_test.shape)
""")

md("## 10. Model Training")
code("""train_result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "train_models.py")],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
)
print(train_result.stdout[-12000:])
""")

md("## 11. Model Comparison")
code("""comparison = pd.read_csv(ROOT / "model_comparison.csv")
display(comparison)

successful = comparison[comparison["Status"] == "Success"]
if not successful.empty:
    successful.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot(
        kind="bar", figsize=(13, 6)
    )
    plt.title("Cross-Validation Model Comparison")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.show()
""")

md("## 12. Hyperparameter Tuning")
code("""metrics = json.loads((ROOT / "model_metrics.json").read_text())
print("Selected model:", metrics["selected_model"])
print("OOF validation threshold:", metrics["threshold_selection"]["threshold"])
""")

md("## 13. Feature Importance")
code("""model = joblib.load(ROOT / "models" / "churn_model.pkl")
estimator = model.named_steps["model"]
if hasattr(estimator, "feature_importances_"):
    names = model.named_steps["preprocessing"].get_feature_names_out()
    importance = pd.Series(estimator.feature_importances_, index=names).sort_values(ascending=False).head(20)
    importance.iloc[::-1].plot(kind="barh", figsize=(9, 7))
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()
else:
    print("The selected model does not expose native feature importances.")
""")

md("## 14. Final Model Evaluation")
code("""model = joblib.load(ROOT / "models" / "churn_model.pkl")
probability = model.predict_proba(X_test)[:, 1]
prediction = (probability >= metrics["threshold_selection"]["threshold"]).astype(int)

print("Baseline:", metrics["baseline"])
print("Final test:", metrics["final_test"])
print(classification_report(y_test, prediction, digits=4))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, prediction, ax=axes[0])
axes[0].set_title("Confusion Matrix")
RocCurveDisplay.from_predictions(y_test, probability, ax=axes[1])
axes[1].set_title("ROC Curve")
plt.tight_layout()
plt.show()

PrecisionRecallDisplay.from_predictions(y_test, probability)
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.show()
""")

md("## 15. Model Saving")
code("""model_path = ROOT / "models" / "churn_model.pkl"
loaded = joblib.load(model_path)
print("Saved model:", model_path.relative_to(ROOT))
print("Smoke-test predictions:", loaded.predict(X_test.iloc[:20]).tolist())
""")

md("## 16. Final Conclusion")
code("""final = metrics["final_test"]
print("Accuracy:", f"{final['accuracy']:.2%}")
print("Precision:", f"{final['precision']:.2%}")
print("Recall:", f"{final['recall']:.2%}")
print("F1:", f"{final['f1']:.4f}")
print("ROC-AUC:", f"{final['roc_auc']:.4f}")
print("90% achieved:", metrics["target_90_percent_achieved"])
""")

notebook["cells"] = notebook["cells"]
nbf.write(notebook, ROOT / "notebooks" / "churn_analysis.ipynb")
