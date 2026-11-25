import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "customer_churn.csv"
RANDOM_STATE = 42

df = pd.read_csv(DATA)
df = df.dropna(subset=["Churn"]).copy()
df["Churn"] = df["Churn"].astype(int)

X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
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

numeric_standard = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
numeric_minmax = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_standard, numeric),
    ("categorical", categorical_pipeline, categorical),
])
knn_preprocessor = ColumnTransformer([
    ("numeric", numeric_minmax, numeric),
    ("categorical", categorical_pipeline, categorical),
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        n_jobs=-1,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Linear SVM": CalibratedClassifierCV(
        estimator=LinearSVC(
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        cv=3,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.08,
        max_depth=8,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ),
    "CatBoost": CatBoostClassifier(
        iterations=350,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    ),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
X_cv, _, y_cv, _ = train_test_split(
    X_train_full,
    y_train_full,
    train_size=30000,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

comparison_rows = []
validation_pipelines = {}

print("Training 11 models...")
for name, estimator in models.items():
    start = time.perf_counter()
    pipeline_preprocessor = knn_preprocessor if name == "K-Nearest Neighbors" else preprocessor
    pipeline = Pipeline([
        ("preprocessing", pipeline_preprocessor),
        ("model", estimator),
    ])
    try:
        cv_scores = cross_validate(
            pipeline,
            X_cv,
            y_cv,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
            },
            n_jobs=-1,
        )
        pipeline.fit(X_fit, y_fit)
        probability = pipeline.predict_proba(X_validation)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        row = {
            "Model": name,
            "Accuracy": float(np.mean(cv_scores["test_accuracy"])),
            "Precision": float(np.mean(cv_scores["test_precision"])),
            "Recall": float(np.mean(cv_scores["test_recall"])),
            "F1": float(np.mean(cv_scores["test_f1"])),
            "ROC-AUC": float(np.mean(cv_scores["test_roc_auc"])),
            "Validation Accuracy": accuracy_score(y_validation, prediction),
            "Validation Precision": precision_score(y_validation, prediction, zero_division=0),
            "Validation Recall": recall_score(y_validation, prediction, zero_division=0),
            "Validation F1": f1_score(y_validation, prediction, zero_division=0),
            "Validation ROC-AUC": roc_auc_score(y_validation, probability),
            "Training Seconds": round(time.perf_counter() - start, 2),
            "Status": "Success",
            "Error": "",
        }
        validation_pipelines[name] = pipeline
    except Exception as error:
        row = {
            "Model": name,
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "ROC-AUC": np.nan,
            "Validation Accuracy": np.nan,
            "Validation Precision": np.nan,
            "Validation Recall": np.nan,
            "Validation F1": np.nan,
            "Validation ROC-AUC": np.nan,
            "Training Seconds": round(time.perf_counter() - start, 2),
            "Status": "Error",
            "Error": str(error),
        }
    comparison_rows.append(row)
    print(row, flush=True)

comparison = pd.DataFrame(comparison_rows).sort_values(
    ["Status", "Validation Accuracy", "Validation ROC-AUC"],
    ascending=[True, False, False],
)
comparison.to_csv(ROOT / "model_comparison.csv", index=False)

successful = comparison[comparison["Status"] == "Success"]
selected_name = successful.iloc[0]["Model"]
selected_pipeline = validation_pipelines[selected_name]

tuning_summary = {}
if selected_name == "LightGBM":
    lightgbm_tuning = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )),
    ])
    tuning_grid = {
        "model__num_leaves": [31, 63],
        "model__learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(
        lightgbm_tuning,
        tuning_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
    )
    search.fit(X_fit, y_fit)
    # Grid search selects parameters on training folds only. Clone the fitted
    # best estimator to get a fresh, unfitted pipeline for the final refit.
    selected_pipeline = clone(search.best_estimator_)
    tuning_summary = {
        "best_params": search.best_params_,
        "best_validation_accuracy": float(search.best_score_),
    }
    print("LightGBM tuning:", tuning_summary, flush=True)
elif selected_name == "K-Nearest Neighbors":
    knn_tuning = Pipeline([
        ("preprocessing", knn_preprocessor),
        ("model", KNeighborsClassifier(weights="distance", n_jobs=-1)),
    ])
    search = GridSearchCV(
        knn_tuning,
        {"model__n_neighbors": [5, 15, 25]},
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
    )
    search.fit(X_fit, y_fit)
    selected_pipeline = Pipeline([
        ("preprocessing", knn_preprocessor),
        ("model", KNeighborsClassifier(
            **{key.removeprefix("model__"): value for key, value in search.best_params_.items()},
            weights="distance",
            n_jobs=-1,
        )),
    ])
    tuning_summary = {
        "best_params": search.best_params_,
        "best_validation_accuracy": float(search.best_score_),
    }

oof_probability = cross_val_predict(
    selected_pipeline,
    X_train_full,
    y_train_full,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    method="predict_proba",
    n_jobs=-1,
)[:, 1]
threshold_results = []
for threshold in np.arange(0.30, 0.71, 0.005):
    threshold_prediction = (oof_probability >= threshold).astype(int)
    threshold_results.append((
        accuracy_score(y_train_full, threshold_prediction),
        threshold,
        f1_score(y_train_full, threshold_prediction),
    ))
best_threshold = max(threshold_results, key=lambda item: item[0])[1]

final_model = selected_pipeline
final_model.fit(X_train_full, y_train_full)
if selected_name == "LightGBM":
    # Keep an explicit auditable record of the full data passed to fit. LightGBM
    # releases its training Dataset after fitting, so the fitted sample count is
    # recorded from the actual fit input here.
    fitted_rows = len(X_train_full)
    if fitted_rows != 51499:
        raise RuntimeError(f"Expected 51,499 final training rows, got {fitted_rows}")

test_probability = final_model.predict_proba(X_test)[:, 1]
test_prediction = (test_probability >= best_threshold).astype(int)

baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
baseline.fit(X_train_full, y_train_full)
baseline_prediction = baseline.predict(X_test)
baseline_probability = np.full(len(y_test), y_train_full.mean())

def metric_dict(prediction, probability):
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }

final_metrics = metric_dict(test_prediction, test_probability)
baseline_metrics = metric_dict(baseline_prediction, baseline_probability)

# Feature diagnostics use training rows only; the held-out test set is never
# consulted for feature selection or interpretation.
permutation = permutation_importance(
    final_model,
    X_train_full,
    y_train_full,
    scoring="accuracy",
    n_repeats=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
permutation_importance_summary = {
    feature: {
        "mean_accuracy_drop": float(mean),
        "std_accuracy_drop": float(std),
    }
    for feature, mean, std in sorted(
        zip(X_train_full.columns, permutation.importances_mean, permutation.importances_std),
        key=lambda item: item[1],
        reverse=True,
    )
}
feature_importance_summary = {}
estimator = final_model.named_steps["model"]
if hasattr(estimator, "feature_importances_"):
    feature_names = final_model.named_steps["preprocessing"].get_feature_names_out()
    feature_importance_summary = {
        name: int(value)
        for name, value in sorted(
            zip(feature_names, estimator.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    }

numeric_corr = X_train_full.select_dtypes(include=[np.number]).corr().abs()
high_correlation_pairs = [
    {"feature_a": a, "feature_b": b, "absolute_correlation": float(numeric_corr.loc[a, b])}
    for i, a in enumerate(numeric_corr.columns)
    for b in numeric_corr.columns[i + 1:]
    if numeric_corr.loc[a, b] >= 0.90
]

joblib.dump(final_model, ROOT / "models" / "churn_model.pkl")

metrics = {
    "dataset": {
        "source": "Muhammad Shahid Azeem - Customer Churn Dataset",
        "kaggle_url": "https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset",
        "shape": list(df.shape),
        "target": "Churn",
        "identifier": "CustomerID",
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["CustomerID"].duplicated().sum()),
        "exact_duplicate_feature_rows": int(df.drop(columns=["Churn", "CustomerID"]).duplicated().sum()),
        "conflicting_duplicate_feature_labels": int(
            df.drop(columns=["CustomerID"])
            .groupby(
                df.drop(columns=["CustomerID", "Churn"]).columns.tolist(),
                dropna=False,
            )["Churn"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "simple_support_calls_ge5_accuracy": float(
            accuracy_score(y, (df["Support Calls"] >= 5).astype(int))
        ),
    },
    "split": {
        "train_full_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
        "validation_rows": int(len(X_validation)),
        "cross_validation_subset_rows": int(len(X_cv)),
        "random_state": RANDOM_STATE,
        "oversampling_used": False,
        "synthetic_generation_used": False,
        "test_set_used_for_selection": False,
    },
    "final_model_fit_rows": int(len(X_train_full)),
    "selected_model": selected_name,
    "hyperparameter_tuning": tuning_summary,
    "threshold_selection": {
        "method": "5-fold out-of-fold predictions on 80% training split",
        "metric": "accuracy",
        "threshold": float(best_threshold),
        "frozen_before_test_evaluation": True,
    },
    "feature_diagnostics_training_only": {
        "permutation_importance_accuracy_drop": permutation_importance_summary,
        "model_feature_importance": feature_importance_summary,
        "numeric_pairs_abs_correlation_ge_0_90": high_correlation_pairs,
        "feature_engineering_added": [],
        "feature_selection_change": False,
        "rationale": "No engineered features or feature removals were justified; numeric feature pairs were checked on training data and permutation importance was also computed on training data only.",
    },
    "leakage_audit": {
        "customer_id_excluded_from_features": "CustomerID" not in X.columns,
        "train_test_customer_id_overlap": int(
            len(set(df.loc[X_train_full.index, "CustomerID"]) & set(df.loc[X_test.index, "CustomerID"]))
        ),
        "train_test_feature_row_overlap": int(
            len(
                set(map(tuple, X_train_full.to_numpy())).intersection(
                    set(map(tuple, X_test.to_numpy()))
                )
            )
        ),
        "customer_id_churn_correlation_descriptive_only": float(df["CustomerID"].corr(y)),
        "churn_rate_by_customer_id_decile": {
            str(int(decile)): float(rate)
            for decile, rate in df.assign(
                customer_id_decile=pd.qcut(df["CustomerID"], 10, labels=False)
            ).groupby("customer_id_decile", observed=True)["Churn"].mean().items()
        },
        "interpretation": "CustomerID is excluded, but churn is highly ordered by CustomerID; high-ID deciles are nearly all churn. This suggests target ordering or source generation structure and limits random-split generalization claims. The source data does not establish the target-generation process.",
    },
    "baseline": baseline_metrics,
    "final_test": final_metrics,
    "confusion_matrix": confusion_matrix(y_test, test_prediction).tolist(),
    "classification_report": classification_report(
        y_test,
        test_prediction,
        output_dict=True,
        zero_division=0,
    ),
    "model_path": "models/churn_model.pkl",
    "target_90_percent_achieved": bool(final_metrics["accuracy"] >= 0.90),
}

(ROOT / "model_metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("Selected:", selected_name)
print("Threshold:", best_threshold)
print("Baseline:", baseline_metrics)
print("Final:", final_metrics)
print("Feature diagnostics:", metrics["feature_diagnostics_training_only"])
print("Leakage audit:", metrics["leakage_audit"])
