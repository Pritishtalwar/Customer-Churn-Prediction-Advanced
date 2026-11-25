# Customer Churn Prediction

An end-to-end telecom customer churn project using a Kaggle-sourced customer churn dataset.

## Dataset

- Kaggle source: `muhammadshahidazeem/customer-churn-dataset`
- Source page: [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)
- Dataset license: **Unverified.** The repository does not include license metadata, and the Kaggle source page did not expose readable license details during the final audit. Do not assume public-domain or redistribution rights; verify the current Kaggle dataset license before republishing the CSV.
- Repository code license: **Unspecified.** No top-level `LICENSE` file is present.
- Verified project dataset: 64,374 rows
- 12 source columns
- Target: `Churn`
- Identifier: `CustomerID`
- No duplicate rows
- No duplicate customer IDs
- No missing values
- CustomerID is excluded from model features

The 505,207-row value resulted from accidentally combining the 440,833-row training-master and 64,374-row testing-master CSV files. This project uses the original testing-master data as the working dataset and performs an independent 80:20 stratified split.

## Technology

- Python
- Pandas and NumPy
- Matplotlib, Seaborn and Plotly
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Joblib
- Streamlit
- Jupyter

## Workflow

1. Verify the original Kaggle dataset
2. Audit data quality and target balance
3. Perform EDA
4. Split 80:20 with `random_state=42`
5. Fit preprocessing only on training data
6. Compare 11 classification models
7. Use 3-fold cross-validation
8. Select model and hyperparameters using validation/training folds only
9. Select and freeze the threshold using 5-fold OOF predictions on the 80% training split
10. Rebuild the selected pipeline with best parameters and refit on all 51,499 training rows
11. Evaluate on the held-out test split after model and threshold selection (the fixed split is re-evaluated during reproducibility checks)
12. Save the complete fitted pipeline
13. Run the Streamlit dashboard

## Final Results

| Metric | Dummy Baseline | Tuned LightGBM |
|---|---:|---:|
| Accuracy | 52.63% | 99.9845% |
| Precision | 0.00% | 99.9672% |
| Recall | 0.00% | 100.00% |
| F1-score | 0.0000 | 0.999836 |
| ROC-AUC | 50.00% | 100.00% |

Confusion matrix:

```text
[[6774, 2],
 [   0, 6099]]
```

The results above were recalculated from the saved LightGBM pipeline on the fixed, stratified 12,875-row test split (`random_state=42`). The saved model and threshold were not changed using test results. LightGBM was fitted on all 51,499 training rows, and the **0.345 threshold** was selected from 5-fold out-of-fold predictions on those training rows before test evaluation. The test partition was not used for model, parameter, threshold, or feature selection. It has been evaluated repeatedly during development and reproducibility checks, so it is a fixed held-out benchmark rather than a one-time confirmatory evaluation.

Dummy baseline confusion matrix (rows = actual [stay, churn], columns = predicted [stay, churn]):

```text
[[6776, 0],
 [6099, 0]]
```

### Test classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Stay (0) | 1.0000 | 0.9997 | 0.9999 | 6,776 |
| Churn (1) | 0.9997 | 1.0000 | 0.9998 | 6,099 |
| Macro average | 0.9998 | 0.9999 | 0.9998 | 12,875 |
| Weighted average | 0.9998 | 0.9998 | 0.9998 | 12,875 |

The complete model and baseline classification reports, confusion matrices, threshold provenance, and split details are saved in `evaluation_results.json` and `model_metrics.json`.

### Metric separation

- **Cross-validation:** 3-fold accuracy and ROC-AUC in `model_comparison.csv`
- **Validation:** held-out 20% of the 80% training split, used for model comparison
- **OOF threshold:** 5-fold out-of-fold predictions from the full 80% training split
- **Held-out test:** fixed 12,875-row split evaluated after model and threshold selection; repeated evaluations are disclosed above
- **Selected threshold:** 0.345 (saved in `model_metrics.json` and used by the app)

## Target-Pattern and Leakage Review

- `CustomerID` is excluded from all model features.
- No duplicate feature rows or conflicting duplicate feature/label groups were found.
- No oversampling or synthetic generation was used.
- A simple `Support Calls >= 5` rule achieved 66.11% accuracy.
- A `Payment Delay >= 15` rule achieved 76.60% accuracy.
- Shallow decision trees reached 94.25% and 95.56% validation accuracy at depths 4 and 5.
- The CSV does not include target-generation logic, so its provenance cannot be verified here. `CustomerID` is excluded from model inputs, but churn is strongly ordered by ID: churn rates across ascending ID deciles are 25.9%, 26.5%, 25.3%, 25.3%, 26.1%, 25.8%, 25.9%, 92.9%, 100%, and 100%. ID/churn correlation is 0.530 (descriptive audit only). This can explain very high random-split scores and limits claims about future customers or other sources. It is an ordering/provenance concern, not proof of direct model leakage.
- No exact duplicate feature rows, conflicting labels for identical feature rows, duplicate IDs, or duplicate full rows were found. The single-CSV stratified split has no shared row samples; related source records cannot be ruled out because no external customer/time key is available.
- Training-only permutation importance ranks Payment Delay and Support Calls highest. No numeric feature pairs had absolute correlation >= 0.90. No derived features or feature removals were justified, so the original 10 predictors remain.

### Detailed Data Leakage Audit

The reproducible audit is in `scripts/data_leakage_audit.py`; its findings are saved in `data_leakage_audit.json`. The script recreates the stratified 80:20 split first. It fits an importance-only clone on 41,199 fit rows, calculates permutation importance on the separate 10,300-row validation partition, and does not pass test rows or labels to either importance calculation. The saved model's native LightGBM split-count importance is also reported (the saved model was fit only on training rows).

| Predictor | Validation permutation accuracy drop | LightGBM split-count importance |
|---|---:|---:|
| Payment Delay | 0.2843 | 260 |
| Support Calls | 0.1551 | 304 |
| Tenure | 0.1188 | 490 |
| Gender | 0.0792 | 153 |
| Usage Frequency | 0.0671 | 689 |
| Total Spend | 0.0477 | 323 |
| Contract Length | 0.0411 | 382 |
| Age | 0.0360 | 239 |
| Subscription Type | 0.0023 | 140 |
| Last Interaction | -0.00001 | 20 |

The highest validation accuracy for any single-feature depth-1 tree was Payment Delay at 78.43%, followed by Support Calls at 66.11%; neither alone reveals the label. These are predictive associations, not causal evidence. The CSV and repository contain no target-generation code or documentation, so whether churn was derived from behavioral features or generated by rules cannot be established. ID-decile churn rates in the 80% training partition are also sharply ordered (about 25–27% in deciles 0–6, 92.9% in decile 7, and 100% in deciles 8–9); `CustomerID` itself is excluded from the model. This suggests source ordering or target-construction structure and makes random-split evaluation potentially optimistic. The audit found zero duplicate full rows, IDs, feature vectors, conflicting feature/label groups, train/validation or train/test ID overlaps, and train/test feature-vector overlaps. No oversampling or synthetic resampling is present; splitting happens before fitting and no resampling package is in the requirements.

Importance reports: `feature_importance_report.csv` contains encoded-column split counts; `feature_importance_by_source_feature.csv` aggregates them to input fields; `permutation_importance_report.csv` contains validation accuracy drops; `univariate_target_signal_report.csv` records fit/validation depth-1 results. Permutation importance is model- and validation-sample-dependent; it does not prove causality, and correlated fields can share or mask importance. No features were removed or added based on this audit.

## Model Comparison

| Model | 3-fold Accuracy | Validation Accuracy | Validation F1 | Validation ROC-AUC |
|---|---:|---:|---:|---:|
| LightGBM | 99.96% | 99.9903% | 99.9898% | 100.00% |
| CatBoost | 99.9733% | 99.9709% | 99.9692% | 100.00% |
| XGBoost | 99.9633% | 99.9709% | 99.9693% | 100.00% |
| HistGradientBoosting | 99.9367% | 99.9806% | 99.9795% | 100.00% |
| Random Forest | 99.5367% | 99.8252% | 99.8154% | 99.9991% |
| Gradient Boosting | 99.1933% | 99.2524% | 99.2079% | 99.9134% |
| Decision Tree | 99.0500% | 99.2816% | 99.2410% | 99.9775% |
| Extra Trees | 95.8567% | 96.6505% | 96.4978% | 99.6206% |
| K-Nearest Neighbors | 89.5933% | 91.1165% | 90.9576% | 97.6629% |
| Linear SVM | 82.5900% | 82.9612% | 82.2314% | 90.5799% |
| Logistic Regression | 82.5833% | 82.9709% | 82.2362% | 90.5873% |

## Run Training

```powershell
python scripts/train_models.py
```

## Run EDA

```powershell
python scripts/eda_analysis.py
```

## Run the Notebook

```powershell
python -m jupyter notebook notebooks/churn_analysis.ipynb
```

## Run the App

```powershell
streamlit run app/app.py
```
