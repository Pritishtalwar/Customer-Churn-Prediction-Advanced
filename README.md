# Customer Churn Prediction

An end-to-end telecom customer churn project using the verified Kaggle Customer Churn Dataset.

## Dataset

- Kaggle source: `muhammadshahidazeem/customer-churn-dataset`
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
11. Evaluate once on the untouched test set
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

LightGBM was rebuilt with the GridSearchCV best parameters and fitted on all 51,499 rows in the 80% training split. The threshold was frozen from training-only 5-fold out-of-fold predictions before evaluation on the untouched 12,875-row test split. The test set was not used for model, parameter, threshold, or feature selection.

### Metric separation

- **Cross-validation:** 3-fold accuracy and ROC-AUC in `model_comparison.csv`
- **Validation:** held-out 20% of the 80% training split, used for model comparison
- **OOF threshold:** 5-fold out-of-fold predictions from the full 80% training split
- **Untouched test:** 12,875 rows evaluated once after model and threshold selection
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
