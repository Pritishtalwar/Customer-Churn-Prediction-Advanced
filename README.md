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
8. Select threshold and model using validation data only
9. Evaluate once on the untouched test set
10. Save the complete fitted pipeline
11. Run the Streamlit dashboard

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

The model was selected using validation data and evaluated once on the untouched test set.

### Metric separation

- **Cross-validation:** 3-fold accuracy and ROC-AUC in `model_comparison.csv`
- **Validation:** held-out 20% of the 80% training split
- **OOF threshold:** 5-fold out-of-fold predictions from the full 80% training split
- **Untouched test:** 12,875 rows evaluated once after model and threshold selection
- **Selected threshold:** 0.345

## Target-Pattern and Leakage Review

- `CustomerID` is excluded from all model features.
- No duplicate feature rows or conflicting duplicate feature/label groups were found.
- No oversampling or synthetic generation was used.
- A simple `Support Calls >= 5` rule achieved 66.11% accuracy.
- A `Payment Delay >= 15` rule achieved 76.60% accuracy.
- Shallow decision trees reached 94.25% and 95.56% validation accuracy at depths 4 and 5.
- These findings indicate strong rule-like target structure, but they are not enough to prove target leakage. The data contains no conflicting duplicate labels or identifier leakage in the implemented pipeline.

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
