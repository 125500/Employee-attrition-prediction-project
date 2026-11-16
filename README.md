# Employee Attrition Prediction

A machine learning project that predicts which employees are likely to leave a
company, explains the main drivers of attrition, and serves predictions through
a FastAPI web app so HR can act early.

## Problem
Losing employees is costly (hiring, training, lost productivity). The goal is to
identify employees at high risk of leaving, using HR data such as age, job role,
overtime, income, and satisfaction scores, and to show *why* each prediction was made.

## Dataset
- **Source:** IBM HR Analytics Employee Attrition & Performance (Kaggle)
- **Size:** 1,470 employees, 35 features
- **Target:** `Attrition` (Yes/No), about 16% Yes, so the classes are imbalanced
- **Note:** this is a synthetic dataset created by IBM, so findings are illustrative, not real-world HR conclusions.

## Approach
1. **EDA** (`notebooks/`): distributions, attrition rate by department, overtime, income, etc.
2. **Preprocessing:** drop constant columns, encode categoricals, scale numerics, all inside one sklearn `Pipeline`.
3. **Imbalance handling:** <class weights / SMOTE / threshold tuning, whichever you used>.
4. **Models compared:** <Logistic Regression, Random Forest, ... >.
5. **Evaluation:** recall, precision, F1, and ROC-AUC (accuracy alone is misleading with imbalanced classes).
6. **Explainability:** SHAP summary plots for global and per-employee explanations.
7. **Deployment:** FastAPI app with a simple HTML frontend.

## Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| <model 1> | <> | <> | <> | <> |
| <model 2> | <> | <> | <> | <> |

**Final model:** <name>, chosen because <reason>.

**Key findings (from SHAP):**
- <e.g. overtime is the strongest predictor of attrition>
- <e.g. low monthly income and few years at company increase risk>

![SHAP summary](reports/shap_summary.png)

## Project structure
