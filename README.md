## Results

Three models were compared using 5-fold cross-validation and a held-out test set
(294 employees: 247 stay, 47 leave).

| Model | CV AUC | Test AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.827** | **0.803** | 0.349 | **0.638** | **0.451** |
| Random Forest | 0.810 | 0.791 | 0.500 | 0.362 | 0.420 |
| XGBoost | 0.796 | 0.774 | 0.500 | 0.404 | 0.447 |

Precision, recall and F1 are for the "Leaves" class.

### Why Logistic Regression?
- Highest CV and test ROC-AUC, with a small gap between them (stable, not overfitting).
- Highest recall for leavers (0.64 vs 0.36 and 0.40). The goal is to catch employees
  at risk of leaving, so missing fewer of them matters most.
- Simple, fast and interpretable; on a small dataset (1,470 rows) the more complex
  models gave no improvement.
- Trade-off: precision is low (0.35), so many flagged employees will not actually
  leave. This is acceptable for an early-warning tool, where a check-in
  conversation costs far less than losing an employee.

### Final model performance (test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Stays | 0.92 | 0.77 | 0.84 | 247 |
| Leaves | 0.35 | 0.64 | 0.45 | 47 |

Overall accuracy is 0.75. Accuracy alone is not a good metric here because only
about 16% of employees leave (always predicting "Stays" would score about 84%
accuracy but catch no leavers), so recall and ROC-AUC were used to choose the model.

### Key findings (SHAP)
- <top 2-3 features from your SHAP plot, e.g. OverTime, MonthlyIncome, ...>

![SHAP summary](reports/shap_summary.png)
