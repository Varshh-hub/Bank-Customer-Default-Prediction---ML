# Loan Default Prediction using Machine Learning

## Overview

This project develops a machine learning classification model to predict whether a loan applicant is likely to **default on a loan**.

The project uses a real-world loan dataset containing financial, demographic, credit, and loan-related attributes. The workflow covers data cleaning, exploratory analysis, preprocessing, outlier treatment, model training, hyperparameter tuning, evaluation, and prediction on new applicant profiles.

The final model uses **Logistic Regression** with preprocessing integrated through a Scikit-learn Pipeline.

---

## Problem Statement

Loan default prediction is an important task in financial risk management. Incorrectly approving high-risk applicants can lead to financial losses, while rejecting low-risk applicants can result in missed business opportunities.

The objective of this project is to build a classification model that can identify applicants who are more likely to default based on their available loan and financial information.

---

## Dataset

The dataset contains **148,670 records and 34 features** before preprocessing.

The target variable is:

* `Status`

  * `0` → Non-default
  * `1` → Default

---

## Machine Learning Model

### Logistic Regression

The primary model used in this project is **Logistic Regression**.

The model was configured with:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=3000,
    random_state=42
)
```

Using `class_weight="balanced"` helps account for the imbalance between default and non-default applicants.

---

## Hyperparameter Tuning

`RandomizedSearchCV` was used to optimize the Logistic Regression model.

The parameters searched were:

* `C`
* `solver`

The search used:

* 8 parameter combinations
* 3-fold cross-validation
* F1-score as the optimization metric
* `n_jobs=-1`
* `random_state=42`

### Best Parameters

```text
solver = lbfgs
C ≈ 0.2848
```

The best cross-validation F1-score was approximately:

**0.6798**

---

## Model Performance

The tuned Logistic Regression model achieved the following results on the test set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 83.57% |
| Precision | 65.41% |
| Recall    | 70.79% |
| F1-Score  | 68.00% |
| ROC-AUC   | 86.35% |

The classification report shows that the model performs particularly well in identifying the majority non-default class, while also achieving meaningful recall for the default class.

---

## Key Takeaways

* Performed end-to-end preprocessing on a large loan dataset.
* Handled missing values using feature-specific imputation strategies.
* Applied IQR-based outlier capping.
* Used separate preprocessing pipelines for numerical and categorical features.
* Addressed class imbalance using balanced Logistic Regression.
* Optimized model hyperparameters using `RandomizedSearchCV`.
* Evaluated the model using multiple classification metrics.
* Tested the model on manually created applicant profiles.
* Generated default probabilities in addition to class predictions.

---

## Limitations

This project is intended for **educational and analytical purposes** and should not be used as the sole basis for real-world lending decisions.

Loan approval involves additional considerations such as regulatory requirements, fairness, explainability, economic conditions, and detailed financial verification.

The model's predictions should therefore be treated as **risk-assessment outputs rather than definitive lending decisions**.

---

## Author

**Varsha A.**

Aspiring Data Scientist | Machine Learning | Data Analytics

---
