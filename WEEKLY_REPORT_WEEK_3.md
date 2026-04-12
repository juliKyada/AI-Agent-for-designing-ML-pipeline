# WEEKLY REPORT - WEEK 3

## Research Details

| | |
|---|---|
| **Research Title:** | Constraints and Heuristics: Leveraging Dataset Metadata for Efficient AutoML |
| **Student Names and Roll No:** | [To be filled by student] |
| **Mentor 1:** | [To be filled] | **Week Number:** | Week 3 |
| **Mentor 2:** | [To be filled] | **Reporting Period:** | [Start Date] – [End Date] |

---

## Weekly Objectives (3 Marks)

Outline the specific objectives set for the week.

- **Objective 1:** Implement data preprocessing module with row-level cleaning and feature-type-specific transformations
- **Objective 2:** Develop rule-based model selector with dataset-size tiers (tiny, small, medium, large, huge) and complexity scoring
- **Objective 3:** Create modifier rules system for handling binary classification, high-dimensionality, class imbalance, outliers, and skewed distributions

---

## Tasks Completed (4 Marks)

Summarize the key tasks completed during the week.

- **Task 1: Data Preprocessing Implementation** - Coded DataPreprocessor module with feature dropping (missing ratio > 0.5), median/mode imputation, and empty row removal
- **Task 2: Pipeline Transformations** - Implemented ColumnTransformer with StandardScaler for numerical features and OneHotEncoder for categorical features
- **Task 3: Rule-Based Model Selector** - Created RuleBasedModelSelector with dataset-size binning (5 categories) and complexity scoring formula (components: high dimensionality, p/n ratio, categorical dominance, feature count)
- **Task 4: Classification Model Pools** - Defined size-tier specific model pools: Tiny (LR, DT, RF, NB, Ridge-LR, KNN), Small-Med (RF, XGB, LGBM, GB, LR, KNN, NB), Large-Huge (LGBM, XGB, GB, LR, RF, Ridge-LR, KNN, NB)
- **Task 5: Regression Model Pools** - Defined regression pools: Tiny (OLS, Ridge, DT, Lasso, ElasticNet, KNN), Small-Med (RF, XGB, LGBM, GB, Ridge, KNN, SVR), Large-Huge (LGBM, XGB, GB, Ridge, RF, Lasso, ElasticNet, SVR)
- **Task 6: Modifier Rules Implementation** - Coded 9 modifier rules (binary classification, high dimensionality, class imbalance, high-cardinality categoricals, outlier density, simple/complex patterns, skewed distributions)

---

## Work Progress (3 Marks)

- **Current Status:** 55% completed
- **Milestones Achieved:** 
  - Complete preprocessing module implemented and tested
  - Rule-based model selector fully functional with base selection rules
  - All 9 modifier rules coded and integrated
  - Model priority ranking system developed
  - Hyperparameter grids defined for each algorithm
  
- **Challenges Faced:**
  - Challenge 1: Balancing model diversity with computational efficiency across different dataset sizes
  - Resolution: Used tiered approach with decreasing ensemble prevalence for tiny datasets and increasing ensemble/tree models for larger datasets
  - Challenge 2: Defining appropriate thresholds for complexity scoring and rule triggers
  - Resolution: Empirically validated thresholds using preliminary data analysis

---

## Plan for Next Week

Outline the objectives and tasks planned for the upcoming week.

- Implement pipeline generation module
- Develop training and evaluation module with cross-validation
- Create hyperparameter tuning with grid search
- Implement model evaluation metrics (accuracy, precision, recall, F1 for classification; R², MSE, RMSE, MAE for regression)
- Build issue detection system (overfitting, underfitting, low performance, high variance)

---

## Remarks from Mentors

**Academic Mentor's Feedback:**

[To be filled by Mentor 1]

---

| | |
|---|---|
| **Student Signature:** | _________________ |
| **Mentor 1 Signature:** | _________________ | **Out of 10 Marks:** | _______ |
| **Mentor 2 Signature:** | _________________ | **Out of 10 Marks:** | _______ |

---

## Annexure (Supporting Documents)

- Preprocessing module code (preprocessor.py)
- Dataset-size tier mapping table
- Complexity scoring formula and examples
- Classification model pool specifications (Table 1)
- Regression model pool specifications (Table 2)
- Modifier rules documentation (Rules 1-9)
- Sample hyperparameter grid configurations
