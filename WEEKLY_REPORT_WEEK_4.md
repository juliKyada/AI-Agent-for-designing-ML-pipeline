# WEEKLY REPORT - WEEK 4

## Research Details

| | |
|---|---|
| **Research Title:** | Constraints and Heuristics: Leveraging Dataset Metadata for Efficient AutoML |
| **Student Names and Roll No:** | [To be filled by student] |
| **Mentor 1:** | [To be filled] | **Week Number:** | Week 4 |
| **Mentor 2:** | [To be filled] | **Reporting Period:** | [Start Date] – [End Date] |

---

## Weekly Objectives (3 Marks)

Outline the specific objectives set for the week.

- **Objective 1:** Implement training, evaluation, and pipeline optimization modules
- **Objective 2:** Execute experiments on 4 case study datasets (Heart Disease, Credit Card Fraud, Brain Cancer Gene Expression, House Sales)
- **Objective 3:** Validate research questions (RQ1: Efficacy, RQ2: Computational Efficiency)

---

## Tasks Completed (4 Marks)

Summarize the key tasks completed during the week.

- **Task 1: Pipeline Generation Module** - Implemented PipelineGenerator that assembles scikit-learn pipelines with ColumnTransformer preprocessing and model instances with hyperparameter grids
- **Task 2: Training and Evaluation Module** - Coded training logic with 80-20 stratified splitting, 5-fold cross-validation, and grid search hyperparameter tuning
- **Task 3: Metrics Computation** - Implemented evaluation metrics: Classification (accuracy, precision, recall, F1-score weighted), Regression (R², MSE, RMSE, MAE, MAPE)
- **Task 4: Issue Detection System** - Built automated detection for: low performance (threshold 0.70 classification, 0.60 regression), overfitting (train-test gap > 0.10), underfitting, high variance (CV std > 0.15)
- **Task 5: Case Study 1 (D1 - Heart Disease)** - Tested on N=1,025, p=13 binary classification. Result: LightGBM selected (Priority 1), CV Score=0.9720, Test Accuracy=1.0000
- **Task 6: Case Study 2 (D2 - Credit Card Fraud)** - Tested on N=284,807, p=30 with 0.17% minority class. Result: Reduced search space 45%, selected LGBM/XGBoost/RF with balanced class weights, CV Score=0.9982-0.9983
- **Task 7: Case Study 3 (D3 - Brain Cancer)** - Tested on N=130, p=54,676 (p/n≈421). Result: Selected lightweight models (DT/LR/NB), Decision Tree best (Test Accuracy=0.9615)
- **Task 8: Case Study 4 (D4 - House Sales)** - Tested on N=21,613, p=20 regression. Result: XGBoost selected (CV R²=0.8546, Test R²=0.8453)

---

## Work Progress (3 Marks)

- **Current Status:** 80% completed
- **Milestones Achieved:** 
  - All core modules implemented and integrated
  - 4 case studies completed with comprehensive results
  - Computational efficiency demonstrated (45-67% reduction in model evaluations)
  - Research questions validated through experiments
  - Performance metrics collected and analyzed
  
- **Challenges Faced:**
  - Challenge 1: High-dimensional data (D3) causing computational issues
  - Resolution: Applied top-1,000 feature selection by variance (reduced p/n from 421 to 7.7)
  - Challenge 2: Class imbalance in fraud dataset affecting model performance
  - Resolution: Implemented class_weight='balanced' in hyperparameter grids per Rule 6
  - Challenge 3: Baseline comparison implementation
  - Resolution: Defined exhaustive baseline of 9 algorithms to compare against MetaFlow's rule-based selection

---

## Plan for Next Week

Outline the objectives and tasks planned for the upcoming week.

- Create comprehensive performance comparison charts and tables
- Document all 4 case study results with detailed analysis
- Prepare research findings and answer research questions
- Generate final system demonstration and workflow documentation
- Compile paper and supporting materials for publication

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

- trainer.py and evaluator.py source code
- Case Study 1: Heart Disease results and analysis (Table 3)
- Case Study 2: Credit Card Fraud results and model comparison (Table 4)
- Case Study 3: Brain Cancer Gene Expression results (Table 5)
- Case Study 4: King County House Sales results (Table 6)
- Metadata heatmap for all 4 datasets (Table 7)
- Search space reduction analysis figure (Figure 2 - Bar chart D1-D4)
- Detailed performance metrics and cross-validation scores
