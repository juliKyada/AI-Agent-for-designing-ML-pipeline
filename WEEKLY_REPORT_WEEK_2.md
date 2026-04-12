# WEEKLY REPORT - WEEK 2

## Research Details

| | |
|---|---|
| **Research Title:** | Constraints and Heuristics: Leveraging Dataset Metadata for Efficient AutoML |
| **Student Names and Roll No:** | [To be filled by student] |
| **Mentor 1:** | [To be filled] | **Week Number:** | Week 2 |
| **Mentor 2:** | [To be filled] | **Reporting Period:** | [Start Date] – [End Date] |

---

## Weekly Objectives (3 Marks)

Outline the specific objectives set for the week.

- **Objective 1:** Design and document the MetaFlow system architecture including all 10 processing stages
- **Objective 2:** Define metadata extraction module with 9 groups of metadata parameters
- **Objective 3:** Develop task detection rules for classification vs. regression with confidence scoring

---

## Tasks Completed (4 Marks)

Summarize the key tasks completed during the week.

- **Task 1: Architecture Design** - Designed 10-step MetaFlow pipeline with metadata extraction, task detection, preprocessing, model selection, pipeline generation, and evaluation stages
- **Task 2: Metadata Extraction Module** - Defined 9 metadata groups: dataset info, feature info, target statistics, quality profile, descriptive statistics, outlier profile, cardinality profile, correlation profile, and class-imbalance profile
- **Task 3: Task Detection Logic** - Created rule-based task detector with parameterized thresholds (τ_cls=20, τ_min=10) distinguishing classification (confidence 0.6-1.0) from regression tasks
- **Task 4: System Documentation** - Created architectural diagrams and process flowcharts documenting the complete methodology

---

## Work Progress (3 Marks)

- **Current Status:** 30% completed
- **Milestones Achieved:** 
  - Complete system architecture designed and documented
  - Metadata extraction parameters fully defined
  - Task detection rules finalized with confidence scores
  - System flowchart created (diagram.png)
  
- **Challenges Faced:**
  - Challenge 1: Balancing metadata granularity with computational efficiency
  - Resolution: Prioritized high-impact metadata groups (size, complexity, imbalance) while maintaining comprehensive profiling
  - Challenge 2: Defining appropriate thresholds for task detection
  - Resolution: Used empirical analysis from related work to establish default parameters

---

## Plan for Next Week

Outline the objectives and tasks planned for the upcoming week.

- Implement data preprocessing module (row-level cleaning and pipeline transformations)
- Develop rule-based model selector with dataset-size tiers and complexity scoring
- Create modifier rules (binary classification, high-dimensionality, class imbalance, etc.)
- Begin coding foundational Python modules

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

- MetaFlow architecture diagram (diagram.png)
- System flowchart with 10 stages
- Metadata extraction parameter specifications
- Task detection rule hierarchy with thresholds
