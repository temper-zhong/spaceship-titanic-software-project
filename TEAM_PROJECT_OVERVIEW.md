# Spaceship Titanic Team Project Overview

## Project Summary

This project solves Kaggle's **Spaceship Titanic** binary-classification challenge. Using passenger demographics, travel information, cabin details, cryogenic-sleep status, and onboard spending, the model predicts whether each passenger was transported to an alternate dimension. Kaggle scores submissions by classification accuracy.

## Current Status

- Data validation and missing-value analysis are complete.
- Feature engineering produces 44 model inputs.
- Three classifiers have been evaluated with grouped five-fold cross-validation.
- An OOF-weighted ensemble has been selected.
- Automated tests and submission-contract checks pass.
- A 4,277-row Kaggle submission can be reproduced from one command.

## Technical Approach

The passenger ID is split into travel-party and member components. Cabin values are split into deck, number, and side. Additional features describe group size, cabin occupancy, surname frequency, age bands, total and log spending, active services, missingness, and group-level passenger behaviour.

Missing values are handled using manifest information and domain rules before median/mode fallbacks. In particular, positive spending is inconsistent with cryogenic sleep, and known members of a travel party can help fill missing travel or cabin fields.

Validation uses `StratifiedGroupKFold`, with passenger groups as the grouping key. This prevents relatives or travelling companions from appearing in both training and validation for the same fold. Preprocessing is fitted inside each fold.

The final ensemble combines logistic regression, Extra Trees, and histogram gradient boosting.

## Results

| Model | OOF accuracy | OOF ROC-AUC |
|---|---:|---:|
| Logistic regression | 0.7947 | 0.8859 |
| Extra Trees | 0.8073 | 0.8965 |
| Histogram gradient boosting | 0.8073 | 0.9010 |
| **Blended ensemble** | **0.8119** | **0.9032** |

The selected model weights are 0.2, 0.3, and 0.5 respectively, with a classification threshold of 0.490. These are local validation results; a Kaggle leaderboard score has not yet been recorded.

## Repository Contents

```text
├── src/train.py              complete training and prediction pipeline
├── tests/test_project.py     data and feature-contract tests
├── raw data/README.md        Kaggle data instructions
├── outputs/metrics.json      full experiment metrics
├── outputs/model_summary.md  generated comparison table
├── README.md                 detailed reproduction guide
├── REPORT.md                 full project report
└── requirements.txt          Python dependencies
```

## Reproduction

1. Clone the private repository.
2. Create and activate a Python 3.11/3.12 virtual environment.
3. Install dependencies with `python -m pip install -r requirements.txt`.
4. Download the three Spaceship Titanic CSV files from Kaggle into `raw data/`.
5. Run `python -m unittest discover -s tests -v`.
6. Run `python -m src.train`.

The final command writes `outputs/submission.csv`, `outputs/metrics.json`, and `outputs/model_summary.md`. Quick installation checks can use `python -m src.train --quick --output-dir outputs-quick`.

## Suggested Team Workflow

Create a short-lived feature branch, run the tests, commit only code and documentation, and open a pull request into `main`. If a modelling change affects validation results, include both the previous and new OOF scores and keep the same fold/seed settings. Do not commit Kaggle source data or generated submission files.
