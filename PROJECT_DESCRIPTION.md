# Project Description: Spaceship Titanic Passenger Classification

## 1. Project Overview

This project develops a complete and reproducible machine-learning solution for Kaggle's **Spaceship Titanic** competition. The task is to predict whether a passenger was transported to an alternate dimension after the Spaceship Titanic collided with a spacetime anomaly.

The project was completed as part of a software course and was designed to demonstrate both data-science practice and software-engineering quality. It includes data validation, exploratory data analysis, feature engineering, missing-value treatment, grouped cross-validation, multiple classification models, parameter tuning, model interpretation, error analysis, automated testing, and Kaggle submission generation.

The final system compares a Dummy baseline, Logistic Regression, Random Forest, XGBoost, and CatBoost. An ensemble of XGBoost and CatBoost achieves the strongest local result, with a grouped out-of-fold accuracy of **0.8158**.

## 2. Project Objectives

The project has five main objectives:

1. Understand the competition problem and verify the quality of the supplied data.
2. Build a shared and reproducible preprocessing and feature-engineering workflow.
3. Train and compare several classification algorithms under the same validation design.
4. Explain the models and identify passenger groups for which prediction is more difficult.
5. Produce a valid Kaggle submission and a repository that can be reproduced by other team members.

Kaggle evaluates submissions using classification accuracy. Therefore, the primary model-selection metric in this project is grouped out-of-fold accuracy. ROC-AUC, precision, recall, F1, log loss, Brier score, and fold variability are also recorded to provide a broader evaluation.

## 3. Dataset

The official dataset contains three files:

- `train.csv`: 8,693 labelled passengers and 14 columns.
- `test.csv`: 4,277 unlabelled passengers and 13 columns.
- `sample_submission.csv`: the required submission structure.

The target variable is `Transported`, a Boolean value indicating whether a passenger was transported to another dimension. The positive target rate in the training data is 50.36%, so the classes are almost balanced.

The predictors describe:

- passenger and travel-group identity;
- home planet and destination;
- cryogenic-sleep and VIP status;
- cabin deck, number, and side;
- passenger age;
- spending on Room Service, Food Court, Shopping Mall, Spa, and VR Deck;
- passenger name.

Every predictor except `PassengerId` contains some missing values. Missing rates are low, generally between 2.0% and 2.5%, and are similar in the training and test sets.

## 4. Data Audit and Exploratory Analysis

The data-audit stage verifies required columns, file dimensions, passenger-ID uniqueness, sample-submission alignment, passenger-ID and cabin formats, age ranges, spending ranges, duplicate records, and important domain constraints.

The audit found:

- no duplicated rows;
- no duplicated passenger IDs;
- no invalid passenger-ID or cabin formats;
- no negative ages or ages above 100;
- no negative spending values;
- no cryosleep passengers with positive recorded spending.

The exploratory analysis identified several important patterns:

- Passengers in cryogenic sleep have an 81.8% transport rate, compared with 32.9% for awake passengers.
- Passengers with zero recorded spending have a 78.6% transport rate, compared with 29.9% for passengers with positive spending.
- Children aged 12 or below have a 70.0% transport rate, compared with 48.3% for older passengers.
- Passengers on cabin side S have a 55.5% transport rate, compared with 45.1% on side P.
- Passengers travelling in multi-person groups have a 56.7% transport rate, compared with 45.2% for solo travellers.
- Cabin decks B and C have substantially higher transport rates than decks E and T.

These findings guided the feature-engineering strategy. The full audit, tables, and figures are available in [`reports/EDA_REPORT.md`](reports/EDA_REPORT.md).

## 5. Preprocessing and Feature Engineering

All learning models use the same shared feature definitions so that their results can be compared fairly. The pipeline creates 44 model features from the original passenger records.

### Passenger-group features

`PassengerId` follows the format `gggg_pp`. The prefix identifies a travel group and the suffix identifies the passenger's position within that group. The pipeline extracts both components and calculates group size, group mean age, group mean spending, group total spending, group cryosleep rate, and whether the passenger is travelling alone.

### Cabin and family features

The `Cabin` field is separated into deck, cabin number, and cabin side. Cabin number regions and exact cabin occupancy are also calculated. Passenger names are used to obtain surname frequency and surname-level spending summaries, but raw names and raw surnames are not passed directly to the models.

### Age and spending features

The pipeline retains continuous age and adds age bands, a child indicator, and a young-adult indicator. Spending features include the five original service charges, their logarithmic transformations, total spending, log total spending, luxury spending, food-and-shopping spending, maximum service spending, number of active services, a zero-spending indicator, and a positive-spending indicator.

### Missing values

Missing values are handled using domain information before generic imputation:

- Positive onboard spending is treated as evidence that a passenger was awake.
- Known travel-group members can help infer missing planet, destination, cryosleep, and cabin information.
- Cabin deck supplies a fallback signal for home planet where appropriate.
- Missing ages use travel-group and planet medians.
- Missing spending for cryosleep passengers is set to zero.
- Remaining spending values use planet-level medians and a final zero fallback.
- Missingness counts and original missing-value indicators remain available to the models.

No target labels are used during feature construction. Numeric and categorical imputation, scaling, and one-hot encoding are fitted independently inside each training fold.

## 6. Validation Strategy

A standard random row split is not appropriate for this dataset because passengers with the same ID prefix are usually travelling together. Members of one travel group may share cabins, destinations, spending patterns, and outcomes. Placing related passengers in both training and validation data could produce an overly optimistic score.

The final evaluation therefore uses five-fold `StratifiedGroupKFold`:

- every passenger group appears in only one validation fold;
- target balance is kept reasonably consistent across folds;
- every model receives the same training and validation rows;
- preprocessing is fitted only on each fold's training rows;
- the random seed is fixed at `42`.

The project also saves the fold assignments so that the comparison is auditable and can be repeated exactly.

## 7. Models

The project evaluates the following classifiers:

### Dummy Classifier

A fold-fitted class-prior Dummy classifier provides the no-skill reference. It verifies that the trained models learn meaningful structure beyond the class distribution.

### Logistic Regression

Logistic Regression provides an interpretable linear baseline. Numeric features are scaled, while categorical variables are imputed and one-hot encoded.

### Random Forest

Random Forest learns nonlinear relationships and interactions using an ensemble of decision trees. It is robust to many feature distributions and provides impurity-based feature importance.

### XGBoost

XGBoost uses regularized gradient-boosted trees. It performs strongly on structured tabular data and can learn complex interactions among cryosleep, spending, cabin, planet, age, and group variables.

### CatBoost

CatBoost uses native categorical inputs and ordered boosting. It reduces the need for manual categorical encoding and is the strongest individual model in the final experiment.

A compact grouped three-fold search selects the parameters for the four learning models. The final evaluation uses fresh five-fold out-of-fold predictions with the selected parameters.

## 8. Results

The final local results are:

| Model | OOF Accuracy | ROC-AUC | F1 | Fold Accuracy SD |
|---|---:|---:|---:|---:|
| Dummy Classifier | 0.4945 | 0.5000 | 0.6118 | 0.0115 |
| Logistic Regression | 0.7945 | 0.8859 | 0.8010 | 0.0180 |
| Random Forest | 0.8026 | 0.8948 | 0.8060 | 0.0164 |
| XGBoost | 0.8100 | 0.9035 | 0.8115 | 0.0139 |
| CatBoost | 0.8134 | 0.9049 | 0.8170 | 0.0185 |
| **Blended ensemble** | **0.8158** | **0.9054** | **0.8255** | **0.0150** |

The final ensemble combines:

- 40% XGBoost probability;
- 60% CatBoost probability;
- a classification threshold of 0.448.

The blend is selected using only training-set out-of-fold predictions. Test-set predictions do not influence the model or threshold selection.

The ensemble confusion matrix contains 3,304 true negatives, 3,788 true positives, 1,011 false positives, and 590 false negatives. Its recall of 0.8652 is higher than that of every individual model.

These values are local grouped cross-validation results, not Kaggle leaderboard scores. A leaderboard score will only be available after the generated submission is uploaded to Kaggle.

## 9. Interpretation and Error Analysis

The most important features are consistent with the EDA findings. Spending, cryosleep, home planet, cabin deck, cabin side, and travel-group context are important across several models.

Model-specific rankings show that:

- Logistic Regression emphasizes cabin deck, luxury spending, Europa origin, Spa, and VR Deck.
- Random Forest emphasizes luxury spending, aggregate spending, and cabin number.
- XGBoost ranks the zero-spending indicator as its strongest feature.
- CatBoost emphasizes luxury spending, home planet, cabin deck, cabin side, and cabin number.

Subgroup analysis shows that the highest error rates among sufficiently large groups occur for passengers on deck G, children aged 0-12, passengers travelling to PSO J318.5-22, passengers with low positive spending, and passengers from Earth.

These results suggest that future improvements should focus on group consistency, child and family context, deck G passengers, and the transition between zero and low positive spending. The complete analysis is available in [`reports/MODEL_ANALYSIS.md`](reports/MODEL_ANALYSIS.md).

## 10. Software Design and Reproducibility

The repository separates shared pipeline logic from the main workflow stages:

- `src/pipeline.py`: validation, feature engineering, grouped folds, preprocessing, and model construction;
- `src/eda.py`: data-quality tables, EDA figures, and EDA report;
- `src/tune.py`: grouped parameter search;
- `src/train.py`: final OOF training, model comparison, blending, and submission generation;
- `src/analyze.py`: feature importance, confusion matrix, subgroup errors, and model-analysis report;
- `src/run_all.py`: one-command orchestration of the complete workflow.

The complete project can be reproduced with:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_all
```

A reduced smoke test is available with:

```bash
python -m src.run_all --quick --output-dir outputs-quick --report-dir outputs-quick/reports
```

The automated test suite checks official file shapes, feature contracts, grouped-fold isolation, required model construction, probability ranges, fold assignments, passenger-ID alignment, and submission format.

## 11. Team Responsibilities

| Member | Main responsibility |
|---|---|
| M1 | Problem understanding, data validation, anomaly checks, EDA, and EDA findings |
| M2 | Shared preprocessing, feature engineering, fixed grouped folds, Dummy baseline, and Logistic Regression |
| M3 | Random Forest training, tuning, prediction, and interpretation |
| M4 | XGBoost training, tuning, prediction, and interpretation |
| M5 | CatBoost training, tuning, prediction, and interpretation |
| M6 | Unified comparison, confusion matrix, subgroup error analysis, conclusions, and project integration |

The presentation component is intentionally excluded from the current stage of the project.

## 12. Limitations and Future Work

The ensemble weights and decision threshold are selected from the same OOF predictions used for reporting, so the final accuracy may be mildly optimistic. Grouped validation is stricter than a random split but may not perfectly reproduce Kaggle's hidden test distribution. Manifest-level aggregates also assume that the full passenger manifest is available during inference, which is suitable for this competition but may not match a real-time production system.

Possible future improvements include:

- repeated grouped cross-validation over several random seeds;
- probability calibration and threshold-stability analysis;
- feature ablation studies;
- group- and family-consistency post-processing;
- multi-seed CatBoost and XGBoost ensembles;
- recording Kaggle public and private leaderboard scores with the corresponding Git commit.

## 13. Conclusion

This project provides a complete solution to the Spaceship Titanic classification task and demonstrates how careful validation, domain-aware feature engineering, complementary models, and structured error analysis can improve a tabular machine-learning system. The final repository is reproducible, tested, documented, and ready for team review or Kaggle submission.
