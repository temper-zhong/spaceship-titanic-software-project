# Spaceship Titanic Project Report

## Executive summary

This project develops a reproducible binary-classification system for Kaggle's Spaceship Titanic challenge. It validates and audits the official data, engineers 44 shared features, prevents travel-group leakage with `StratifiedGroupKFold`, tunes and evaluates Logistic Regression, Random Forest, XGBoost and CatBoost, blends the two strongest models, and produces a contract-checked Kaggle submission.

The final grouped five-fold OOF accuracy is **0.8158**, compared with 0.4945 for the fold-fitted Dummy baseline, 0.7945 for Logistic Regression, 0.8026 for Random Forest, 0.8100 for XGBoost and 0.8134 for CatBoost. The blend uses 40% XGBoost and 60% CatBoost probability with a threshold of 0.448.

## 1. Problem definition

The target, `Transported`, records whether a passenger was transported to another dimension. Input variables describe identity, travel group, origin, destination, cryogenic sleep, cabin, age, VIP status and onboard spending. Kaggle uses classification accuracy, and the submission must preserve the official test-set ID order with Boolean predictions.

The data contains 8,693 labelled rows and 4,277 unlabelled rows. The target is nearly balanced: 50.36% of training passengers are positive. Every predictor except `PassengerId` contains a small number of missing values.

## 2. Data audit and EDA

The audit checks required columns, file alignment, duplicate rows and IDs, ID/cabin formats, age ranges, spending ranges and the cryosleep-spending domain rule. No structural anomaly was found. Missing rates are between roughly 2.0% and 2.5% and are similar between train and test.

The strongest descriptive patterns are:

- Cryosleep: 81.8% transported when true and 32.9% when false.
- Spending: 78.6% transported among zero-spend passengers and 29.9% among passengers with positive spending.
- Age: 70.0% for children aged 12 or below and 48.3% for older passengers.
- Cabin side: 55.5% on side S and 45.1% on side P.
- Group travel: 56.7% for multi-person groups and 45.2% for solo travellers.

These findings motivate the cryosleep, spending, age, cabin and travel-party features. Full tables and plots are in `reports/EDA_REPORT.md` and `reports/figures/`.

## 3. Preprocessing and feature engineering

Passenger IDs are split into group and member components. Cabin strings are split into deck, number and side, with an additional cabin-region category and occupancy count. Names contribute surname frequency and surname-level spending context without passing raw names to the estimators.

Continuous and indicator features include age, five spending fields, their log transforms, total and log total spending, luxury spending, food/shopping spending, service counts, maximum spending, missingness counts, child/young-adult status, zero-spend status and multiple group aggregates.

Missing values are filled using domain logic and passenger-manifest context before model-level imputation. Positive spending is treated as evidence that a passenger is awake. Group members can supply missing planet, destination, cryosleep and cabin information. Ages use group and planet medians. Cryosleep-passenger spending is set to zero; residual spending uses planet medians and zero fallback.

No target label enters feature construction. Encoders, imputers and scaling are fitted inside each fold.

## 4. Validation and model design

The PassengerId prefix identifies a travel group. Related passengers often share cabins, destinations, spending patterns and outcomes. Random row-level validation could place group members on both sides of a split, so final evaluation uses five-fold `StratifiedGroupKFold` grouped by this prefix.

All four models use the same engineered table and folds:

- A most-frequent Dummy classifier provides the no-skill reference.
- Logistic Regression is the interpretable linear baseline.
- Random Forest learns bagged nonlinear interactions.
- XGBoost learns regularized boosted trees over imputed/encoded features.
- CatBoost uses native categorical inputs and ordered boosting.

A compact grouped three-fold parameter search selects a fixed configuration for each model. Final five-fold training produces OOF probabilities, fold-averaged test probabilities, importance values and timings. A deterministic OOF-only grid then selects ensemble weights and a decision threshold.

## 5. Results

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 | Fold SD |
|---|---:|---:|---:|---:|---:|---:|
| Blended ensemble | **0.8158** | **0.9054** | 0.7893 | **0.8652** | **0.8255** | 0.0150 |
| CatBoost | 0.8134 | 0.9049 | 0.8072 | 0.8271 | 0.8170 | 0.0185 |
| XGBoost | 0.8100 | 0.9035 | **0.8108** | 0.8122 | 0.8115 | **0.0139** |
| Random Forest | 0.8026 | 0.8948 | 0.7979 | 0.8143 | 0.8060 | 0.0164 |
| Logistic Regression | 0.7945 | 0.8859 | 0.7820 | 0.8209 | 0.8010 | 0.0180 |
| Dummy Classifier | 0.4945 | 0.5000 | 0.4988 | 0.7910 | 0.6118 | 0.0115 |

The nonlinear boosted models consistently improve on the linear baseline. CatBoost is the strongest single model, while the XGBoost/CatBoost blend provides the best accuracy, AUC and F1. The ensemble confusion matrix contains 3,304 true negatives, 3,788 true positives, 1,011 false positives and 590 false negatives.

## 6. Interpretation and subgroup errors

Spending, cryosleep, planet and cabin features dominate the model rankings. Random Forest places greatest importance on luxury and aggregate spending. XGBoost ranks the no-spend indicator highest. CatBoost gives the largest importance to luxury spending, followed by home planet and cabin attributes.

The largest subgroup error rates among groups with at least 50 passengers occur for deck G (28.9%), ages 0-12 (28.3%), destination PSO J318.5-22 (26.5%), passengers spending 1-499 (26.0%) and Earth passengers (25.6%). These errors show that high overall accuracy does not imply uniform performance across travel segments.

## 7. Reproducibility and software quality

The repository provides:

- one-command orchestration through `python -m src.run_all`;
- fixed random seeds and recorded environment versions;
- strict source-data and submission validation;
- explicit assertions against group leakage;
- shared features and folds for fair model comparison;
- parameter, fold-score, OOF, importance and error-analysis artifacts;
- automated tests for data, features, splits, models and generated outputs.

The recorded run used Python 3.12.14, pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.7.2, XGBoost 3.4.1 and CatBoost 1.2.10.

## 8. Limitations and future work

The ensemble weights and decision threshold are selected from the same OOF predictions used for reporting and can be mildly optimistic. Manifest aggregates use both unlabelled manifests, which is valid for this competition but differs from single-passenger production inference. Local grouped CV can also differ from Kaggle's hidden distribution.

Future work should evaluate repeated grouped CV across several seeds, perform feature ablations, investigate probability calibration, test group-level consistency rules and record Kaggle public/private scores with the matching commit hash.

## 9. Conclusion

The completed system satisfies the requested workflow from data audit through error analysis. It demonstrates that domain-aware shared features, leakage-resistant validation and complementary boosted models provide a reliable, reproducible solution. The generated submission is ready for Kaggle upload, while the reports and structured outputs support team review and extension.
