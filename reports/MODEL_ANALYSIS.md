# Model Comparison and Error Analysis

## Model comparison

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 | Fold accuracy SD |
|---|---:|---:|---:|---:|---:|---:|
| Blended ensemble | 0.8158 | 0.9054 | 0.7893 | 0.8652 | 0.8255 | 0.0150 |
| CatBoost | 0.8134 | 0.9049 | 0.8072 | 0.8271 | 0.8170 | 0.0185 |
| XGBoost | 0.8100 | 0.9035 | 0.8108 | 0.8122 | 0.8115 | 0.0139 |
| Random Forest | 0.8026 | 0.8948 | 0.7979 | 0.8143 | 0.8060 | 0.0164 |
| Logistic regression | 0.7945 | 0.8859 | 0.7820 | 0.8209 | 0.8010 | 0.0180 |
| Dummy classifier | 0.4945 | 0.5000 | 0.4988 | 0.7910 | 0.6118 | 0.0115 |

The highest grouped OOF accuracy is 0.8158, achieved by Blended ensemble. Fold standard deviation is reported because mean accuracy alone can hide unstable splits.

## Ensemble confusion matrix

| Actual | Predicted | Count |
|---|---|---:|
| False | False | 3,304 |
| False | True | 1,011 |
| True | False | 590 |
| True | True | 3,788 |

False positives and false negatives are reported separately in the subgroup table. This matters because a model can preserve overall accuracy while failing systematically for one passenger segment.

## Highest-error subgroups

Only groups with at least 50 passengers are shown below.

| Segment | Value | Count | Error rate | False positives | False negatives |
|---|---|---:|---:|---:|---:|
| CabinDeck | G | 2,559 | 28.9% | 593 | 146 |
| AgeBand | 0-12 | 806 | 28.3% | 184 | 44 |
| Destination | PSO J318.5-22 | 796 | 26.5% | 161 | 50 |
| SpendBand | 1-499 | 104 | 26.0% | 11 | 16 |
| HomePlanet | Earth | 4,602 | 25.6% | 785 | 395 |
| GroupSizeBand | 5+ | 774 | 23.6% | 127 | 56 |
| CabinDeck | E | 876 | 21.6% | 99 | 90 |
| HomePlanet | Missing | 201 | 21.4% | 23 | 20 |
| SpendBand | 500-1999 | 3,263 | 20.5% | 272 | 397 |
| AgeBand | 18-25 | 2,351 | 20.3% | 283 | 195 |

## Feature interpretation

Importance values are model-specific and should be interpreted as rankings within a model, not compared numerically across algorithms.

| Model | Five highest-ranked features |
|---|---|
| Logistic regression | CabinDeck_C, LuxurySpend, HomePlanet_Europa, Spa, VRDeck |
| Random Forest | LuxurySpend, FoodShoppingSpend, TotalSpend, LogTotalSpend, MaximumServiceSpend |
| XGBoost | NoSpend, HomePlanet_Earth, LuxurySpend, ActiveServiceCount, CryoSleep_False |
| CatBoost | LuxurySpend, HomePlanet, CabinDeck, CabinSide, CabinNumber |

## Conclusions

1. Nonlinear boosted models improve on the logistic baseline, confirming that interactions among cryosleep, spending, cabin, age, and group context matter.
2. The ensemble reduces model-specific errors and provides the strongest grouped OOF accuracy.
3. Low-confidence predictions and the worst demographic/travel subgroups should be reviewed before further feature work.
4. Feature importance supports the EDA findings but does not establish causality.
5. Kaggle leaderboard performance must be recorded separately because grouped local CV and the hidden test distribution are not identical.

## Artifacts

- `outputs/model_comparison.csv`
- `outputs/fold_scores.csv`
- `outputs/confusion_matrix.csv`
- `outputs/error_analysis.csv`
- `outputs/feature_importance.csv`
- `reports/figures/model_comparison.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png`
- `reports/figures/subgroup_error_rates.png`
