# Spaceship Titanic Project Report

## 1. Problem Definition

The project addresses Kaggle's Spaceship Titanic binary-classification challenge. Given personal, travel, cabin, and onboard-spending records, the model predicts whether each passenger was transported to an alternate dimension. Kaggle evaluates submissions using classification accuracy:

\[
Accuracy = \frac{\text{correct predictions}}{\text{all predictions}}
\]

The final submission must contain `PassengerId` and Boolean `Transported` columns in exactly the same passenger order as the official sample submission.

## 2. Data Overview

The training data contains 8,693 passengers and 14 columns, including the target. The test data contains 4,277 passengers and the same 13 predictors. The positive target rate is 50.36%, so the target is almost perfectly balanced.

All predictors except `PassengerId` contain missing values, generally between 2.1% and 2.5%. The five monetary variables are `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, and `VRDeck`. Important categorical variables include origin planet, destination, cryogenic-sleep status, VIP status, cabin deck, and cabin side.

The original Kaggle CSV files are excluded from Git history. Team members must download them from the competition page and place them in `raw data/`.

## 3. Data Preparation and Feature Engineering

### 3.1 Passenger and travel-party features

`PassengerId` follows the format `gggg_pp`. The prefix identifies a travel party and the suffix identifies the member within that party. The pipeline extracts both parts and calculates group size and a solo-traveller flag from the passenger manifest.

### 3.2 Cabin and family features

`Cabin` is split into deck, cabin number, and side. Cabin numbers are also grouped into regions, and exact cabin occupancy is counted. The final word of `Name` is treated as a surname for frequency and group-spending summaries; the raw high-cardinality surname is not passed directly to the models.

### 3.3 Age and spending features

The pipeline retains continuous age and adds age bands, child status, and young-adult status. Spending features include total spending, log-transformed spending, luxury spending, food-and-shopping spending, number of active services, and maximum service spending.

### 3.4 Missing-value handling

The imputation sequence applies domain constraints before generic fallbacks. Positive onboard spending implies that a passenger was not in cryogenic sleep. Missing group-related categorical values are inferred from known members of the same passenger group where possible. Missing ages use party and planet medians. Missing spending for cryosleep passengers is set to zero, while remaining spending values use planet medians and a final zero fallback.

The pipeline preserves missingness signals through missing-field counts and original age/cabin missing indicators. No target label is used during feature construction.

## 4. Model and Validation Design

The project compares three complementary estimators:

- Logistic regression provides a stable linear baseline.
- Extra Trees captures nonlinear interactions through randomized tree averaging.
- Histogram gradient boosting incrementally learns nonlinear structure in the engineered table.

All models are implemented as scikit-learn pipelines. Numeric imputation, scaling where applicable, categorical imputation, and one-hot encoding are fitted independently inside each training fold.

Validation uses five-fold `StratifiedGroupKFold`, grouped by passenger-ID prefix. This prevents related passengers from appearing in both a training fold and its validation fold. The stratification component keeps the target rate similar across folds. This design is stricter and more realistic than an ordinary row-level split.

Each estimator generates out-of-fold probabilities and fold-averaged test probabilities. A deterministic grid search selects non-negative model weights and a classification threshold using OOF accuracy.

## 5. Experimental Results

| Model | OOF accuracy | OOF ROC-AUC |
|---|---:|---:|
| Logistic regression | 0.7947 | 0.8859 |
| Extra Trees | 0.8073 | 0.8965 |
| Histogram gradient boosting | 0.8073 | 0.9010 |
| **Blended ensemble** | **0.8119** | **0.9032** |

The selected blend assigns weights of 0.2, 0.3, and 0.5 to logistic regression, Extra Trees, and histogram gradient boosting respectively. Its classification threshold is 0.490, and 52.72% of test passengers are predicted as transported.

The ensemble improves accuracy by approximately 0.46 percentage points over either tree model alone. Its ROC-AUC also increases, indicating stronger probability ranking. These values are local cross-validation results, not Kaggle leaderboard scores.

## 6. Software Quality and Reproducibility

- Random seeds and package versions are recorded.
- Input files, required columns, official sample order, row count, identifier uniqueness, and Boolean output type are validated.
- Unit tests cover official data contracts and feature-engineering output contracts.
- The training entry point supports configurable data/output paths, fold count, and random seed.
- Quick mode provides a smaller three-fold smoke test.
- Final metrics are written both as JSON and as a generated Markdown table.

The committed result used Python 3.12.14, pandas 3.0.1, NumPy 2.5.4, and scikit-learn 1.7.2.

## 7. Limitations

The optimized blend threshold may be mildly overfit to the selected folds. Grouped validation may differ from the hidden Kaggle test distribution. Manifest-level counts use both supplied passenger lists, although no test labels are available or used. The model comparison is also limited to scikit-learn estimators.

Useful follow-up work includes repeated grouped cross-validation, CatBoost, multi-seed ensembles, feature ablation, probability calibration, and recording public/private Kaggle leaderboard scores with matching commit hashes.
