# Spaceship Titanic Prediction

Predicting whether a passenger was transported to an alternate dimension, in Python. Kaggle metric: **classification accuracy**.

This repository contains a complete, reproducible machine-learning pipeline for the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic). It covers data validation, domain-aware missing-value handling, feature engineering, leakage-resistant cross-validation, model comparison, probability blending, automated tests, and submission generation.

---

## 1. The Task

The Spaceship Titanic collided with a spacetime anomaly while carrying passengers to distant planets. Given records recovered from the ship's computer system, the task is to predict the binary target `Transported` for every passenger in the test set.

| Item | Value |
|---|---:|
| Training set | 8,693 labelled passengers |
| Test set | 4,277 unlabelled passengers |
| Raw predictors | 13 |
| Engineered model features | 44 |
| Metric | Accuracy |
| Training positive rate | 50.36% |
| Validation | 5-fold `StratifiedGroupKFold` |

The target is nearly balanced, so accuracy is informative without class reweighting. The important modelling challenges are missing data, related passengers, mixed numeric/categorical inputs, and strong interactions between cryogenic sleep and onboard spending.

### Raw features

Every predictor except `PassengerId` has missing values in the training set.

| Feature | Meaning | Training missing rate |
|---|---|---:|
| `PassengerId` | Passenger identifier in `gggg_pp` format | 0.00% |
| `HomePlanet` | Planet of departure | 2.31% |
| `CryoSleep` | Whether the passenger entered suspended animation | 2.50% |
| `Cabin` | Cabin in `deck/number/side` format | 2.29% |
| `Destination` | Destination planet | 2.09% |
| `Age` | Passenger age | 2.06% |
| `VIP` | Whether VIP service was purchased | 2.34% |
| `RoomService` | Room-service charges | 2.08% |
| `FoodCourt` | Food-court charges | 2.11% |
| `ShoppingMall` | Shopping-mall charges | 2.39% |
| `Spa` | Spa charges | 2.11% |
| `VRDeck` | VR-deck charges | 2.16% |
| `Name` | Passenger name | 2.30% |

**The competition data is not stored in this repository.** Download it from Kaggle as described in [Reproducing the results](#5-reproducing-the-results).

---

## 2. Repository Structure

```text
spaceship-titanic-software-project/
├── raw data/
│   └── README.md              data download and placement instructions
├── src/
│   ├── __init__.py
│   └── train.py               validation, features, models, blending, outputs
├── tests/
│   └── test_project.py        source-data and feature-contract tests
├── outputs/
│   ├── metrics.json           machine-readable final experiment results
│   └── model_summary.md       compact model comparison table
├── README.md                  detailed reproduction guide
├── REPORT.md                  methods, findings, and limitations
├── TEAM_PROJECT_OVERVIEW.md   short hand-off document for team members
└── requirements.txt           Python dependencies
```

The generated `outputs/submission.csv` and the three Kaggle CSV files are ignored by Git. This keeps competition data and generated predictions out of repository history while preserving all code needed to rebuild them.

---

## 3. Pipeline Design

The entire pipeline is implemented in `src/train.py`. One command performs the following steps:

1. Validate the three source files and their required columns.
2. Build manifest-level and passenger-level features without using target labels.
3. Construct five grouped, stratified validation folds.
4. Fit preprocessing separately inside every training fold.
5. Train three complementary classifiers.
6. Collect out-of-fold and test-set probabilities.
7. Select ensemble weights and a classification threshold from OOF predictions.
8. Validate the final submission against `sample_submission.csv`.
9. Write the submission, metrics, and Markdown results summary.

### 3.1 Feature engineering

The 44 model features include:

| Feature family | Examples |
|---|---|
| Travel party | group ID components, member number, group size, solo-travel flag |
| Cabin | deck, cabin number, side, cabin region, cabin occupancy |
| Family | surname frequency and surname-level mean spending |
| Demographics | imputed age, age band, child flag, young-adult flag |
| Spending | five original services, total spend, log spend, luxury spend, active-service count, maximum service spend |
| Missingness | total missing count, missing service count, original age/cabin missing flags |
| Group context | group mean age, mean spend, total spend, and cryosleep rate |

High-cardinality raw identifiers, full names, and raw surnames are not fed directly to the models. They are converted into compact numeric or categorical summaries instead.

### 3.2 Missing data and domain constraints

Imputation uses information available in the passenger manifest, never the target label.

- A passenger with positive onboard spending cannot be treated as being in cryogenic sleep.
- Missing planet, destination, cryosleep, deck, and side values are first inferred from known members of the same travel group where possible.
- Cabin deck can provide a fallback signal for `HomePlanet`, and `HomePlanet` provides a fallback deck when the cabin is absent.
- Missing ages use group and planet medians.
- Missing spending for cryosleep passengers is set to zero; remaining spending values use planet-level medians and then zero where necessary.
- Missingness counts and original missing flags remain available to the model.

Train and test rows are combined only for **unlabelled manifest statistics**, such as travel-party size. No test labels exist, and no target-derived values are used in feature engineering.

### 3.3 Validation discipline

The passenger ID prefix identifies a travel group. Related passengers may share destination, cabin, spending behaviour, and outcome. A random row-level split could put members of one group into both the training and validation sets, producing an overly optimistic estimate.

This project therefore uses five-fold `StratifiedGroupKFold`:

- `Stratified` keeps the target balance similar across folds.
- `Group` keeps every passenger group entirely within one fold.
- Encoders and imputers are fitted inside each fold through scikit-learn pipelines.
- The random seed is fixed at `42` by default.

This validation scheme is stricter than ordinary stratified cross-validation and better measures performance on unseen travel parties.

### 3.4 Models and ensemble

| Model | Role in the ensemble |
|---|---|
| Logistic regression | Stable linear baseline after scaling and one-hot encoding |
| Extra Trees | Nonlinear interactions and robust tree averaging |
| Histogram gradient boosting | Sequential nonlinear learning for structured tabular data |

Each model generates OOF probabilities and fold-averaged test probabilities. The script searches a small deterministic grid of non-negative ensemble weights and thresholds using OOF accuracy. The selected blend gives the strongest local validation result while remaining easy to reproduce.

---

## 4. Results

The committed results were produced with five grouped folds and seed `42`.

| Model | OOF accuracy | OOF ROC-AUC |
|---|---:|---:|
| Logistic regression | 0.7947 | 0.8859 |
| Extra Trees | 0.8073 | 0.8965 |
| Histogram gradient boosting | 0.8073 | 0.9010 |
| **Blended ensemble** | **0.8119** | **0.9032** |

Selected ensemble configuration:

| Setting | Value |
|---|---:|
| Logistic regression weight | 0.2 |
| Extra Trees weight | 0.3 |
| Histogram gradient boosting weight | 0.5 |
| Classification threshold | 0.490 |
| Test predictions labelled `True` | 52.72% |

The ensemble improves OOF accuracy by about 0.46 percentage points over either tree model alone. The exact fold scores, execution times, environment versions, weights, and threshold are stored in `outputs/metrics.json` and summarized in `outputs/model_summary.md`.

These are **local cross-validation results**, not a Kaggle public-leaderboard score. A leaderboard score is only available after uploading the generated submission.

---

## 5. Reproducing the Results

### Requirements

- Python 3.11 or 3.12
- Approximately 1 GB of free memory
- The Kaggle competition CSV files

| Package | Purpose |
|---|---|
| `pandas` | data loading and feature construction |
| `numpy` | numerical operations and probability blending |
| `scikit-learn` | preprocessing, grouped validation, models, and metrics |

Versions are constrained in `requirements.txt`; the recorded final run used Python 3.12.14, pandas 3.0.1, NumPy 2.5.4, and scikit-learn 1.7.2.

### Step 1: Clone the repository

```bash
git clone https://github.com/temper-zhong/spaceship-titanic-software-project.git
cd spaceship-titanic-software-project
```

The repository is currently private, so the cloning account must be added as a collaborator.

### Step 2: Create the environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3: Get the data

Download the competition files from the [Kaggle data page](https://www.kaggle.com/competitions/spaceship-titanic/data) and place them in `raw data/` with these exact names:

```text
raw data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Expected shapes:

| File | Expected shape |
|---|---:|
| `train.csv` | 8,693 × 14 |
| `test.csv` | 4,277 × 13 |
| `sample_submission.csv` | 4,277 × 2 |

### Step 4: Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests verify official file shapes, submission-column order, passenger-ID alignment, feature row preservation, feature-column order, and the absence of missing numeric model features.

### Step 5: Run a quick smoke test

```bash
python -m src.train --quick --output-dir outputs-quick
```

Quick mode uses three folds and smaller estimators. It is intended to verify installation and data placement, not to reproduce the committed final score.

### Step 6: Run the final pipeline

```bash
python -m src.train
```

The command trains all three models across five grouped folds, creates the ensemble, and writes:

| Output | Description |
|---|---|
| `outputs/submission.csv` | Kaggle-ready Boolean predictions |
| `outputs/metrics.json` | full run metadata and evaluation results |
| `outputs/model_summary.md` | human-readable model comparison |

The final submission contract is checked automatically:

- exactly 4,277 data rows;
- columns are exactly `PassengerId,Transported`;
- passenger IDs and row order match `sample_submission.csv`;
- IDs are unique;
- predictions are Boolean values.

### Command-line options

```text
python -m src.train [--data-dir PATH] [--output-dir PATH]
                    [--folds N] [--seed N] [--quick]
```

| Option | Default | Description |
|---|---:|---|
| `--data-dir` | `raw data` | directory containing the three Kaggle CSV files |
| `--output-dir` | `outputs` | directory for predictions and metrics |
| `--folds` | `5` | number of grouped validation folds |
| `--seed` | `42` | random seed used by splitting and models |
| `--quick` | off | use three folds and smaller estimators |

Changing the number of folds, seed, package versions, or source data can change the reported result. Keep the defaults when comparing against the committed metrics.

---

## 6. Collaboration Workflow

For team development:

```bash
git checkout -b feature/short-description
# make and test changes
git add .
git commit -m "Describe the change"
git push -u origin feature/short-description
```

Open a pull request into `main` and include:

1. the motivation for the change;
2. the validation command used;
3. before/after OOF accuracy if modelling logic changed;
4. confirmation that Kaggle data and generated submissions were not committed.

Do not compare experiments created with different fold settings as though they used the same validation contract.

---

## 7. Limitations and Next Steps

- OOF threshold selection can slightly overfit the current folds.
- Local grouped CV is intentionally strict and may not match the Kaggle leaderboard distribution.
- Manifest-level statistics use the supplied train and test passenger lists, although they never use test labels.
- The current model family is limited to scikit-learn; CatBoost and repeated-seed ensembles are promising extensions.
- A public or private leaderboard score has not yet been recorded in the repository.

Recommended next experiments are repeated grouped CV across several seeds, CatBoost with native categorical handling, probability calibration, feature ablation, and a documented Kaggle submission log.
