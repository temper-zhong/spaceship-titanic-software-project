# Spaceship Titanic Team Project Overview

## Project status

The modelling project is complete and reproducible. Presentation slides are intentionally postponed. The repository now contains the shared preprocessing code, all four required models, grouped cross-validation, parameter search, model comparison, interpretation, subgroup error analysis, reports, charts, tests and a Kaggle-ready submission generator.

## Work allocation and deliverables

| Member | Responsibility | Main files and outputs |
|---|---|---|
| M1 | Problem understanding, data checks and EDA | `src/eda.py`, `reports/EDA_REPORT.md`, `outputs/data_quality.csv`, `outputs/anomaly_checks.csv`, EDA figures |
| M2 | Preprocessing, features, fixed split/CV and baseline | `src/pipeline.py`, Dummy and Logistic Regression baselines in `src/train.py`, grouped fold assignments and preprocessing tests |
| M3 | Random Forest | Tuning/training code, OOF metrics, fold scores and importance records |
| M4 | XGBoost | Tuning/training code, OOF metrics, fold scores and importance records |
| M5 | CatBoost | Native categorical tuning/training, OOF metrics, fold scores and importance records |
| M6 | Comparison, interpretation and error analysis | `src/analyze.py`, `reports/MODEL_ANALYSIS.md`, comparison/confusion/error tables and figures |

Every final model uses the same 44 features and the same five grouped folds. This makes the comparison fair and prevents travel-group leakage.

## Final local results

| Model | Grouped OOF accuracy | ROC-AUC |
|---|---:|---:|
| Dummy Classifier | 0.4945 | 0.5000 |
| Logistic Regression | 0.7945 | 0.8859 |
| Random Forest | 0.8026 | 0.8948 |
| XGBoost | 0.8100 | 0.9035 |
| CatBoost | 0.8134 | 0.9049 |
| **XGBoost/CatBoost blend** | **0.8158** | **0.9054** |

The final blend uses 40% XGBoost and 60% CatBoost probabilities with a threshold of 0.448. These are local cross-validation values, not Kaggle leaderboard scores.

## How a team member can reproduce the project

```bash
git clone https://github.com/temper-zhong/spaceship-titanic-software-project.git
cd spaceship-titanic-software-project
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.run_all
```

Before the last command, download the three Kaggle CSV files into `raw data/`. A faster installation check is available with:

```bash
python -m src.run_all --quick --output-dir outputs-quick --report-dir reports-quick
```

## Handoff rules

- Do not commit the Kaggle source data, generated OOF rows or `submission.csv`.
- Create a short-lived branch for model or documentation changes.
- Keep seed `42` and the committed fold design for direct comparisons.
- Run the unit tests before pushing.
- Report both mean accuracy and fold variability when changing a model.
- Record any Kaggle score together with the Git commit and submission filename.

See `README.md` for the full methodology and `REPORT.md` for the concise academic report.
