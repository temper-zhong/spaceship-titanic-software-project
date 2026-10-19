"""Train and evaluate models for Kaggle's Spaceship Titanic competition.

The script performs feature engineering, leakage-resistant grouped cross-validation,
model comparison, probability blending, and submission generation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SPEND_COLUMNS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
]

CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "CabinDeck",
    "CabinSide",
    "AgeBand",
    "CabinRegion",
]

NUMERIC_FEATURES = [
    "Age",
    *SPEND_COLUMNS,
    *[f"Log{column}" for column in SPEND_COLUMNS],
    "CabinNumber",
    "GroupMemberNumber",
    "GroupSize",
    "SurnameSize",
    "CabinOccupancy",
    "MissingCount",
    "MissingServiceCount",
    "TotalSpend",
    "LogTotalSpend",
    "LuxurySpend",
    "FoodShoppingSpend",
    "ActiveServiceCount",
    "MaximumServiceSpend",
    "GroupMeanAge",
    "GroupMeanSpend",
    "GroupTotalSpend",
    "GroupCryoRate",
    "SurnameMeanSpend",
    "IsSolo",
    "IsChild",
    "IsYoungAdult",
    "NoSpend",
    "HasPositiveSpend",
    "AgeWasMissing",
    "CabinWasMissing",
]


@dataclass(frozen=True)
class ModelResult:
    name: str
    fold_accuracy: list[float]
    oof_accuracy: float
    oof_roc_auc: float
    seconds: float


def _mode_or_nan(series: pd.Series) -> Any:
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def _fill_with_group_mode(frame: pd.DataFrame, column: str, group: str) -> None:
    inferred = frame.groupby(group, dropna=False)[column].transform(_mode_or_nan)
    frame[column] = frame[column].fillna(inferred)


def _parse_cabin(cabin: pd.Series) -> pd.DataFrame:
    parts = cabin.astype("string").str.split("/", n=2, expand=True)
    parts = parts.reindex(columns=range(3))
    parts.columns = ["CabinDeck", "CabinNumber", "CabinSide"]
    parts["CabinNumber"] = pd.to_numeric(
        parts["CabinNumber"], errors="coerce"
    ).astype(float)
    return parts


def engineer_features(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create manifest-level features without using the target column.

    Train and test are combined only for unlabeled manifest counts and imputations.
    This mirrors the information available when Kaggle supplies the full passenger
    manifest and never uses validation/test labels.
    """

    train_count = len(train_features)
    frame = pd.concat([train_features.copy(), test_features.copy()], ignore_index=True)

    if frame["PassengerId"].duplicated().any():
        raise ValueError("PassengerId must be unique across train and test data.")

    original_missing = frame.isna()
    frame["MissingCount"] = original_missing.sum(axis=1).astype(float)
    frame["MissingServiceCount"] = original_missing[SPEND_COLUMNS].sum(axis=1).astype(float)
    frame["AgeWasMissing"] = original_missing["Age"].astype(float)
    frame["CabinWasMissing"] = original_missing["Cabin"].astype(float)

    passenger_parts = frame["PassengerId"].str.split("_", n=1, expand=True)
    frame["PassengerGroup"] = passenger_parts[0]
    frame["GroupMemberNumber"] = pd.to_numeric(passenger_parts[1], errors="coerce")
    frame["GroupSize"] = frame.groupby("PassengerGroup")["PassengerId"].transform("size")
    frame["IsSolo"] = (frame["GroupSize"] == 1).astype(float)

    surname = frame["Name"].astype("string").str.rsplit(n=1).str[-1]
    missing_surname = surname.isna()
    surname = surname.astype("object")
    surname.loc[missing_surname] = "Missing-" + frame.loc[missing_surname, "PassengerId"]
    frame["Surname"] = surname
    frame["SurnameSize"] = frame.groupby("Surname")["PassengerId"].transform("size")

    cabin_parts = _parse_cabin(frame["Cabin"])
    frame = pd.concat([frame, cabin_parts], axis=1)
    raw_cabin_key = frame["Cabin"].astype("string")
    missing_cabin = raw_cabin_key.isna()
    raw_cabin_key = raw_cabin_key.astype("object")
    raw_cabin_key.loc[missing_cabin] = "Missing-" + frame.loc[missing_cabin, "PassengerId"]
    frame["CabinOccupancy"] = raw_cabin_key.groupby(raw_cabin_key).transform("size").astype(float)

    # Logical constraints and within-party inference reduce arbitrary median fills.
    known_total = frame[SPEND_COLUMNS].sum(axis=1, min_count=1)
    has_positive_spend = frame[SPEND_COLUMNS].gt(0).any(axis=1)
    frame.loc[frame["CryoSleep"].isna() & has_positive_spend, "CryoSleep"] = False
    _fill_with_group_mode(frame, "CryoSleep", "PassengerGroup")
    frame.loc[frame["CryoSleep"].isna() & known_total.eq(0), "CryoSleep"] = True
    frame["CryoSleep"] = frame["CryoSleep"].fillna(False)

    _fill_with_group_mode(frame, "HomePlanet", "PassengerGroup")
    _fill_with_group_mode(frame, "HomePlanet", "Surname")
    deck_planet = {"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"}
    frame["HomePlanet"] = frame["HomePlanet"].fillna(frame["CabinDeck"].map(deck_planet))
    frame["HomePlanet"] = frame["HomePlanet"].fillna("Unknown")

    _fill_with_group_mode(frame, "Destination", "PassengerGroup")
    frame["Destination"] = frame["Destination"].fillna("Unknown")
    frame["VIP"] = frame["VIP"].fillna(False)

    _fill_with_group_mode(frame, "CabinDeck", "PassengerGroup")
    _fill_with_group_mode(frame, "CabinSide", "PassengerGroup")
    cabin_number_by_group = frame.groupby("PassengerGroup")["CabinNumber"].transform("median")
    frame["CabinNumber"] = frame["CabinNumber"].fillna(cabin_number_by_group)
    deck_fallback = {"Earth": "G", "Europa": "B", "Mars": "F"}
    frame["CabinDeck"] = frame["CabinDeck"].fillna(frame["HomePlanet"].map(deck_fallback))
    frame["CabinDeck"] = frame["CabinDeck"].fillna("Unknown")
    frame["CabinSide"] = frame["CabinSide"].fillna("Unknown")
    cabin_number_by_deck = frame.groupby("CabinDeck")["CabinNumber"].transform("median")
    frame["CabinNumber"] = frame["CabinNumber"].fillna(cabin_number_by_deck)
    frame["CabinNumber"] = frame["CabinNumber"].fillna(frame["CabinNumber"].median())

    age_by_group = frame.groupby("PassengerGroup")["Age"].transform("median")
    frame["Age"] = frame["Age"].fillna(age_by_group)
    age_by_planet = frame.groupby("HomePlanet")["Age"].transform("median")
    frame["Age"] = frame["Age"].fillna(age_by_planet).fillna(frame["Age"].median())

    age_bins = [-np.inf, 12, 17, 25, 35, 50, 65, np.inf]
    age_labels = ["0-12", "13-17", "18-25", "26-35", "36-50", "51-65", "66+"]
    frame["AgeBand"] = pd.cut(frame["Age"], bins=age_bins, labels=age_labels).astype("string")
    frame["IsChild"] = (frame["Age"] <= 12).astype(float)
    frame["IsYoungAdult"] = frame["Age"].between(13, 25).astype(float)

    cryo_mask = frame["CryoSleep"].eq(True)
    for column in SPEND_COLUMNS:
        frame.loc[cryo_mask & frame[column].isna(), column] = 0.0
        median_by_planet = frame.groupby("HomePlanet")[column].transform("median")
        frame[column] = frame[column].fillna(median_by_planet).fillna(0.0)

    frame["TotalSpend"] = frame[SPEND_COLUMNS].sum(axis=1)
    frame["LuxurySpend"] = frame[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
    frame["FoodShoppingSpend"] = frame[["FoodCourt", "ShoppingMall"]].sum(axis=1)
    frame["ActiveServiceCount"] = frame[SPEND_COLUMNS].gt(0).sum(axis=1).astype(float)
    frame["MaximumServiceSpend"] = frame[SPEND_COLUMNS].max(axis=1)
    frame["NoSpend"] = frame["TotalSpend"].eq(0).astype(float)
    frame["HasPositiveSpend"] = frame["TotalSpend"].gt(0).astype(float)
    frame["LogTotalSpend"] = np.log1p(frame["TotalSpend"])
    for column in SPEND_COLUMNS:
        frame[f"Log{column}"] = np.log1p(frame[column])

    region_bins = [-np.inf, 299, 599, 899, 1199, 1499, np.inf]
    region_labels = ["000-299", "300-599", "600-899", "900-1199", "1200-1499", "1500+"]
    frame["CabinRegion"] = pd.cut(
        frame["CabinNumber"], bins=region_bins, labels=region_labels
    ).astype("string")

    frame["GroupMeanAge"] = frame.groupby("PassengerGroup")["Age"].transform("mean")
    frame["GroupMeanSpend"] = frame.groupby("PassengerGroup")["TotalSpend"].transform("mean")
    frame["GroupTotalSpend"] = frame.groupby("PassengerGroup")["TotalSpend"].transform("sum")
    cryo_numeric = frame["CryoSleep"].astype(float)
    frame["GroupCryoRate"] = cryo_numeric.groupby(frame["PassengerGroup"]).transform("mean")
    frame["SurnameMeanSpend"] = frame.groupby("Surname")["TotalSpend"].transform("mean")

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").fillna("Unknown")

    features = frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    if features.shape[0] != train_count + len(test_features):
        raise AssertionError("Feature engineering changed the number of rows.")
    return features.iloc[:train_count].reset_index(drop=True), features.iloc[train_count:].reset_index(drop=True)


def make_preprocessor(*, dense: bool, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=not dense,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0 if dense else 0.3,
    )


def build_models(seed: int, quick: bool) -> dict[str, Pipeline]:
    tree_count = 180 if quick else 650
    boosting_iterations = 100 if quick else 300
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", make_preprocessor(dense=False, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        C=0.55,
                        max_iter=3000,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocessor", make_preprocessor(dense=False, scale_numeric=False)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=tree_count,
                        max_depth=16,
                        min_samples_leaf=2,
                        max_features=0.80,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", make_preprocessor(dense=True, scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.055,
                        max_iter=boosting_iterations,
                        max_leaf_nodes=23,
                        min_samples_leaf=22,
                        l2_regularization=2.0,
                        early_stopping=True,
                        validation_fraction=0.12,
                        n_iter_no_change=25,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def find_best_blend(
    oof_probabilities: dict[str, np.ndarray], target: np.ndarray
) -> tuple[dict[str, float], float, float]:
    names = list(oof_probabilities)
    best_score = -1.0
    best_threshold = 0.5
    best_weights = {name: 1.0 / len(names) for name in names}

    # A small, deterministic grid is enough for three complementary models.
    for first_weight in np.arange(0.0, 1.01, 0.1):
        for second_weight in np.arange(0.0, 1.01 - first_weight, 0.1):
            third_weight = 1.0 - first_weight - second_weight
            weights = [first_weight, second_weight, third_weight]
            probabilities = sum(
                weight * oof_probabilities[name] for name, weight in zip(names, weights)
            )
            for threshold in np.arange(0.44, 0.561, 0.002):
                score = accuracy_score(target, probabilities >= threshold)
                if score > best_score + 1e-12:
                    best_score = float(score)
                    best_threshold = float(threshold)
                    best_weights = {name: float(weight) for name, weight in zip(names, weights)}
    return best_weights, best_threshold, best_score


def write_model_summary(metrics: dict[str, Any], path: Path) -> None:
    rows = []
    for name, result in metrics["models"].items():
        fold_text = ", ".join(f"{score:.4f}" for score in result["fold_accuracy"])
        rows.append(
            f"| {name} | {result['oof_accuracy']:.4f} | {result['oof_roc_auc']:.4f} | {fold_text} |"
        )
    weights = ", ".join(
        f"{name}={weight:.1f}" for name, weight in metrics["ensemble"]["weights"].items()
    )
    content = f"""# Model results

Generated by `python -m src.train`.

| Model | Grouped OOF accuracy | OOF ROC-AUC | Fold accuracy |
|---|---:|---:|---|
{chr(10).join(rows)}
| blended ensemble | {metrics['ensemble']['oof_accuracy']:.4f} | {metrics['ensemble']['oof_roc_auc']:.4f} | - |

- Validation: {metrics['validation']['method']}, {metrics['validation']['folds']} folds
- Blend: {weights}
- Classification threshold: {metrics['ensemble']['threshold']:.3f}
- Test predictions marked `True`: {metrics['submission']['true_rate']:.2%}
"""
    path.write_text(content, encoding="utf-8")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    for path in [train_path, test_path, sample_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    required_train_columns = {"PassengerId", "Transported", *SPEND_COLUMNS}
    if missing := required_train_columns.difference(train.columns):
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")
    if list(sample.columns) != ["PassengerId", "Transported"]:
        raise ValueError("sample_submission.csv must contain PassengerId and Transported.")
    if len(sample) != len(test) or not sample["PassengerId"].equals(test["PassengerId"]):
        raise ValueError("Sample submission IDs do not exactly match test.csv.")

    target = train["Transported"].astype(bool).to_numpy()
    groups = train["PassengerId"].str.split("_", n=1).str[0].to_numpy()
    train_features, test_features = engineer_features(
        train.drop(columns="Transported"), test
    )

    fold_count = 3 if args.quick else args.folds
    splitter = StratifiedGroupKFold(
        n_splits=fold_count, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(train_features, target, groups))
    models = build_models(args.seed, args.quick)

    oof_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    model_results: dict[str, ModelResult] = {}

    for model_name, pipeline in models.items():
        start = time.perf_counter()
        oof = np.zeros(len(train_features), dtype=float)
        test_fold_probabilities = []
        fold_accuracy = []

        print(f"\nTraining {model_name}...")
        for fold_number, (fit_index, validation_index) in enumerate(splits, start=1):
            pipeline.fit(train_features.iloc[fit_index], target[fit_index])
            validation_probability = pipeline.predict_proba(
                train_features.iloc[validation_index]
            )[:, 1]
            oof[validation_index] = validation_probability
            test_fold_probabilities.append(pipeline.predict_proba(test_features)[:, 1])
            score = accuracy_score(target[validation_index], validation_probability >= 0.5)
            fold_accuracy.append(float(score))
            print(f"  fold {fold_number}/{fold_count}: accuracy={score:.4f}")

        elapsed = time.perf_counter() - start
        oof_accuracy = float(accuracy_score(target, oof >= 0.5))
        oof_auc = float(roc_auc_score(target, oof))
        oof_probabilities[model_name] = oof
        test_probabilities[model_name] = np.mean(test_fold_probabilities, axis=0)
        model_results[model_name] = ModelResult(
            name=model_name,
            fold_accuracy=fold_accuracy,
            oof_accuracy=oof_accuracy,
            oof_roc_auc=oof_auc,
            seconds=float(elapsed),
        )
        print(f"  OOF accuracy={oof_accuracy:.4f}, ROC-AUC={oof_auc:.4f}, time={elapsed:.1f}s")

    weights, threshold, blend_accuracy = find_best_blend(oof_probabilities, target)
    blended_oof = sum(weights[name] * oof_probabilities[name] for name in models)
    blended_test = sum(weights[name] * test_probabilities[name] for name in models)
    blend_auc = float(roc_auc_score(target, blended_oof))
    predictions = blended_test >= threshold

    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Transported": predictions.astype(bool)}
    )
    if submission.shape != sample.shape:
        raise AssertionError("Submission shape does not match sample_submission.csv.")
    if not submission["PassengerId"].equals(sample["PassengerId"]):
        raise AssertionError("Submission passenger order changed.")
    if submission["Transported"].dtype != bool:
        raise AssertionError("Submission predictions must be Boolean.")

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "model_summary.md"
    submission.to_csv(submission_path, index=False)

    metrics: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "engineered_features": int(train_features.shape[1]),
            "target_true_rate": float(target.mean()),
        },
        "validation": {
            "method": "StratifiedGroupKFold grouped by PassengerId prefix",
            "folds": fold_count,
            "seed": args.seed,
        },
        "models": {
            name: {
                "fold_accuracy": result.fold_accuracy,
                "oof_accuracy": result.oof_accuracy,
                "oof_roc_auc": result.oof_roc_auc,
                "seconds": result.seconds,
            }
            for name, result in model_results.items()
        },
        "ensemble": {
            "weights": weights,
            "threshold": threshold,
            "oof_accuracy": blend_accuracy,
            "oof_roc_auc": blend_auc,
        },
        "submission": {
            "path": str(submission_path),
            "rows": int(len(submission)),
            "true_rate": float(predictions.mean()),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_model_summary(metrics, summary_path)

    print("\nFinished.")
    print(f"Ensemble OOF accuracy: {blend_accuracy:.4f}")
    print(f"Submission: {submission_path}")
    print(f"Metrics: {metrics_path}")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("raw data"),
        help="Directory containing train.csv, test.csv, and sample_submission.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for submission.csv and evaluation artifacts.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of grouped CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 3 folds and smaller estimators for a fast smoke test.",
    )
    args = parser.parse_args(argv)
    if args.folds < 3:
        parser.error("--folds must be at least 3")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        run_training(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
