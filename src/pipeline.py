"""Shared data, feature, split, preprocessing, and model utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


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

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "catboost",
]

DEFAULT_MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "C": 0.55,
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "xgboost": {
        "n_estimators": 450,
        "max_depth": 4,
        "learning_rate": 0.04,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 4.0,
        "reg_alpha": 0.1,
    },
    "catboost": {
        "iterations": 450,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
    },
}


def validate_source_files(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> None:
    """Validate the official input contract before any modelling work."""

    required = {
        "PassengerId",
        "HomePlanet",
        "CryoSleep",
        "Cabin",
        "Destination",
        "Age",
        "VIP",
        *SPEND_COLUMNS,
        "Name",
    }
    missing_train = (required | {"Transported"}).difference(train.columns)
    missing_test = required.difference(test.columns)
    if missing_train:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"test.csv is missing required columns: {sorted(missing_test)}")
    if list(sample.columns) != ["PassengerId", "Transported"]:
        raise ValueError("sample_submission.csv must contain PassengerId and Transported.")
    if len(sample) != len(test) or not sample["PassengerId"].equals(test["PassengerId"]):
        raise ValueError("Sample submission IDs do not exactly match test.csv.")
    combined_ids = pd.concat([train["PassengerId"], test["PassengerId"]], ignore_index=True)
    if combined_ids.isna().any() or combined_ids.duplicated().any():
        raise ValueError("PassengerId must be populated and unique across train and test.")
    if train["Transported"].isna().any():
        raise ValueError("Transported contains missing labels.")


def load_official_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample"])
    validate_source_files(train, test, sample)
    return train, test, sample


def passenger_groups(passenger_id: pd.Series) -> np.ndarray:
    return passenger_id.str.split("_", n=1).str[0].to_numpy()


def build_grouped_splits(
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(features, target, groups))
    for fit_index, validation_index in splits:
        fit_groups = set(groups[fit_index])
        validation_groups = set(groups[validation_index])
        if fit_groups.intersection(validation_groups):
            raise AssertionError("A passenger group appears in both train and validation.")
    return splits


def fold_assignment(
    passenger_ids: pd.Series, splits: list[tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    assignment = np.full(len(passenger_ids), -1, dtype=int)
    for fold_number, (_, validation_index) in enumerate(splits, start=1):
        assignment[validation_index] = fold_number
    if (assignment < 1).any():
        raise AssertionError("Every training row must have one validation fold.")
    return pd.DataFrame({"PassengerId": passenger_ids, "Fold": assignment})


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
    parts["CabinNumber"] = pd.to_numeric(parts["CabinNumber"], errors="coerce").astype(float)
    return parts


def engineer_features(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create model features using no target-derived information."""

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
    group_cabin_number = frame.groupby("PassengerGroup")["CabinNumber"].transform("median")
    frame["CabinNumber"] = frame["CabinNumber"].fillna(group_cabin_number)
    deck_fallback = {"Earth": "G", "Europa": "B", "Mars": "F"}
    frame["CabinDeck"] = frame["CabinDeck"].fillna(frame["HomePlanet"].map(deck_fallback))
    frame["CabinDeck"] = frame["CabinDeck"].fillna("Unknown")
    frame["CabinSide"] = frame["CabinSide"].fillna("Unknown")
    deck_cabin_number = frame.groupby("CabinDeck")["CabinNumber"].transform("median")
    frame["CabinNumber"] = frame["CabinNumber"].fillna(deck_cabin_number)
    frame["CabinNumber"] = frame["CabinNumber"].fillna(frame["CabinNumber"].median())

    group_age = frame.groupby("PassengerGroup")["Age"].transform("median")
    frame["Age"] = frame["Age"].fillna(group_age)
    planet_age = frame.groupby("HomePlanet")["Age"].transform("median")
    frame["Age"] = frame["Age"].fillna(planet_age).fillna(frame["Age"].median())

    age_bins = [-np.inf, 12, 17, 25, 35, 50, 65, np.inf]
    age_labels = ["0-12", "13-17", "18-25", "26-35", "36-50", "51-65", "66+"]
    frame["AgeBand"] = pd.cut(frame["Age"], bins=age_bins, labels=age_labels).astype("string")
    frame["IsChild"] = (frame["Age"] <= 12).astype(float)
    frame["IsYoungAdult"] = frame["Age"].between(13, 25).astype(float)

    cryo_mask = frame["CryoSleep"].eq(True)
    for column in SPEND_COLUMNS:
        frame.loc[cryo_mask & frame[column].isna(), column] = 0.0
        planet_spend = frame.groupby("HomePlanet")[column].transform("median")
        frame[column] = frame[column].fillna(planet_spend).fillna(0.0)

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
        frame[column] = frame[column].astype("string").fillna("Unknown").astype(str)

    features = frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].astype(float)
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise AssertionError(f"Engineered features still contain missing values: {missing}")
    return (
        features.iloc[:train_count].reset_index(drop=True),
        features.iloc[train_count:].reset_index(drop=True),
    )


def make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.3,
    )


def build_model(
    model_name: str,
    seed: int,
    quick: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Any:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    params = {**DEFAULT_MODEL_PARAMETERS[model_name], **(overrides or {})}

    if model_name == "logistic_regression":
        model = LogisticRegression(
            C=float(params["C"]),
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        )
        return Pipeline(
            [("preprocessor", make_preprocessor(scale_numeric=True)), ("model", model)]
        )

    if model_name == "random_forest":
        estimators = min(int(params["n_estimators"]), 180) if quick else int(params["n_estimators"])
        model = RandomForestClassifier(
            n_estimators=estimators,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline(
            [("preprocessor", make_preprocessor(scale_numeric=False)), ("model", model)]
        )

    if model_name == "xgboost":
        estimators = min(int(params["n_estimators"]), 180) if quick else int(params["n_estimators"])
        model = XGBClassifier(
            n_estimators=estimators,
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline(
            [("preprocessor", make_preprocessor(scale_numeric=False)), ("model", model)]
        )

    iterations = min(int(params["iterations"]), 180) if quick else int(params["iterations"])
    return CatBoostClassifier(
        iterations=iterations,
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_strength=float(params["random_strength"]),
        loss_function="Logloss",
        eval_metric="Accuracy",
        cat_features=CATEGORICAL_FEATURES,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        random_seed=seed,
    )


def extract_feature_importance(estimator: Any, model_name: str) -> pd.DataFrame:
    if model_name == "catboost":
        names = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        importance = np.asarray(estimator.get_feature_importance(), dtype=float)
    else:
        preprocessor = estimator.named_steps["preprocessor"]
        fitted_model = estimator.named_steps["model"]
        names = preprocessor.get_feature_names_out().tolist()
        if model_name == "logistic_regression":
            importance = np.abs(fitted_model.coef_[0]).astype(float)
        else:
            importance = np.asarray(fitted_model.feature_importances_, dtype=float)
    if len(names) != len(importance):
        raise AssertionError(f"Feature-name mismatch for {model_name}.")
    return pd.DataFrame(
        {"Model": model_name, "Feature": names, "Importance": importance}
    )

