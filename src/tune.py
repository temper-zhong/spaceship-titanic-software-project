"""Run a compact, reproducible grouped-CV hyperparameter search."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.pipeline import (
    MODEL_NAMES,
    build_grouped_splits,
    build_model,
    engineer_features,
    load_official_data,
    passenger_groups,
)


SEARCH_SPACE: dict[str, list[dict[str, Any]]] = {
    "logistic_regression": [
        {"C": 0.25},
        {"C": 0.55},
        {"C": 1.0},
    ],
    "random_forest": [
        {
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 400,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 400,
            "max_depth": 16,
            "min_samples_leaf": 1,
            "max_features": 0.7,
        },
    ],
    "xgboost": [
        {
            "n_estimators": 350,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 4.0,
            "reg_alpha": 0.1,
        },
        {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.04,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 4.0,
            "reg_alpha": 0.1,
        },
        {
            "n_estimators": 350,
            "max_depth": 5,
            "learning_rate": 0.035,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "reg_lambda": 5.0,
            "reg_alpha": 0.2,
        },
        {
            "n_estimators": 450,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.90,
            "colsample_bytree": 0.85,
            "reg_lambda": 6.0,
            "reg_alpha": 0.1,
        },
    ],
    "catboost": [
        {
            "iterations": 350,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
        },
        {
            "iterations": 400,
            "depth": 6,
            "learning_rate": 0.04,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
        },
        {
            "iterations": 450,
            "depth": 7,
            "learning_rate": 0.035,
            "l2_leaf_reg": 6.0,
            "random_strength": 0.5,
        },
        {
            "iterations": 400,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 7.0,
            "random_strength": 1.0,
        },
    ],
}


def run_tuning(args: argparse.Namespace) -> dict[str, Any]:
    train, test, _ = load_official_data(args.data_dir.resolve())
    features, _ = engineer_features(train.drop(columns="Transported"), test)
    target = train["Transported"].astype(bool).to_numpy()
    groups = passenger_groups(train["PassengerId"])
    splits = build_grouped_splits(features, target, groups, args.folds, args.seed)

    rows: list[dict[str, Any]] = []
    best_parameters: dict[str, dict[str, Any]] = {}
    best_scores: dict[str, float] = {}

    for model_name in MODEL_NAMES:
        candidates = SEARCH_SPACE[model_name]
        if args.quick:
            candidates = candidates[:2]
        candidate_summaries: list[tuple[float, int, dict[str, Any]]] = []
        print(f"\nTuning {model_name} ({len(candidates)} candidates)...")

        for candidate_number, params in enumerate(candidates, start=1):
            fold_scores = []
            candidate_start = time.perf_counter()
            for fold_number, (fit_index, validation_index) in enumerate(splits, start=1):
                estimator = build_model(
                    model_name,
                    seed=args.seed + fold_number,
                    quick=args.quick,
                    overrides=params,
                )
                fold_start = time.perf_counter()
                estimator.fit(features.iloc[fit_index], target[fit_index])
                probability = estimator.predict_proba(features.iloc[validation_index])[:, 1]
                accuracy = float(
                    accuracy_score(target[validation_index], probability >= 0.5)
                )
                fold_scores.append(accuracy)
                rows.append(
                    {
                        "Model": model_name,
                        "Candidate": candidate_number,
                        "Fold": fold_number,
                        "Accuracy": accuracy,
                        "ROC_AUC": float(
                            roc_auc_score(target[validation_index], probability)
                        ),
                        "Seconds": float(time.perf_counter() - fold_start),
                        "Parameters": json.dumps(params, sort_keys=True),
                    }
                )
            mean_accuracy = float(np.mean(fold_scores))
            candidate_summaries.append((mean_accuracy, candidate_number, params))
            print(
                f"  candidate {candidate_number}: mean accuracy={mean_accuracy:.4f}, "
                f"time={time.perf_counter() - candidate_start:.1f}s"
            )

        best_accuracy, best_candidate, best_params = max(
            candidate_summaries, key=lambda item: (item[0], -item[1])
        )
        best_parameters[model_name] = best_params
        best_scores[model_name] = best_accuracy
        print(f"  selected candidate {best_candidate}: {best_accuracy:.4f}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output_dir / "tuning_results.csv", index=False)
    payload = {
        "selection_metric": "mean grouped 3-fold accuracy at threshold 0.5",
        "folds": args.folds,
        "seed": args.seed,
        "best_parameters": best_parameters,
        "best_scores": best_scores,
    }
    (output_dir / "best_params.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("raw data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.folds < 3:
        parser.error("--folds must be at least 3")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        run_tuning(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

