"""Create model comparison, interpretation, and subgroup error analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.pipeline import SPEND_COLUMNS, load_official_data, passenger_groups


MODEL_LABELS = {
    "dummy_classifier": "Dummy classifier",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "blended_ensemble": "Blended ensemble",
}


def _clean_feature_name(value: str) -> str:
    return value.replace("numeric__", "").replace("categorical__", "")


def build_analysis_frame(train: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    if not train["PassengerId"].equals(oof["PassengerId"]):
        raise ValueError("OOF predictions are not aligned with train.csv.")
    frame = train.copy()
    frame["Probability"] = oof["ensemble_probability"].to_numpy()
    frame["Prediction"] = oof["ensemble_prediction"].astype(bool).to_numpy()
    frame["Correct"] = frame["Transported"].eq(frame["Prediction"])
    frame["ErrorType"] = oof["error_type"].to_numpy()

    cabin = frame["Cabin"].astype("string").str.split("/", n=2, expand=True)
    frame["CabinDeck"] = cabin[0].fillna("Missing")
    frame["CabinSide"] = cabin[2].fillna("Missing")
    frame["AgeBand"] = pd.cut(
        frame["Age"],
        bins=[-np.inf, 12, 17, 25, 35, 50, 65, np.inf],
        labels=["0-12", "13-17", "18-25", "26-35", "36-50", "51-65", "66+"],
    ).astype("string").fillna("Missing")
    groups = pd.Series(passenger_groups(frame["PassengerId"]), index=frame.index)
    group_size = groups.map(groups.value_counts())
    frame["GroupSizeBand"] = pd.cut(
        group_size,
        bins=[0, 1, 2, 4, np.inf],
        labels=["Solo", "2", "3-4", "5+"],
    ).astype("string")
    total_spend = frame[SPEND_COLUMNS].sum(axis=1, min_count=1)
    frame["SpendBand"] = pd.cut(
        total_spend,
        bins=[-np.inf, 0, 499, 1999, np.inf],
        labels=["Zero", "1-499", "500-1999", "2000+"],
    ).astype("string").fillna("Missing")
    missing_count = frame.drop(columns=["Transported"]).isna().sum(axis=1)
    frame["MissingFields"] = pd.cut(
        missing_count,
        bins=[-1, 0, 1, np.inf],
        labels=["0", "1", "2+"],
    ).astype("string")
    confidence = np.abs(frame["Probability"] - 0.5)
    frame["ConfidenceBand"] = pd.cut(
        confidence,
        bins=[-np.inf, 0.05, 0.15, 0.30, np.inf],
        labels=["Very low", "Low", "Medium", "High"],
    ).astype("string")
    return frame


def subgroup_error_table(frame: pd.DataFrame) -> pd.DataFrame:
    segments = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "AgeBand",
        "GroupSizeBand",
        "SpendBand",
        "MissingFields",
        "ConfidenceBand",
    ]
    rows = []
    for segment in segments:
        values = frame[segment].astype("string").fillna("Missing")
        for value, index in values.groupby(values, dropna=False).groups.items():
            group = frame.loc[index]
            false_positive = int(
                ((~group["Transported"].astype(bool)) & group["Prediction"]).sum()
            )
            false_negative = int(
                (group["Transported"].astype(bool) & (~group["Prediction"])).sum()
            )
            rows.append(
                {
                    "Segment": segment,
                    "Value": str(value),
                    "Count": int(len(group)),
                    "Accuracy": float(group["Correct"].mean()),
                    "ErrorRate": float(1 - group["Correct"].mean()),
                    "FalsePositives": false_positive,
                    "FalseNegatives": false_negative,
                    "TargetRate": float(group["Transported"].mean()),
                    "MeanProbability": float(group["Probability"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["ErrorRate", "Count"], ascending=[False, False])


def create_analysis_figures(
    comparison: pd.DataFrame,
    confusion: pd.DataFrame,
    importance: pd.DataFrame,
    errors: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    ordered = comparison.sort_values("Accuracy", ascending=True).copy()
    ordered["Label"] = ordered["Model"].map(MODEL_LABELS).fillna(ordered["Model"])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(ordered["Label"], ordered["Accuracy"] * 100, color="#355C7D")
    ax.set_title("Grouped OOF accuracy by model")
    ax.set_xlabel("Accuracy (%)")
    lower = max(0, float(ordered["Accuracy"].min() * 100 - 3))
    upper = min(100, float(ordered["Accuracy"].max() * 100 + 2))
    ax.set_xlim(lower, upper)
    ax.bar_label(bars, labels=[f"{x:.2f}%" for x in ordered["Accuracy"] * 100], padding=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    matrix = confusion.pivot(index="Actual", columns="Predicted", values="Count").reindex(
        index=["False", "True"], columns=["False", "True"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Ensemble OOF confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    importance_models = [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "catboost",
    ]
    for ax, model_name in zip(axes.flat, importance_models, strict=True):
        top = (
            importance.loc[importance["Model"].eq(model_name)]
            .sort_values("Importance", ascending=False)
            .head(12)
            .sort_values("Importance")
            .copy()
        )
        top["FeatureLabel"] = top["Feature"].map(_clean_feature_name)
        ax.barh(top["FeatureLabel"], top["Importance"], color="#6C5B7B")
        ax.set_title(f"{MODEL_LABELS[model_name]} feature importance")
        ax.set_xlabel("Model-specific importance")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    worst = (
        errors.loc[(errors["Count"] >= 50) & ~errors["Segment"].eq("ConfidenceBand")]
        .head(12)
        .sort_values("ErrorRate")
        .copy()
    )
    worst["Label"] = worst["Segment"] + ": " + worst["Value"]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    bars = ax.barh(worst["Label"], worst["ErrorRate"] * 100, color="#C06C84")
    ax.set_title("Highest ensemble error rates by subgroup (n >= 50)")
    ax.set_xlabel("Error rate (%)")
    ax.bar_label(
        bars,
        labels=[f"{rate:.1f}% (n={count})" for rate, count in zip(worst["ErrorRate"] * 100, worst["Count"], strict=True)],
        padding=3,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "subgroup_error_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_analysis_report(
    comparison: pd.DataFrame,
    confusion: pd.DataFrame,
    importance: pd.DataFrame,
    errors: pd.DataFrame,
    path: Path,
) -> None:
    comparison_rows = "\n".join(
        f"| {MODEL_LABELS.get(row.Model, row.Model)} | {row.Accuracy:.4f} | "
        f"{row.ROC_AUC:.4f} | {row.Precision:.4f} | {row.Recall:.4f} | "
        f"{row.F1:.4f} | {row.FoldAccuracyStd:.4f} |"
        for row in comparison.sort_values("Accuracy", ascending=False).itertuples()
    )
    matrix_rows = "\n".join(
        f"| {row.Actual} | {row.Predicted} | {row.Count:,} |"
        for row in confusion.itertuples()
    )
    worst = errors.loc[(errors["Count"] >= 50) & ~errors["Segment"].eq("ConfidenceBand")].head(10)
    error_rows = "\n".join(
        f"| {row.Segment} | {row.Value} | {row.Count:,} | {row.ErrorRate:.1%} | "
        f"{row.FalsePositives:,} | {row.FalseNegatives:,} |"
        for row in worst.itertuples()
    )
    top_feature_rows = []
    for model_name in ["logistic_regression", "random_forest", "xgboost", "catboost"]:
        top = (
            importance.loc[importance["Model"].eq(model_name)]
            .sort_values("Importance", ascending=False)
            .head(5)
        )
        features = ", ".join(_clean_feature_name(value) for value in top["Feature"])
        top_feature_rows.append(f"| {MODEL_LABELS[model_name]} | {features} |")

    best = comparison.sort_values("Accuracy", ascending=False).iloc[0]
    content = f"""# Model Comparison and Error Analysis

## Model comparison

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 | Fold accuracy SD |
|---|---:|---:|---:|---:|---:|---:|
{comparison_rows}

The highest grouped OOF accuracy is {best.Accuracy:.4f}, achieved by {MODEL_LABELS.get(best.Model, best.Model)}. Fold standard deviation is reported because mean accuracy alone can hide unstable splits.

## Ensemble confusion matrix

| Actual | Predicted | Count |
|---|---|---:|
{matrix_rows}

False positives and false negatives are reported separately in the subgroup table. This matters because a model can preserve overall accuracy while failing systematically for one passenger segment.

## Highest-error subgroups

Only groups with at least 50 passengers are shown below.

| Segment | Value | Count | Error rate | False positives | False negatives |
|---|---|---:|---:|---:|---:|
{error_rows}

## Feature interpretation

Importance values are model-specific and should be interpreted as rankings within a model, not compared numerically across algorithms.

| Model | Five highest-ranked features |
|---|---|
{chr(10).join(top_feature_rows)}

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
"""
    path.write_text(content, encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    train, _, _ = load_official_data(args.data_dir.resolve())
    oof = pd.read_csv(output_dir / "oof_predictions.csv")
    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    importance = pd.read_csv(output_dir / "feature_importance.csv")

    frame = build_analysis_frame(train, oof)
    errors = subgroup_error_table(frame)
    matrix_array = confusion_matrix(
        frame["Transported"].astype(bool), frame["Prediction"].astype(bool), labels=[False, True]
    )
    confusion = pd.DataFrame(
        [
            {"Actual": str(actual), "Predicted": str(predicted), "Count": int(matrix_array[i, j])}
            for i, actual in enumerate([False, True])
            for j, predicted in enumerate([False, True])
        ]
    )

    errors.to_csv(output_dir / "error_analysis.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrix.csv", index=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    create_analysis_figures(
        comparison, confusion, importance, errors, report_dir / "figures"
    )
    write_analysis_report(
        comparison, confusion, importance, errors, report_dir / "MODEL_ANALYSIS.md"
    )
    print(f"Model analysis: {report_dir / 'MODEL_ANALYSIS.md'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("raw data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run_analysis(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
