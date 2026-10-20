"""Run data-quality checks, exploratory analysis, and chart generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.pipeline import SPEND_COLUMNS, load_official_data, passenger_groups


PALETTE = {False: "#355C7D", True: "#C06C84"}


def data_dictionary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in test.columns:
        numeric = pd.api.types.is_numeric_dtype(train[column])
        non_missing = train[column].dropna()
        q1 = float(non_missing.quantile(0.25)) if numeric and len(non_missing) else np.nan
        q3 = float(non_missing.quantile(0.75)) if numeric and len(non_missing) else np.nan
        if numeric and len(non_missing):
            iqr = q3 - q1
            outliers = int(((non_missing < q1 - 1.5 * iqr) | (non_missing > q3 + 1.5 * iqr)).sum())
        else:
            outliers = 0
        rows.append(
            {
                "Feature": column,
                "Type": "numeric" if numeric else "categorical/text",
                "TrainMissingCount": int(train[column].isna().sum()),
                "TrainMissingRate": float(train[column].isna().mean()),
                "TestMissingCount": int(test[column].isna().sum()),
                "TestMissingRate": float(test[column].isna().mean()),
                "TrainUnique": int(train[column].nunique(dropna=True)),
                "TestUnique": int(test[column].nunique(dropna=True)),
                "TrainMin": float(non_missing.min()) if numeric and len(non_missing) else np.nan,
                "TrainMedian": float(non_missing.median()) if numeric and len(non_missing) else np.nan,
                "TrainMax": float(non_missing.max()) if numeric and len(non_missing) else np.nan,
                "IQR_OutlierCount": outliers,
            }
        )
    return pd.DataFrame(rows)


def anomaly_checks(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [train.drop(columns="Transported"), test], ignore_index=True
    )
    total_spend = combined[SPEND_COLUMNS].sum(axis=1, min_count=1)
    checks = [
        ("duplicate_train_rows", int(train.duplicated().sum()), "expected 0"),
        ("duplicate_test_rows", int(test.duplicated().sum()), "expected 0"),
        (
            "duplicate_passenger_ids",
            int(combined["PassengerId"].duplicated().sum()),
            "expected 0",
        ),
        (
            "invalid_passenger_id_format",
            int((~combined["PassengerId"].str.fullmatch(r"\d{4}_\d{2}").fillna(False)).sum()),
            "expected 0",
        ),
        (
            "invalid_cabin_format",
            int(
                (~combined["Cabin"].dropna().str.fullmatch(r"[A-Z]/\d+/[PS]")).sum()
            ),
            "expected 0",
        ),
        (
            "negative_age",
            int(combined["Age"].lt(0).fillna(False).sum()),
            "expected 0",
        ),
        (
            "age_above_100",
            int(combined["Age"].gt(100).fillna(False).sum()),
            "review if nonzero",
        ),
        (
            "negative_spend_values",
            int(combined[SPEND_COLUMNS].lt(0).sum().sum()),
            "expected 0",
        ),
        (
            "cryo_sleep_with_positive_spend",
            int((combined["CryoSleep"].eq(True) & total_spend.gt(0)).sum()),
            "expected 0 under domain constraint",
        ),
    ]
    return pd.DataFrame(checks, columns=["Check", "Count", "Interpretation"])


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(train: pd.DataFrame, test: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    missing = pd.DataFrame(
        {
            "Train": train.drop(columns="Transported").isna().mean() * 100,
            "Test": test.isna().mean() * 100,
        }
    ).sort_values("Train", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    missing.plot.barh(ax=ax, color=["#355C7D", "#C06C84"], width=0.75)
    ax.set_title("Missing values by feature")
    ax.set_xlabel("Missing rows (%)")
    ax.set_ylabel("")
    ax.legend(frameon=False)
    _save_figure(fig, figure_dir / "missingness.png")

    target_counts = train["Transported"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(
        ["Not transported", "Transported"],
        target_counts.values,
        color=[PALETTE[False], PALETTE[True]],
    )
    ax.set_title("Training target distribution")
    ax.set_ylabel("Passengers")
    ax.bar_label(bars, labels=[f"{value:,}" for value in target_counts.values], padding=3)
    ax.set_ylim(0, target_counts.max() * 1.14)
    _save_figure(fig, figure_dir / "target_distribution.png")

    chart_data = train.copy()
    cabin = chart_data["Cabin"].astype("string").str.split("/", n=2, expand=True)
    chart_data["CabinDeck"] = cabin[0]
    chart_data["CabinSide"] = cabin[2]
    categorical_columns = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, column in zip(axes.flat, categorical_columns, strict=True):
        rates = (
            chart_data.assign(**{column: chart_data[column].astype("string").fillna("Missing")})
            .groupby(column, observed=False)["Transported"]
            .agg(["mean", "size"])
            .sort_values("mean", ascending=False)
        )
        bars = ax.bar(rates.index.astype(str), rates["mean"] * 100, color="#6C5B7B")
        ax.set_title(f"Transport rate by {column}")
        ax.set_ylabel("Transported (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=25)
        for bar, count in zip(bars, rates["size"], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    _save_figure(fig, figure_dir / "categorical_transport_rates.png")

    numeric_data = train.copy()
    numeric_data["TotalSpend"] = numeric_data[SPEND_COLUMNS].sum(axis=1, min_count=1)
    numeric_data["LogTotalSpend"] = np.log1p(numeric_data["TotalSpend"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.histplot(
        data=numeric_data,
        x="Age",
        hue="Transported",
        bins=35,
        stat="density",
        common_norm=False,
        palette=PALETTE,
        element="step",
        ax=axes[0],
    )
    axes[0].set_title("Age distribution by target")
    sns.histplot(
        data=numeric_data,
        x="LogTotalSpend",
        hue="Transported",
        bins=35,
        stat="density",
        common_norm=False,
        palette=PALETTE,
        element="step",
        ax=axes[1],
    )
    axes[1].set_title("Log total spending by target")
    axes[1].set_xlabel("log(1 + total spend)")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "numeric_distributions.png")

    group = passenger_groups(train["PassengerId"])
    group_size = pd.Series(group).map(pd.Series(group).value_counts()).to_numpy()
    group_data = pd.DataFrame(
        {"GroupSize": group_size, "Transported": train["Transported"].astype(float)}
    )
    group_rates = group_data.groupby("GroupSize")["Transported"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(group_rates.index.astype(str), group_rates["mean"] * 100, color="#355C7D")
    ax.set_title("Transport rate by travel-party size")
    ax.set_xlabel("Passengers in group")
    ax.set_ylabel("Transported (%)")
    ax.set_ylim(0, 100)
    for bar, count in zip(bars, group_rates["size"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"n={count}",
            ha="center",
            fontsize=8,
        )
    _save_figure(fig, figure_dir / "group_size_transport_rate.png")

    correlation_data = numeric_data[["Age", *SPEND_COLUMNS, "TotalSpend", "Transported"]]
    correlation = correlation_data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title("Numeric feature correlations")
    _save_figure(fig, figure_dir / "correlation_heatmap.png")


def build_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    raw_total_spend = train[SPEND_COLUMNS].sum(axis=1, min_count=1)
    cabin = train["Cabin"].astype("string").str.split("/", n=2, expand=True)
    deck = cabin[0]
    side = cabin[2]
    group = pd.Series(passenger_groups(train["PassengerId"]), index=train.index)
    group_size = group.map(group.value_counts())

    def rate(mask: pd.Series) -> float:
        return float(train.loc[mask.fillna(False), "Transported"].mean())

    summary = {
        "train_shape": [int(x) for x in train.shape],
        "test_shape": [int(x) for x in test.shape],
        "target_true_rate": float(train["Transported"].mean()),
        "complete_train_rows": int(train.drop(columns="Transported").notna().all(axis=1).sum()),
        "complete_train_rate": float(train.drop(columns="Transported").notna().all(axis=1).mean()),
        "unique_travel_groups": int(group.nunique()),
        "largest_travel_group": int(group_size.max()),
        "key_rates": {
            "cryo_sleep_true": rate(train["CryoSleep"].eq(True)),
            "cryo_sleep_false": rate(train["CryoSleep"].eq(False)),
            "zero_spend": rate(raw_total_spend.eq(0)),
            "positive_spend": rate(raw_total_spend.gt(0)),
            "child_age_12_or_less": rate(train["Age"].le(12)),
            "adult_age_over_12": rate(train["Age"].gt(12)),
            "cabin_side_s": rate(side.eq("S")),
            "cabin_side_p": rate(side.eq("P")),
            "solo_traveller": rate(group_size.eq(1)),
            "group_traveller": rate(group_size.gt(1)),
        },
        "deck_transport_rates": {
            str(key): float(value)
            for key, value in train.assign(CabinDeck=deck)
            .groupby("CabinDeck", observed=False)["Transported"]
            .mean()
            .sort_values(ascending=False)
            .items()
        },
    }
    return summary


def write_eda_report(
    summary: dict[str, Any], quality: pd.DataFrame, anomalies: pd.DataFrame, path: Path
) -> None:
    missing_rows = quality.sort_values("TrainMissingRate", ascending=False).head(5)
    missing_table = "\n".join(
        f"| {row.Feature} | {row.TrainMissingCount:,} | {row.TrainMissingRate:.2%} | "
        f"{row.TestMissingCount:,} | {row.TestMissingRate:.2%} |"
        for row in missing_rows.itertuples()
    )
    anomaly_table = "\n".join(
        f"| {row.Check} | {row.Count:,} | {row.Interpretation} |"
        for row in anomalies.itertuples()
    )
    rates = summary["key_rates"]
    deck_text = ", ".join(
        f"{deck}={rate:.1%}" for deck, rate in summary["deck_transport_rates"].items()
    )
    content = f"""# Exploratory Data Analysis

## Data scope

- Training data: {summary['train_shape'][0]:,} rows and {summary['train_shape'][1]} columns.
- Test data: {summary['test_shape'][0]:,} rows and {summary['test_shape'][1]} columns.
- Positive target rate: {summary['target_true_rate']:.2%}.
- Complete training rows: {summary['complete_train_rows']:,} ({summary['complete_train_rate']:.2%}).
- Travel groups: {summary['unique_travel_groups']:,}; largest group: {summary['largest_travel_group']} passengers.

## Highest missing rates

| Feature | Train missing | Train rate | Test missing | Test rate |
|---|---:|---:|---:|---:|
{missing_table}

Missingness is spread across every predictor except `PassengerId`. Rates are low and similar between train and test, so there is no obvious train/test missingness shift.

## Data-quality and anomaly checks

| Check | Count | Interpretation |
|---|---:|---|
{anomaly_table}

IQR outlier counts are included in `outputs/data_quality.csv`. High spending values are retained because they are plausible passenger charges and tree models can use them without arbitrary clipping.

## Main findings

1. Cryogenic sleep is the strongest raw segment: transported rates are {rates['cryo_sleep_true']:.1%} for sleeping passengers and {rates['cryo_sleep_false']:.1%} for awake passengers.
2. Spending separates outcomes: zero-spend passengers have a {rates['zero_spend']:.1%} transport rate versus {rates['positive_spend']:.1%} for passengers with positive recorded spending.
3. Children aged 12 or below have a {rates['child_age_12_or_less']:.1%} transport rate versus {rates['adult_age_over_12']:.1%} for older passengers.
4. Cabin side matters: side S has a {rates['cabin_side_s']:.1%} rate versus {rates['cabin_side_p']:.1%} for side P.
5. Group travel matters: solo passengers have a {rates['solo_traveller']:.1%} rate versus {rates['group_traveller']:.1%} for passengers in multi-person groups.
6. Deck-level rates are heterogeneous: {deck_text}.

These findings motivate the cryosleep, spending, cabin, age, and travel-party features used by the modelling pipeline.

## Figures

- `reports/figures/missingness.png`
- `reports/figures/target_distribution.png`
- `reports/figures/categorical_transport_rates.png`
- `reports/figures/numeric_distributions.png`
- `reports/figures/group_size_transport_rate.png`
- `reports/figures/correlation_heatmap.png`
"""
    path.write_text(content, encoding="utf-8")


def run_eda(args: argparse.Namespace) -> dict[str, Any]:
    train, test, _ = load_official_data(args.data_dir.resolve())
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    figure_dir = report_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    quality = data_dictionary(train, test)
    anomalies = anomaly_checks(train, test)
    summary = build_summary(train, test)
    quality.to_csv(output_dir / "data_quality.csv", index=False)
    anomalies.to_csv(output_dir / "anomaly_checks.csv", index=False)
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    create_figures(train, test, figure_dir)
    write_eda_report(summary, quality, anomalies, report_dir / "EDA_REPORT.md")
    print(f"EDA report: {report_dir / 'EDA_REPORT.md'}")
    print(f"Figures: {figure_dir}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("raw data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run_eda(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
